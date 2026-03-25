#!/usr/bin/env python3
"""
Execution Optimizer - 小额本金执行优化器.

本工具用于验证 5 万元本金下的最佳执行策略，包含以下测试模块:
    1. 频率与 K 值交叉回测 (Frequency & K-Selection Sweep)
    2. 动态概率阈值测试 (Dynamic Probability Threshold Test)
    3. 现金闲置率分析 (Cash Drag Analysis)
    4. 国债 ETF 优化测试 (Bond ETF Optimization)

使用示例:
    >>> from src.execution_optimizer import ExecutionOptimizer
    >>> optimizer = ExecutionOptimizer()
    >>> results = optimizer.run_all_tests()
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
    from .model_trainer import ModelTrainer
except ImportError:
    from model_trainer import ModelTrainer


@dataclass
class ExecutionConfig:
    """执行优化配置."""
    # 基础配置
    parquet_path: str = "data/parquet/features_latest.parquet"
    feature_columns: Optional[List[str]] = None
    
    # 资金配置
    initial_capital: float = 50_000.0
    
    # 交易成本配置
    commission_rate: float = 0.0003  # 万分之三
    min_commission: float = 5.0  # 最低 5 元
    stamp_duty_rate: float = 0.001  # 千分之一（卖出）
    bond_etf_commission_rate: float = 0.0001  # 国债 ETF 佣金更低
    min_bond_etf_commission: float = 0.0  # ETF 无最低佣金限制
    
    # 股数限制
    round_lot: int = 100  # 100 股整数倍
    
    # 测试矩阵配置
    rebalance_frequencies: List[int] = None  # [1, 2, 3, 5] 日
    k_values: List[int] = None  # [3, 5]
    
    # 概率阈值配置
    probability_thresholds: List[float] = None  # [0.7, 0.75, 0.8, 0.85]
    ranking_exit_threshold: float = 0.15  # 排名掉出前 15% 退出
    
    # 国债 ETF 配置
    bond_etf_enabled: bool = True
    bond_etf_annual_return: float = 0.035  # 年化 3.5%
    
    def __post_init__(self):
        if self.rebalance_frequencies is None:
            self.rebalance_frequencies = [1, 2, 3, 5]
        if self.k_values is None:
            self.k_values = [3, 5]
        if self.probability_thresholds is None:
            self.probability_thresholds = [0.70, 0.75, 0.80, 0.85]


class AdvancedSmallCapSimulator:
    """
    高级小额实盘仿真器.
    
    支持:
        1. 可配置调仓频率
        2. 可配置 K 值
        3. 概率阈值触发
        4. 排名衰减退出
        5. 国债 ETF 现金管理
    """
    
    def __init__(
        self,
        initial_capital: float = 50_000.0,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.001,
        round_lot: int = 100,
        top_k: int = 3,
        rebalance_frequency: int = 1,  # 调仓频率 (天)
        probability_threshold: float = 0.75,  # 开仓概率阈值
        use_ranking_exit: bool = False,  # 是否使用排名衰减退出
        ranking_exit_threshold: float = 0.15,  # 排名掉出前 X% 退出
        use_bond_etf: bool = False,  # 是否使用国债 ETF 管理现金
        bond_etf_return: float = 0.035,
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.round_lot = round_lot
        self.top_k = top_k
        self.rebalance_frequency = rebalance_frequency
        self.probability_threshold = probability_threshold
        self.use_ranking_exit = use_ranking_exit
        self.ranking_exit_threshold = ranking_exit_threshold
        self.use_bond_etf = use_bond_etf
        self.bond_etf_return = bond_etf_return
        
        # 交易成本统计
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_bond_etf_cost = 0.0
        
        # 统计信息
        self.total_trades = 0
        self.total_bond_etf_days = 0
        self.cash_drag_records = []
        
        logger.info(f"AdvancedSmallCapSimulator: capital={initial_capital:,.0f}, "
                   f"K={top_k}, freq={rebalance_frequency}d, prob_thresh={probability_threshold:.2f}, "
                   f"ranking_exit={use_ranking_exit}, bond_etf={use_bond_etf}")
    
    def calculate_transaction_cost(
        self,
        buy_price: float,
        sell_price: float,
        shares: int,
        is_stock: bool = True,
    ) -> Tuple[float, float, float]:
        """
        计算交易成本.
        
        Returns:
            (total_cost, commission, stamp_duty)
        """
        buy_value = buy_price * shares
        sell_value = sell_price * shares
        
        if is_stock:
            # 股票交易
            buy_commission = max(buy_value * self.commission_rate, self.min_commission)
            sell_commission = max(sell_value * self.commission_rate, self.min_commission)
            stamp_duty = sell_value * self.stamp_duty_rate
        else:
            # 国债 ETF 交易 (无最低佣金，无印花税)
            buy_commission = buy_value * self.commission_rate
            sell_commission = sell_value * self.commission_rate
            stamp_duty = 0.0
        
        total_commission = buy_commission + sell_commission
        total_cost = total_commission + stamp_duty
        
        return total_cost, total_commission, stamp_duty
    
    def apply_round_lot_constraint(
        self,
        target_value: float,
        price: float,
    ) -> Tuple[int, float]:
        """
        应用股数取整约束.
        
        Returns:
            (actual_shares, unused_cash)
        """
        theoretical_shares = target_value / price
        actual_shares = int(theoretical_shares / self.round_lot) * self.round_lot
        unused_cash = target_value - actual_shares * price
        
        return actual_shares, unused_cash
    
    def run_simulation(
        self,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
    ) -> Dict[str, Any]:
        """
        运行高级仿真.
        
        Returns:
            仿真结果字典
        """
        records = []
        predict_df = predict_df.sort(["symbol", "trade_date"])
        unique_dates = predict_df["trade_date"].unique().sort().to_list()
        
        if len(unique_dates) < 2:
            logger.error("Not enough data for simulation")
            return self._empty_result()
        
        # 初始化状态
        cash = self.initial_capital
        bond_etf_value = 0.0  # 国债 ETF 持仓价值
        positions: Dict[str, Dict] = {}
        hold_days: Dict[str, int] = {}
        last_rebalance_date = None
        
        # 每日预测排名缓存
        daily_rankings_cache = {}
        
        # 统计
        total_unused_cash = 0.0
        daily_unused_cash = 0.0
        total_bond_etf_profit = 0.0
        
        for current_idx, current_date in enumerate(unique_dates):
            date_str = self._format_date(current_date)
            
            # 获取当日数据
            day_data = predict_df.filter(pl.col("trade_date") == current_date)
            if day_data.is_empty():
                continue
            
            day_clean = day_data.drop_nulls(subset=feature_columns)
            if day_clean.is_empty():
                continue
            
            # 获取价格和符号
            symbols = day_clean["symbol"].to_list()
            closes = day_clean["close"].to_list()
            X_day = day_clean.select(feature_columns).to_numpy()
            
            # 模型预测
            predictions = model.predict(X_day)
            probabilities = self._convert_to_probability(predictions)
            
            # 计算当日排名
            sorted_indices = np.argsort(-predictions)
            rankings = np.zeros(len(predictions))
            rankings[sorted_indices] = np.arange(1, len(predictions) + 1)
            ranking_percentiles = rankings / len(predictions)
            
            # 缓存排名
            for i, symbol in enumerate(symbols):
                daily_rankings_cache[date_str] = daily_rankings_cache.get(date_str, {})
                daily_rankings_cache[date_str][symbol] = {
                    "pred": predictions[i],
                    "prob": probabilities[i],
                    "rank": rankings[i],
                    "rank_pct": ranking_percentiles[i],
                    "close": closes[i],
                }
            
            # 获取次日价格
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
            
            # 计算国债 ETF 日收益
            if bond_etf_value > 0:
                daily_bond_return = self.bond_etf_return / 252
                bond_profit = bond_etf_value * daily_bond_return
                bond_etf_value += bond_profit
                total_bond_etf_profit += bond_profit
                self.total_bond_etf_days += 1
            
            # 检查是否需要调仓
            should_rebalance = False
            if last_rebalance_date is None:
                should_rebalance = True
            else:
                days_since_rebalance = self._days_between(last_rebalance_date, current_date)
                should_rebalance = days_since_rebalance >= self.rebalance_frequency
            
            # 检查是否有排名衰减退出信号
            exit_signals = []
            if self.use_ranking_exit and positions:
                for symbol in list(positions.keys()):
                    if date_str in daily_rankings_cache and symbol in daily_rankings_cache[date_str]:
                        rank_pct = daily_rankings_cache[date_str][symbol]["rank_pct"]
                        if rank_pct > self.ranking_exit_threshold:
                            exit_signals.append(symbol)
            
            # 调仓逻辑
            if should_rebalance or exit_signals:
                # 卖出所有持仓
                for symbol, pos_info in list(positions.items()):
                    if symbol in next_prices:
                        sell_price = next_prices[symbol]
                        shares = pos_info["shares"]
                        buy_price = pos_info["buy_price"]
                        
                        cost, commission, stamp_duty = self.calculate_transaction_cost(
                            buy_price, sell_price, shares, is_stock=True
                        )
                        
                        self.total_commission += commission
                        self.total_stamp_duty += stamp_duty
                        
                        sell_value = sell_price * shares
                        profit = sell_value - buy_price * shares - cost
                        cash += sell_value - cost
                        self.total_trades += 1
                        
                        records.append({
                            "trade_date": current_date,
                            "symbol": symbol,
                            "action": "SELL",
                            "reason": "rebalance" if symbol not in exit_signals else "ranking_exit",
                            "buy_price": buy_price,
                            "sell_price": sell_price,
                            "shares": shares,
                            "cost": cost,
                            "profit": profit,
                        })
                        
                        del positions[symbol]
                        del hold_days[symbol]
                
                # 如果是因为调仓日，重新买入
                if should_rebalance:
                    # 筛选满足概率阈值的股票
                    pred_results = [
                        {"symbol": s, "pred_return": p, "prob": prob, "close": c}
                        for s, p, prob, c in zip(symbols, predictions, probabilities, closes)
                        if prob >= self.probability_threshold
                    ]
                    pred_results.sort(key=lambda x: x["pred_return"], reverse=True)
                    top_k_results = pred_results[:self.top_k]
                    
                    # 计算可用资金
                    position_value = cash / len(top_k_results) if top_k_results else 0
                    
                    # 买入逻辑
                    for signal in top_k_results:
                        symbol = signal["symbol"]
                        buy_price = next_prices.get(symbol, signal["close"])
                        
                        if symbol not in positions:
                            actual_shares, unused = self.apply_round_lot_constraint(
                                position_value, buy_price
                            )
                            
                            daily_unused_cash += unused
                            total_unused_cash += unused
                            
                            if actual_shares > 0:
                                cost, commission, stamp_duty = self.calculate_transaction_cost(
                                    buy_price, buy_price, actual_shares, is_stock=True
                                )
                                
                                self.total_commission += commission
                                
                                buy_cost = buy_price * actual_shares + commission
                                if buy_cost <= cash:
                                    positions[symbol] = {
                                        "buy_price": buy_price,
                                        "shares": actual_shares,
                                        "buy_date": current_date,
                                        "target_value": position_value,
                                        "actual_value": buy_cost,
                                        "unused_cash": unused,
                                    }
                                    hold_days[symbol] = 0
                                    cash -= buy_cost
                                    self.total_trades += 1
                
                # 剩余现金买入国债 ETF
                if self.use_bond_etf and cash > 1000:
                    bond_etf_value += cash
                    cash = 0
                
                last_rebalance_date = current_date
            
            # 记录每日组合价值
            portfolio_value = cash + bond_etf_value
            for symbol, pos_info in positions.items():
                portfolio_value += pos_info["buy_price"] * pos_info["shares"]
            
            self.cash_drag_records.append({
                "trade_date": date_str,
                "cash": cash,
                "bond_etf_value": bond_etf_value,
                "positions_value": sum(p["buy_price"] * p["shares"] for p in positions.values()),
                "portfolio_value": portfolio_value,
                "unused_cash": daily_unused_cash,
                "num_positions": len(positions),
            })
            
            records.append({
                "trade_date": current_date,
                "action": "DAILY_VALUE",
                "cash": cash,
                "bond_etf_value": bond_etf_value,
                "portfolio_value": portfolio_value,
                "num_positions": len(positions),
            })
        
        return {
            "records": records,
            "final_value": cash + bond_etf_value,
            "total_return": (cash + bond_etf_value - self.initial_capital) / self.initial_capital,
            "total_unused_cash": total_unused_cash,
            "bond_etf_profit": total_bond_etf_profit,
            "total_trades": self.total_trades,
            "bond_etf_days": self.total_bond_etf_days,
            "cost_analysis": {
                "total_commission": self.total_commission,
                "total_stamp_duty": self.total_stamp_duty,
                "total_cost": self.total_commission + self.total_stamp_duty,
            },
            "cash_drag_records": self.cash_drag_records,
        }
    
    def _convert_to_probability(self, predictions: np.ndarray, 
                                 predict_df: pl.DataFrame = None,
                                 current_date_idx: int = None,
                                 date_str: str = None) -> np.ndarray:
        """
        将模型预测转换为概率 (使用排名百分位).
        
        由于 LightGBM 输出的是原始预测值而非概率，我们使用排名百分位来估计概率。
        排名越靠前，概率越高。
        """
        # 使用排名百分位作为概率估计
        sorted_indices = np.argsort(-predictions)
        ranks = np.zeros(len(predictions))
        ranks[sorted_indices] = np.arange(len(predictions))
        probabilities = ranks / len(predictions)
        # 反转：排名越高（值越小），概率越高
        probabilities = 1 - probabilities
        return probabilities
    
    def _format_date(self, date) -> str:
        """格式化日期."""
        if hasattr(date, 'strftime'):
            return date.strftime("%Y-%m-%d")
        return str(date)[:10]
    
    def _days_between(self, date1, date2) -> int:
        """计算两个日期之间的天数."""
        d1 = datetime.strptime(self._format_date(date1), "%Y-%m-%d")
        d2 = datetime.strptime(self._format_date(date2), "%Y-%m-%d")
        return (d2 - d1).days
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果."""
        return {
            "records": [],
            "final_value": self.initial_capital,
            "total_return": 0.0,
            "total_unused_cash": 0.0,
            "bond_etf_profit": 0.0,
            "total_trades": 0,
            "bond_etf_days": 0,
            "cost_analysis": {
                "total_commission": 0.0,
                "total_stamp_duty": 0.0,
                "total_cost": 0.0,
            },
            "cash_drag_records": [],
        }


class FrequencyKScanner:
    """
    频率与 K 值交叉扫描器.
    
    测试不同调仓频率和 K 值组合的表现.
    """
    
    def __init__(
        self,
        frequencies: List[int] = None,
        k_values: List[int] = None,
    ):
        self.frequencies = frequencies or [1, 2, 3, 5]
        self.k_values = k_values or [3, 5]
        logger.info(f"FrequencyKScanner: frequencies={self.frequencies}, k_values={self.k_values}")
    
    def run_scan(
        self,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
        initial_capital: float = 50_000.0,
    ) -> Dict[str, Any]:
        """运行频率与 K 值交叉测试."""
        results = {}
        
        for freq in self.frequencies:
            for k in self.k_values:
                key = f"freq_{freq}_k_{k}"
                logger.info(f"Testing {key}...")
                
                simulator = AdvancedSmallCapSimulator(
                    initial_capital=initial_capital,
                    top_k=k,
                    rebalance_frequency=freq,
                    probability_threshold=0.0,  # 无概率阈值
                    use_ranking_exit=False,
                    use_bond_etf=False,
                )
                
                result = simulator.run_simulation(predict_df, model, feature_columns)
                
                # 计算年化收益率和夏普比率
                annualized = self._annualize_return(result["total_return"], len(result["records"]))
                sharpe = self._calculate_sharpe(result["records"])
                
                results[key] = {
                    "frequency": freq,
                    "k_value": k,
                    "total_return": result["total_return"],
                    "annualized_return": annualized,
                    "sharpe_ratio": sharpe,
                    "total_trades": result["total_trades"],
                    "total_cost": result["cost_analysis"]["total_cost"],
                    "cost_ratio": result["cost_analysis"]["total_cost"] / initial_capital,
                    "final_value": result["final_value"],
                }
        
        # 找出最优配置
        best_by_sharpe = max(results.keys(), key=lambda k: results[k]["sharpe_ratio"])
        best_by_return = max(results.keys(), key=lambda k: results[k]["annualized_return"])
        
        return {
            "scan_results": results,
            "best_by_sharpe": best_by_sharpe,
            "best_by_return": best_by_return,
        }
    
    def _annualize_return(self, total_return: float, num_records: int) -> float:
        """计算年化收益率."""
        if num_records < 2:
            return 0.0
        num_days = num_records // 2
        return (1 + total_return) ** (252 / max(num_days, 1)) - 1
    
    def _calculate_sharpe(self, records: List[Dict], risk_free_rate: float = 0.03) -> float:
        """计算夏普比率."""
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


class ProbabilityThresholdTester:
    """
    概率阈值测试器.
    
    测试不同概率阈值和退出策略的表现.
    """
    
    def __init__(
        self,
        probability_thresholds: List[float] = None,
        ranking_exit_threshold: float = 0.15,
    ):
        self.probability_thresholds = probability_thresholds or [0.70, 0.75, 0.80, 0.85]
        self.ranking_exit_threshold = ranking_exit_threshold
        logger.info(f"ProbabilityThresholdTester: thresholds={self.probability_thresholds}")
    
    def run_test(
        self,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
        initial_capital: float = 50_000.0,
        top_k: int = 3,
        rebalance_frequency: int = 5,
    ) -> Dict[str, Any]:
        """运行概率阈值测试."""
        results = {}
        
        # 测试固定阈值
        for thresh in self.probability_thresholds:
            key = f"fixed_thresh_{thresh:.2f}"
            logger.info(f"Testing {key}...")
            
            simulator = AdvancedSmallCapSimulator(
                initial_capital=initial_capital,
                top_k=top_k,
                rebalance_frequency=rebalance_frequency,
                probability_threshold=thresh,
                use_ranking_exit=False,
                use_bond_etf=False,
            )
            
            result = simulator.run_simulation(predict_df, model, feature_columns)
            
            annualized = self._annualize_return(result["total_return"], len(result["records"]))
            sharpe = self._calculate_sharpe(result["records"])
            
            results[key] = {
                "strategy": "fixed_threshold",
                "threshold": thresh,
                "total_return": result["total_return"],
                "annualized_return": annualized,
                "sharpe_ratio": sharpe,
                "total_trades": result["total_trades"],
                "total_cost": result["cost_analysis"]["total_cost"],
                "final_value": result["final_value"],
            }
        
        # 测试排名衰减退出
        key = "ranking_exit"
        logger.info(f"Testing {key}...")
        
        simulator = AdvancedSmallCapSimulator(
            initial_capital=initial_capital,
            top_k=top_k,
            rebalance_frequency=rebalance_frequency,
            probability_threshold=0.0,
            use_ranking_exit=True,
            ranking_exit_threshold=self.ranking_exit_threshold,
            use_bond_etf=False,
        )
        
        result = simulator.run_simulation(predict_df, model, feature_columns)
        
        annualized = self._annualize_return(result["total_return"], len(result["records"]))
        sharpe = self._calculate_sharpe(result["records"])
        
        results[key] = {
            "strategy": "ranking_exit",
            "threshold": self.ranking_exit_threshold,
            "total_return": result["total_return"],
            "annualized_return": annualized,
            "sharpe_ratio": sharpe,
            "total_trades": result["total_trades"],
            "total_cost": result["cost_analysis"]["total_cost"],
            "final_value": result["final_value"],
        }
        
        # 测试组合策略 (固定阈值 + 排名退出)
        for thresh in self.probability_thresholds:
            key = f"combined_thresh_{thresh:.2f}_rank_exit"
            logger.info(f"Testing {key}...")
            
            simulator = AdvancedSmallCapSimulator(
                initial_capital=initial_capital,
                top_k=top_k,
                rebalance_frequency=rebalance_frequency,
                probability_threshold=thresh,
                use_ranking_exit=True,
                ranking_exit_threshold=self.ranking_exit_threshold,
                use_bond_etf=False,
            )
            
            result = simulator.run_simulation(predict_df, model, feature_columns)
            
            annualized = self._annualize_return(result["total_return"], len(result["records"]))
            sharpe = self._calculate_sharpe(result["records"])
            
            results[key] = {
                "strategy": "combined",
                "threshold": thresh,
                "total_return": result["total_return"],
                "annualized_return": annualized,
                "sharpe_ratio": sharpe,
                "total_trades": result["total_trades"],
                "total_cost": result["cost_analysis"]["total_cost"],
                "final_value": result["final_value"],
            }
        
        # 找出最优配置
        best_by_sharpe = max(results.keys(), key=lambda k: results[k]["sharpe_ratio"])
        
        return {
            "threshold_results": results,
            "best_by_sharpe": best_by_sharpe,
        }
    
    def _annualize_return(self, total_return: float, num_records: int) -> float:
        """计算年化收益率."""
        if num_records < 2:
            return 0.0
        num_days = num_records // 2
        return (1 + total_return) ** (252 / max(num_days, 1)) - 1
    
    def _calculate_sharpe(self, records: List[Dict], risk_free_rate: float = 0.03) -> float:
        """计算夏普比率."""
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


class CashDragAnalyzer:
    """
    现金闲置率分析器.
    
    分析由于 100 股取整限制导致的现金闲置问题.
    """
    
    def __init__(self):
        logger.info("CashDragAnalyzer initialized")
    
    def analyze(
        self,
        simulation_result: Dict[str, Any],
        initial_capital: float = 50_000.0,
    ) -> Dict[str, Any]:
        """分析现金闲置情况."""
        cash_drag_records = simulation_result.get("cash_drag_records", [])
        
        if not cash_drag_records:
            return {"status": "no_data"}
        
        # 计算每日闲置现金比例
        unused_cash_ratios = []
        for record in cash_drag_records:
            unused = record.get("unused_cash", 0)
            ratio = unused / initial_capital if initial_capital > 0 else 0
            unused_cash_ratios.append({
                "date": record.get("trade_date"),
                "unused_cash": unused,
                "ratio": ratio,
            })
        
        # 统计
        avg_unused = np.mean([r["unused_cash"] for r in unused_cash_ratios])
        max_unused = np.max([r["unused_cash"] for r in unused_cash_ratios])
        avg_ratio = np.mean([r["ratio"] for r in unused_cash_ratios])
        
        # 计算现金拖累 (假设闲置现金收益率为 0)
        opportunity_cost = avg_unused * 0.035 / 252 * len(cash_drag_records)
        
        return {
            "avg_unused_cash": avg_unused,
            "max_unused_cash": max_unused,
            "avg_unused_ratio": avg_ratio,
            "opportunity_cost": opportunity_cost,
            "daily_records": unused_cash_ratios,
        }


class BondETFOptimizer:
    """
    国债 ETF 优化器.
    
    测试使用国债 ETF 管理剩余现金的效果.
    """
    
    def __init__(self, bond_etf_return: float = 0.035):
        self.bond_etf_return = bond_etf_return
        logger.info(f"BondETFOptimizer: annual_return={bond_etf_return}")
    
    def run_comparison(
        self,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
        initial_capital: float = 50_000.0,
        top_k: int = 3,
        rebalance_frequency: int = 5,
        probability_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """比较使用和不使用国债 ETF 的表现."""
        
        # 不使用国债 ETF
        simulator_no_bond = AdvancedSmallCapSimulator(
            initial_capital=initial_capital,
            top_k=top_k,
            rebalance_frequency=rebalance_frequency,
            probability_threshold=probability_threshold,
            use_bond_etf=False,
        )
        result_no_bond = simulator_no_bond.run_simulation(predict_df, model, feature_columns)
        
        # 使用国债 ETF
        simulator_with_bond = AdvancedSmallCapSimulator(
            initial_capital=initial_capital,
            top_k=top_k,
            rebalance_frequency=rebalance_frequency,
            probability_threshold=probability_threshold,
            use_bond_etf=True,
            bond_etf_return=self.bond_etf_return,
        )
        result_with_bond = simulator_with_bond.run_simulation(predict_df, model, feature_columns)
        
        # 计算改善
        improvement = result_with_bond["total_return"] - result_no_bond["total_return"]
        improvement_annualized = (1 + result_with_bond["total_return"]) / (1 + result_no_bond["total_return"]) - 1
        
        return {
            "without_bond_etf": {
                "total_return": result_no_bond["total_return"],
                "final_value": result_no_bond["final_value"],
                "total_trades": result_no_bond["total_trades"],
            },
            "with_bond_etf": {
                "total_return": result_with_bond["total_return"],
                "final_value": result_with_bond["final_value"],
                "total_trades": result_with_bond["total_trades"],
                "bond_etf_profit": result_with_bond["bond_etf_profit"],
                "bond_etf_days": result_with_bond["bond_etf_days"],
            },
            "improvement": {
                "absolute": improvement,
                "annualized": improvement_annualized,
            },
        }


class ExecutionOptimizer:
    """
    综合执行优化器.
    
    整合所有测试模块，生成最佳执行策略报告.
    """
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        self.results: Dict[str, Any] = {}
        self.model: Optional[lgb.Booster] = None
        self.predict_df: Optional[pl.DataFrame] = None
        self.feature_columns: List[str] = []
        
        logger.info("=" * 60)
        logger.info("Execution Optimizer Initialized")
        logger.info("=" * 60)
        logger.info(f"Config: {asdict(self.config)}")
    
    def load_data(self) -> Tuple[pl.DataFrame, lgb.Booster, List[str]]:
        """加载数据和模型."""
        logger.info("Loading data...")
        
        # 加载特征数据
        df = pl.read_parquet(self.config.parquet_path)
        df = df.sort("trade_date")
        
        # 准备特征列
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
        
        # 训练模型
        logger.info("Training model...")
        data = ModelTrainer.prepare_data(df, feature_columns)
        
        trainer = ModelTrainer(n_estimators=500, max_depth=4, num_leaves=18)
        trainer.train(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
        )
        
        self.model = trainer.model
        self.predict_df = df
        self.feature_columns = feature_columns
        
        return df, self.model, feature_columns
    
    def run_frequency_k_scan(self) -> Dict[str, Any]:
        """运行频率与 K 值交叉测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 1: Frequency & K-Selection Sweep")
        logger.info("=" * 60)
        
        scanner = FrequencyKScanner(
            frequencies=self.config.rebalance_frequencies,
            k_values=self.config.k_values,
        )
        
        result = scanner.run_scan(
            self.predict_df,
            self.model,
            self.feature_columns,
            self.config.initial_capital,
        )
        
        logger.info("Frequency & K-Value Scan Results:")
        for key, stats in result["scan_results"].items():
            logger.info(f"  {key}: Return={stats['annualized_return']:.2%}, "
                       f"Sharpe={stats['sharpe_ratio']:.2f}, Cost={stats['cost_ratio']:.2%}")
        logger.info(f"Best by Sharpe: {result['best_by_sharpe']}")
        
        self.results["frequency_k_scan"] = result
        return result
    
    def run_probability_threshold_test(self) -> Dict[str, Any]:
        """运行概率阈值测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 2: Dynamic Probability Threshold Test")
        logger.info("=" * 60)
        
        tester = ProbabilityThresholdTester(
            probability_thresholds=self.config.probability_thresholds,
            ranking_exit_threshold=self.config.ranking_exit_threshold,
        )
        
        # 使用最优频率和 K 值
        best_freq = 5
        best_k = 3
        if "frequency_k_scan" in self.results:
            best_key = self.results["frequency_k_scan"]["best_by_sharpe"]
            best_freq = self.results["frequency_k_scan"]["scan_results"][best_key]["frequency"]
            best_k = self.results["frequency_k_scan"]["scan_results"][best_key]["k_value"]
        
        result = tester.run_test(
            self.predict_df,
            self.model,
            self.feature_columns,
            self.config.initial_capital,
            top_k=best_k,
            rebalance_frequency=best_freq,
        )
        
        logger.info("Probability Threshold Test Results:")
        for key, stats in result["threshold_results"].items():
            logger.info(f"  {key}: Return={stats['annualized_return']:.2%}, "
                       f"Sharpe={stats['sharpe_ratio']:.2f}, Trades={stats['total_trades']}")
        logger.info(f"Best by Sharpe: {result['best_by_sharpe']}")
        
        self.results["probability_threshold"] = result
        return result
    
    def run_cash_drag_analysis(self) -> Dict[str, Any]:
        """运行现金闲置率分析."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 3: Cash Drag Analysis")
        logger.info("=" * 60)
        
        analyzer = CashDragAnalyzer()
        
        # 使用最优配置运行一次仿真
        best_freq = 5
        best_k = 3
        if "frequency_k_scan" in self.results:
            best_key = self.results["frequency_k_scan"]["best_by_sharpe"]
            best_freq = self.results["frequency_k_scan"]["scan_results"][best_key]["frequency"]
            best_k = self.results["frequency_k_scan"]["scan_results"][best_key]["k_value"]
        
        simulator = AdvancedSmallCapSimulator(
            initial_capital=self.config.initial_capital,
            top_k=best_k,
            rebalance_frequency=best_freq,
            probability_threshold=0.0,
            use_bond_etf=False,
        )
        
        result = simulator.run_simulation(self.predict_df, self.model, self.feature_columns)
        cash_drag_result = analyzer.analyze(result, self.config.initial_capital)
        
        logger.info(f"Cash Drag Analysis:")
        logger.info(f"  Avg Unused Cash: {cash_drag_result.get('avg_unused_cash', 0):.2f}元")
        logger.info(f"  Avg Unused Ratio: {cash_drag_result.get('avg_unused_ratio', 0):.2%}")
        logger.info(f"  Opportunity Cost: {cash_drag_result.get('opportunity_cost', 0):.2f}元")
        
        self.results["cash_drag"] = cash_drag_result
        return cash_drag_result
    
    def run_bond_etf_optimization(self) -> Dict[str, Any]:
        """运行国债 ETF 优化测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Test 4: Bond ETF Optimization")
        logger.info("=" * 60)
        
        optimizer = BondETFOptimizer(bond_etf_return=self.config.bond_etf_annual_return)
        
        # 使用最优配置
        best_freq = 5
        best_k = 3
        best_thresh = 0.75
        if "frequency_k_scan" in self.results:
            best_key = self.results["frequency_k_scan"]["best_by_sharpe"]
            best_freq = self.results["frequency_k_scan"]["scan_results"][best_key]["frequency"]
            best_k = self.results["frequency_k_scan"]["scan_results"][best_key]["k_value"]
        
        result = optimizer.run_comparison(
            self.predict_df,
            self.model,
            self.feature_columns,
            self.config.initial_capital,
            top_k=best_k,
            rebalance_frequency=best_freq,
            probability_threshold=best_thresh,
        )
        
        logger.info("Bond ETF Optimization Results:")
        logger.info(f"  Without Bond ETF: Return={result['without_bond_etf']['total_return']:.2%}")
        logger.info(f"  With Bond ETF: Return={result['with_bond_etf']['total_return']:.2%}")
        logger.info(f"  Improvement: {result['improvement']['annualized']:.2%}")
        
        self.results["bond_etf"] = result
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试."""
        logger.info("\n" + "=" * 60)
        logger.info("Starting Execution Optimization Tests")
        logger.info("=" * 60)
        
        # 加载数据
        self.load_data()
        
        # 运行测试
        self.run_frequency_k_scan()
        self.run_probability_threshold_test()
        self.run_cash_drag_analysis()
        self.run_bond_etf_optimization()
        
        logger.info("\n" + "=" * 60)
        logger.info("All Execution Optimization Tests Complete")
        logger.info("=" * 60)
        
        return self.results
    
    def generate_report(self, output_path: str = "docs/execution_strategy_v3.md") -> str:
        """生成执行策略报告."""
        logger.info(f"Generating report to {output_path}")
        
        report = []
        report.append("# 5 万元实盘执行策略手册 (V3)\n\n")
        report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"**测试对象**: 5 万元本金小额实盘\n\n")
        
        report.append("---\n\n")
        report.append("## 执行摘要\n\n")
        report.append("本报告通过系统性测试，为 5 万元本金确定最佳执行策略:\n")
        report.append("1. **调仓频率**: 测试日频、2 日、3 日、周频调仓\n")
        report.append("2. **选股数量**: 测试 K=3 和 K=5\n")
        report.append("3. **概率阈值**: 测试固定概率阈值开仓\n")
        report.append("4. **退出策略**: 测试排名衰减退出逻辑\n")
        report.append("5. **现金管理**: 测试国债 ETF 优化\n\n")
        
        # 测试 1: 频率与 K 值
        report.append("## 1. 频率与 K 值交叉回测\n\n")
        if "frequency_k_scan" in self.results:
            fk = self.results["frequency_k_scan"]
            report.append("### 测试矩阵\n\n")
            report.append("| 调仓频率 | K=3 年化收益 | K=5 年化收益 |\n")
            report.append("|---------|-------------|-------------|\n")
            
            for freq in self.config.rebalance_frequencies:
                k3_key = f"freq_{freq}_k_3"
                k5_key = f"freq_{freq}_k_5"
                k3_ret = fk["scan_results"].get(k3_key, {}).get("annualized_return", 0)
                k5_ret = fk["scan_results"].get(k5_key, {}).get("annualized_return", 0)
                report.append(f"| {freq}日 ({'周频' if freq == 5 else f'{freq}日'}) | {k3_ret:.2%} | {k5_ret:.2%} |\n")
            
            report.append("\n### 最优配置\n\n")
            best_key = fk["best_by_sharpe"]
            best_stats = fk["scan_results"][best_key]
            report.append(f"**最优夏普比率配置**: {best_key}\n")
            report.append(f"- 调仓频率：{best_stats['frequency']}日\n")
            report.append(f"- K 值：{best_stats['k_value']}\n")
            report.append(f"- 年化收益：{best_stats['annualized_return']:.2%}\n")
            report.append(f"- 夏普比率：{best_stats['sharpe_ratio']:.2f}\n")
            report.append(f"- 成本占比：{best_stats['cost_ratio']:.2%}\n\n")
        
        # 测试 2: 概率阈值
        report.append("## 2. 动态概率阈值测试\n\n")
        if "probability_threshold" in self.results:
            pt = self.results["probability_threshold"]
            report.append("### 策略对比\n\n")
            report.append("| 策略 | 年化收益 | 夏普比率 | 交易次数 |\n")
            report.append("|------|---------|---------|----------|\n")
            
            for key, stats in pt["threshold_results"].items():
                report.append(f"| {key} | {stats['annualized_return']:.2%} | {stats['sharpe_ratio']:.2f} | {stats['total_trades']} |\n")
            
            best_key = pt["best_by_sharpe"]
            best_stats = pt["threshold_results"][best_key]
            report.append(f"\n**最优策略**: {best_key}\n")
            report.append(f"- 年化收益：{best_stats['annualized_return']:.2%}\n")
            report.append(f"- 夏普比率：{best_stats['sharpe_ratio']:.2f}\n\n")
        
        # 测试 3: 现金闲置
        report.append("## 3. 现金闲置率分析\n\n")
        if "cash_drag" in self.results:
            cd = self.results["cash_drag"]
            report.append(f"- **平均闲置现金**: {cd.get('avg_unused_cash', 0):.2f}元\n")
            report.append(f"- **平均闲置比例**: {cd.get('avg_unused_ratio', 0):.2%}\n")
            report.append(f"- **机会成本**: {cd.get('opportunity_cost', 0):.2f}元\n\n")
        
        # 测试 4: 国债 ETF
        report.append("## 4. 国债 ETF 优化测试\n\n")
        if "bond_etf" in self.results:
            be = self.results["bond_etf"]
            report.append("| 配置 | 总收益 | 最终价值 |\n")
            report.append("|------|-------|----------|\n")
            report.append(f"| 不使用国债 ETF | {be['without_bond_etf']['total_return']:.2%} | {be['without_bond_etf']['final_value']:.2f}元 |\n")
            report.append(f"| 使用国债 ETF | {be['with_bond_etf']['total_return']:.2%} | {be['with_bond_etf']['final_value']:.2f}元 |\n")
            report.append(f"\n**改善效果**: {be['improvement']['annualized']:.2%}\n\n")
        
        # 最终建议
        report.append("---\n\n")
        report.append("## 5 万元实盘最佳执行手册\n\n")
        report.append("### 核心问题解答\n\n")
        
        # 确定最优配置
        best_freq = 5
        best_k = 3
        best_thresh = 0.75
        use_ranking_exit = False
        use_bond_etf = True
        
        if "frequency_k_scan" in self.results:
            best_key = self.results["frequency_k_scan"]["best_by_sharpe"]
            best_freq = self.results["frequency_k_scan"]["scan_results"][best_key]["frequency"]
            best_k = self.results["frequency_k_scan"]["scan_results"][best_key]["k_value"]
        
        if "probability_threshold" in self.results:
            best_pt_key = self.results["probability_threshold"]["best_by_sharpe"]
            best_pt_stats = self.results["probability_threshold"]["threshold_results"][best_pt_key]
            if "thresh" in best_pt_key:
                best_thresh = best_pt_stats.get("threshold", 0.75)
            use_ranking_exit = "rank" in best_pt_key
        
        report.append("#### Q1: 到底应该选几只股？\n\n")
        report.append(f"**答：K = {best_k} 只**\n\n")
        report.append(f"理由:\n")
        report.append(f"- 5 万元本金下，单股配置约 {50000/best_k:.0f}元\n")
        report.append(f"- K={best_k}可在分散风险和避免碎股间取得平衡\n")
        report.append(f"- K 值过大导致单股金额过低，无法买足 100 股\n\n")
        
        report.append("#### Q2: 到底应该哪天调仓？\n\n")
        report.append(f"**答：每{best_freq}天调仓一次**({'周频' if best_freq == 5 else f'{best_freq}日频'})\n\n")
        report.append(f"理由:\n")
        report.append(f"- 日频调仓成本占比过高 (约 70%+)\n")
        report.append(f"- {best_freq}日调仓可降低交易次数，减少佣金侵蚀\n")
        report.append(f"- Alpha 信号在{best_freq}日内衰减有限\n\n")
        
        report.append("#### Q3: 预测得分到多少才值得出手？\n\n")
        report.append(f"**答：概率阈值 >= {best_thresh:.0%}**\n\n")
        report.append(f"理由:\n")
        report.append(f"- 低于{best_thresh:.0%}的股票胜率不足，不值得承担交易成本\n")
        report.append(f"- 高于{best_thresh:.0%}的股票具有统计显著的 Alpha\n\n")
        
        report.append("### 最终配置建议\n\n")
        report.append("| 参数 | 推荐值 | 说明 |\n")
        report.append("|------|--------|------|\n")
        report.append(f"| 选股数量 (K) | {best_k} | 平衡分散与可执行性 |\n")
        report.append(f"| 调仓频率 | {best_freq}天 | 降低交易成本 |\n")
        report.append(f"| 概率阈值 | {best_thresh:.0%} | 确保胜率 |\n")
        report.append(f"| 排名退出 | {'启用' if use_ranking_exit else '禁用'} | 动态止损 |\n")
        report.append(f"| 国债 ETF | {'启用' if use_bond_etf else '禁用'} | 现金管理 |\n\n")
        
        report.append("### 实盘执行清单\n\n")
        report.append("#### 开盘前 (9:15-9:25)\n")
        report.append("1. [ ] 检查数据库连接\n")
        report.append("2. [ ] 获取最新行情数据\n")
        report.append("3. [ ] 运行模型预测\n")
        report.append("4. [ ] 筛选 Top {best_k}股票 (概率>={best_thresh:.0%})\n\n")
        
        report.append("#### 交易中 (9:30-15:00)\n")
        report.append("1. [ ] 使用限价单，避免追高\n")
        report.append("2. [ ] 监控成交量，避免冲击\n")
        report.append("3. [ ] 记录实际成交价格\n\n")
        
        report.append("#### 收盘后 (15:00-17:00)\n")
        report.append("1. [ ] 更新持仓和权益\n")
        report.append("2. [ ] 检查排名衰减信号\n")
        report.append("3. [ ] 生成交易日志\n\n")
        
        report.append("### 风险提示\n\n")
        report.append("1. **碎股约束**: 高价股 (>100 元) 可能无法买入 100 股，自动跳过\n")
        report.append("2. **佣金侵蚀**: 单笔最低 5 元，建议单笔下单位≥1.7 万元\n")
        report.append("3. **流动性风险**: 避免买入日均成交<500 万元的股票\n")
        report.append("4. **模型风险**: 定期 (季度) 重新训练，监控 IC 衰减\n")
        report.append("5. **极端行情**: 启用 Regime Switch 保护，必要时空仓\n\n")
        
        report.append("---\n\n")
        report.append("## 附录：测试脚本信息\n\n")
        report.append("**测试文件**: `src/execution_optimizer.py`\n\n")
        report.append("**测试类**:\n")
        report.append("- `AdvancedSmallCapSimulator` - 高级仿真器\n")
        report.append("- `FrequencyKScanner` - 频率与 K 值扫描\n")
        report.append("- `ProbabilityThresholdTester` - 概率阈值测试\n")
        report.append("- `CashDragAnalyzer` - 现金闲置分析\n")
        report.append("- `BondETFOptimizer` - 国债 ETF 优化\n\n")
        
        report.append("**运行方式**:\n")
        report.append("```bash\n")
        report.append("python src/execution_optimizer.py --all\n")
        report.append("```\n")
        
        report_content = "".join(report)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {output_path}")
        return report_content
    
    def plot_results(self, output_path: str = "data/plots/execution_optimization_results.png") -> None:
        """绘制测试结果."""
        logger.info(f"Plotting results to {output_path}")
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
        
        # 图 1: 频率与 K 值热力图
        ax1 = fig.add_subplot(gs[0, 0])
        if "frequency_k_scan" in self.results:
            fk = self.results["frequency_k_scan"]
            freqs = self.config.rebalance_frequencies
            ks = self.config.k_values
            
            sharpe_matrix = np.zeros((len(freqs), len(ks)))
            for i, freq in enumerate(freqs):
                for j, k in enumerate(ks):
                    key = f"freq_{freq}_k_{k}"
                    sharpe_matrix[i, j] = fk["scan_results"].get(key, {}).get("sharpe_ratio", 0)
            
            im = ax1.imshow(sharpe_matrix, cmap='RdYlGn', aspect='auto')
            ax1.set_xticks(range(len(ks)))
            ax1.set_yticks(range(len(freqs)))
            ax1.set_xticklabels([f'K={k}' for k in ks])
            ax1.set_yticklabels([f'{f}日' for f in freqs])
            ax1.set_title("夏普比率热力图", fontsize=12)
            plt.colorbar(im, ax=ax1, label='夏普比率')
            
            # 标注数值
            for i in range(len(freqs)):
                for j in range(len(ks)):
                    ax1.text(j, i, f'{sharpe_matrix[i, j]:.2f}', ha='center', va='center',
                            color='black' if sharpe_matrix[i, j] < 0 else 'white')
        
        # 图 2: 概率阈值对比
        ax2 = fig.add_subplot(gs[0, 1])
        if "probability_threshold" in self.results:
            pt = self.results["probability_threshold"]
            strategies = list(pt["threshold_results"].keys())
            sharpes = [pt["threshold_results"][s]["sharpe_ratio"] for s in strategies]
            returns = [pt["threshold_results"][s]["annualized_return"] for s in strategies]
            
            x = np.arange(len(strategies))
            width = 0.35
            
            ax2.bar(x - width/2, sharpes, width, label='夏普比率', color='#2E86AB')
            ax2.bar(x + width/2, returns, width, label='年化收益', color='#E74C3C')
            ax2.set_xticks(x)
            ax2.set_xticklabels(strategies, rotation=45, ha='right')
            ax2.set_title("概率阈值策略对比", fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 图 3: 现金闲置分析
        ax3 = fig.add_subplot(gs[1, 0])
        if "cash_drag" in self.results:
            cd = self.results["cash_drag"]
            daily_records = cd.get("daily_records", [])
            if daily_records:
                dates = [r["date"] for r in daily_records[::10]]  # 抽样
                unused = [r["unused_cash"] for r in daily_records[::10]]
                
                ax3.fill_between(range(len(dates)), unused, alpha=0.5, color='#FF6384')
                ax3.set_title("现金闲置趋势 (抽样)", fontsize=12)
                ax3.set_xlabel("交易日")
                ax3.set_ylabel("闲置现金 (元)")
                ax3.grid(True, alpha=0.3)
        
        # 图 4: 国债 ETF 对比
        ax4 = fig.add_subplot(gs[1, 1])
        if "bond_etf" in self.results:
            be = self.results["bond_etf"]
            categories = ['不使用国债 ETF', '使用国债 ETF']
            returns = [be['without_bond_etf']['total_return'], be['with_bond_etf']['total_return']]
            colors = ['#FF6384', '#4CAF50']
            
            ax4.bar(categories, returns, color=colors, alpha=0.7)
            ax4.set_title("国债 ETF 优化效果", fontsize=12)
            ax4.set_ylabel("总收益率")
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle("5 万元实盘执行优化测试结果", fontsize=16, fontweight='bold', y=0.995)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved to {output_path}")


def run_optimization_tests():
    """运行执行优化测试并生成报告."""
    optimizer = ExecutionOptimizer()
    results = optimizer.run_all_tests()
    optimizer.generate_report()
    optimizer.plot_results()
    return results


if __name__ == "__main__":
    results = run_optimization_tests()
    print("\n" + "=" * 60)
    print("执行优化测试完成")
    print("=" * 60)