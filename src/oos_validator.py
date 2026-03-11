#!/usr/bin/env python3
"""
Out-of-Sample (OOS) Validator - Walk-forward validation for quantitative strategies.

This script performs strict out-of-sample validation:
- In-Sample (IS): 2023-01-01 to 2025-06-30 (training period)
- Out-of-Sample (OOS): 2025-07-01 to 2026-03-10 (validation period)

核心功能:
    - 时间序列数据分割
    - 样本内模型训练
    - 样本外回测验证
    - 性能对比分析
    - 过拟合检测

使用示例:
    python src/oos_validator.py
    python src/oos_validator.py --threshold 0.01 --min-hold-days 3
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from loguru import logger
import polars as pl
import numpy as np
import lightgbm as lgb

from backtester import Backtester
from visualizer import Visualizer

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class OOSValidator:
    """
    样本外验证器，用于评估策略在未见数据上的表现。
    
    功能特性:
        - 严格的时间序列分割
        - 样本内模型训练
        - 样本外回测
        - 过拟合检测
        - 性能对比分析
    
    数据分割:
        - In-Sample (IS): 2023-01-01 至 2025-06-30
        - Out-of-Sample (OOS): 2025-07-01 至 2026-03-10
    
    过拟合判断标准:
        - Sharpe 衰减率 = Sharpe_OOS / Sharpe_IS
        - 衰减率 < 0.6: 严重过拟合
        - 衰减率 0.6-0.8: 轻度过拟合
        - 衰减率 > 0.8: 稳健策略
    """
    
    # 数据分割日期
    IS_START_DATE = "2023-01-01"
    IS_END_DATE = "2025-06-30"
    OOS_START_DATE = "2025-07-01"
    OOS_END_DATE = "2026-03-10"
    
    def __init__(
        self,
        parquet_path: str = "data/parquet/features_latest.parquet",
        initial_capital: float = 1_000_000.0,
        prediction_threshold: float = 0.01,
        max_positions: int = 10,
        min_hold_days: int = 3,
        use_dynamic_position: bool = True,
        output_dir: str = "data/plots",
    ) -> None:
        """
        初始化 OOS 验证器。
        
        Args:
            parquet_path (str): Parquet 特征文件路径
            initial_capital (float): 初始资金
            prediction_threshold (float): 预测收益率阈值
            max_positions (int): 最大持仓数量
            min_hold_days (int): 最小持仓天数
            use_dynamic_position (bool): 是否使用动态仓位
            output_dir (str): 输出目录
        """
        self.parquet_path = parquet_path
        self.initial_capital = initial_capital
        self.prediction_threshold = prediction_threshold
        self.max_positions = max_positions
        self.min_hold_days = min_hold_days
        self.use_dynamic_position = use_dynamic_position
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"OOSValidator initialized")
        logger.info(f"  In-Sample: {self.IS_START_DATE} to {self.IS_END_DATE}")
        logger.info(f"  Out-of-Sample: {self.OOS_START_DATE} to {self.OOS_END_DATE}")
    
    def load_and_split_data(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        加载数据并按时间分割 IS/OOS。
        
        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: (IS 数据，OOS 数据)
        """
        logger.info(f"Loading data from {self.parquet_path}")
        df = pl.read_parquet(self.parquet_path)
        
        # 转换日期列为字符串进行比较
        df = df.with_columns([
            pl.col("trade_date").cast(pl.Utf8).alias("trade_date_str")
        ])
        
        # IS 数据
        is_df = df.filter(
            (pl.col("trade_date_str") >= self.IS_START_DATE) &
            (pl.col("trade_date_str") <= self.IS_END_DATE)
        )
        
        # OOS 数据
        oos_df = df.filter(
            (pl.col("trade_date_str") >= self.OOS_START_DATE) &
            (pl.col("trade_date_str") <= self.OOS_END_DATE)
        )
        
        logger.info(f"IS data: {len(is_df)} rows")
        logger.info(f"OOS data: {len(oos_df)} rows")
        
        return is_df, oos_df
    
    def train_model_on_is(
        self,
        is_df: pl.DataFrame,
        feature_columns: list[str],
        label_column: str = "future_return_5",
    ) -> lgb.Booster:
        """
        在 IS 数据上训练模型。
        
        Args:
            is_df: IS 数据
            feature_columns: 特征列
            label_column: 标签列
        
        Returns:
            lgb.Booster: 训练好的模型
        """
        logger.info("=" * 60)
        logger.info("Training Model on In-Sample Data")
        logger.info("=" * 60)
        
        # 删除空值
        is_clean = is_df.drop_nulls(subset=feature_columns + [label_column])
        logger.info(f"Clean IS data: {len(is_clean)} rows")
        
        # 按日期排序
        is_clean = is_clean.sort(["trade_date"])
        
        # 时间序列分割 (80% 训练，20% 验证)
        n = len(is_clean)
        train_end = int(n * 0.8)
        
        train_df = is_clean.slice(0, train_end)
        val_df = is_clean.slice(train_end, n - train_end)
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
        
        # 准备训练数据
        X_train = train_df.select(feature_columns).to_numpy()
        y_train = train_df[label_column].to_numpy()
        X_val = val_df.select(feature_columns).to_numpy()
        y_val = val_df[label_column].to_numpy()
        
        # 训练模型
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_samples": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=100),
            ],
        )
        
        # 获取特征重要性
        importance = model.feature_importance(importance_type="gain")
        feature_importance = dict(zip(feature_columns, importance.tolist()))
        
        logger.info("=" * 60)
        logger.info("Top 10 Features (IS Training):")
        logger.info("=" * 60)
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (name, imp) in enumerate(sorted_features[:10], 1):
            logger.info(f"  {i}. {name}: {imp:.2f}")
        
        return model, feature_importance
    
    def run_oos_backtest(
        self,
        oos_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: list[str],
    ) -> dict[str, Any]:
        """
        在 OOS 数据上运行回测。
        
        Args:
            oos_df: OOS 数据
            model: 训练好的模型
            feature_columns: 特征列
        
        Returns:
            dict: 回测结果
        """
        logger.info("=" * 60)
        logger.info("Running Out-of-Sample Backtest")
        logger.info("=" * 60)
        
        # 使用自定义回测逻辑
        oos_clean = oos_df.drop_nulls(subset=feature_columns)
        oos_clean = oos_clean.sort(["symbol", "trade_date"])
        
        # 生成预测
        logger.info("Generating predictions...")
        X_oos = oos_clean.select(feature_columns).to_numpy()
        predictions = model.predict(X_oos)
        
        oos_with_pred = oos_clean.with_columns([
            pl.Series("prediction", predictions)
        ])
        
        # 运行回测
        backtester = Backtester(
            initial_capital=self.initial_capital,
            prediction_threshold=self.prediction_threshold,
            max_positions=self.max_positions,
            min_hold_days=self.min_hold_days,
            use_dynamic_position=self.use_dynamic_position,
        )
        
        # 自定义回测逻辑 - 使用预测值
        records_df = self._run_custom_backtest(
            oos_with_pred,
            feature_columns,
            backtester,
        )
        
        # 计算指标
        metrics = backtester.calculate_metrics(records_df)
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        return {
            "records": records_df,
            "metrics": metrics,
            "equity_curve": equity_curve,
        }
    
    def _run_custom_backtest(
        self,
        predictions_df: pl.DataFrame,
        feature_columns: list[str],
        backtester: Backtester,
    ) -> pl.DataFrame:
        """
        自定义回测逻辑，使用预训练的模型预测。
        """
        predictions_df = predictions_df.sort(["symbol", "trade_date"])
        unique_dates = predictions_df["trade_date"].unique().sort().to_list()
        
        daily_records = []
        positions: dict[str, dict] = {}
        cash = self.initial_capital
        portfolio_value = self.initial_capital
        
        hold_days: dict[str, int] = {}
        position_predictions: dict[str, float] = {}
        
        logger.info(f"Starting OOS backtest with {len(unique_dates)} trading days")
        
        for i, current_date in enumerate(unique_dates):
            # 获取当日预测
            day_data = predictions_df.filter(pl.col("trade_date") == current_date)
            if day_data.is_empty():
                continue
            
            # 选取 Top-N 预测
            day_sorted = day_data.sort("prediction", descending=True)
            top_n = day_sorted.head(backtester.max_positions)
            
            # 计算平均预测
            avg_pred = top_n["prediction"].mean()
            
            # 获取次日价格
            next_prices_open = {}
            next_prices_for_sell = {}
            if i + 1 < len(unique_dates):
                next_date = unique_dates[i + 1]
                next_data = predictions_df.filter(pl.col("trade_date") == next_date)
                if not next_data.is_empty():
                    next_prices_open = {
                        row[0]: row[1]
                        for row in next_data.select(["symbol", "open"]).iter_rows()
                    }
                    next_prices_for_sell = {
                        row[0]: row[1]
                        for row in next_data.select(["symbol", "open"]).iter_rows()
                    }
            
            # 买入逻辑
            if avg_pred > backtester.prediction_threshold:
                for row in top_n.iter_rows():
                    symbol = row[0]
                    pred = row[predictions_df.columns.index("prediction")]
                    close = row[predictions_df.columns.index("close")]
                    
                    if symbol in positions:
                        continue
                    
                    buy_price = next_prices_open.get(symbol, close)
                    position_value = cash * backtester.position_size_pct
                    shares = int(position_value / buy_price / 100) * 100
                    
                    if shares > 0:
                        positions[symbol] = {
                            "buy_price": buy_price,
                            "shares": shares,
                            "buy_date": current_date,
                        }
                        hold_days[symbol] = 0
                        position_predictions[symbol] = pred
                        cash -= buy_price * shares
            
            # 更新持仓天数
            for symbol in list(positions.keys()):
                hold_days[symbol] = hold_days.get(symbol, 0) + 1
            
            # 卖出逻辑
            for symbol, pos_info in list(positions.items()):
                if hold_days.get(symbol, 0) < backtester.min_hold_days:
                    continue
                
                if symbol in next_prices_for_sell:
                    sell_price = next_prices_for_sell[symbol]
                    shares = pos_info["shares"]
                    buy_price = pos_info["buy_price"]
                    
                    cost, commission, stamp_duty = backtester.calculate_transaction_cost(
                        buy_price, sell_price, shares
                    )
                    
                    backtester.total_commission += commission
                    backtester.total_stamp_duty += stamp_duty
                    
                    sell_value = sell_price * shares
                    profit = sell_value - buy_price * shares - cost
                    
                    daily_records.append({
                        "trade_date": current_date,
                        "sell_date": next_date if i + 1 < len(unique_dates) else current_date,
                        "symbol": symbol,
                        "action": "SELL",
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "shares": shares,
                        "cost": cost,
                        "profit": profit,
                    })
                    
                    cash += sell_value - cost
                    del positions[symbol]
                    if symbol in hold_days:
                        del hold_days[symbol]
                    if symbol in position_predictions:
                        del position_predictions[symbol]
            
            # 计算组合价值
            portfolio_value = cash
            for symbol, pos_info in positions.items():
                symbol_price = next_prices_open.get(symbol, pos_info["buy_price"])
                portfolio_value += symbol_price * pos_info["shares"]
            
            daily_records.append({
                "trade_date": current_date,
                "action": "DAILY_VALUE",
                "cash": cash,
                "portfolio_value": portfolio_value,
                "num_positions": len(positions),
            })
        
        return pl.DataFrame(daily_records)
    
    def run_is_backtest_for_comparison(
        self,
        is_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: list[str],
    ) -> dict[str, Any]:
        """
        在 IS 数据上运行回测用于对比。
        """
        logger.info("=" * 60)
        logger.info("Running In-Sample Backtest (for comparison)")
        logger.info("=" * 60)
        
        is_clean = is_df.drop_nulls(subset=feature_columns)
        is_clean = is_clean.sort(["symbol", "trade_date"])
        
        # 生成预测
        X_is = is_clean.select(feature_columns).to_numpy()
        predictions = model.predict(X_is)
        
        is_with_pred = is_clean.with_columns([
            pl.Series("prediction", predictions)
        ])
        
        backtester = Backtester(
            initial_capital=self.initial_capital,
            prediction_threshold=self.prediction_threshold,
            max_positions=self.max_positions,
            min_hold_days=self.min_hold_days,
            use_dynamic_position=self.use_dynamic_position,
        )
        
        records_df = self._run_custom_backtest(
            is_with_pred,
            feature_columns,
            backtester,
        )
        
        metrics = backtester.calculate_metrics(records_df)
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        return {
            "records": records_df,
            "metrics": metrics,
            "equity_curve": equity_curve,
        }
    
    def generate_comparison_report(
        self,
        is_results: dict[str, Any],
        oos_results: dict[str, Any],
        feature_importance: dict[str, float],
    ) -> dict[str, Any]:
        """
        生成 IS vs OOS 对比报告。
        
        Args:
            is_results: IS 回测结果
            oos_results: OOS 回测结果
            feature_importance: 特征重要性
        
        Returns:
            dict: 对比报告
        """
        logger.info("=" * 60)
        logger.info("Generating IS vs OOS Comparison Report")
        logger.info("=" * 60)
        
        is_metrics = is_results["metrics"]
        oos_metrics = oos_results["metrics"]
        
        # 计算夏普比率衰减率
        is_sharpe = is_metrics.get("sharpe_ratio", 0)
        oos_sharpe = oos_metrics.get("sharpe_ratio", 0)
        
        sharpe_decay = oos_sharpe / is_sharpe if is_sharpe != 0 else 0
        
        # 其他对比指标
        is_return = is_metrics.get("annualized_return", 0)
        oos_return = oos_metrics.get("annualized_return", 0)
        return_decay = oos_return / is_return if is_return != 0 else 0
        
        is_win_rate = is_metrics.get("win_rate", 0)
        oos_win_rate = oos_metrics.get("win_rate", 0)
        
        is_profit_factor = is_metrics.get("profit_factor", 0)
        oos_profit_factor = oos_metrics.get("profit_factor", 0)
        
        # 过拟合判断
        overfitting_status = "严重过拟合"
        if sharpe_decay > 0.8:
            overfitting_status = "稳健策略"
        elif sharpe_decay > 0.6:
            overfitting_status = "轻度过拟合"
        
        report = {
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "sharpe_decay": sharpe_decay,
            "return_decay": return_decay,
            "overfitting_status": overfitting_status,
            "feature_importance": feature_importance,
            "comparison": {
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "is_annualized_return": is_return,
                "oos_annualized_return": oos_return,
                "is_win_rate": is_win_rate,
                "oos_win_rate": oos_win_rate,
                "is_profit_factor": is_profit_factor,
                "oos_profit_factor": oos_profit_factor,
                "is_max_drawdown": is_metrics.get("max_drawdown", 0),
                "oos_max_drawdown": oos_metrics.get("max_drawdown", 0),
                "is_total_return": is_metrics.get("total_return", 0),
                "oos_total_return": oos_metrics.get("total_return", 0),
                "is_num_trading_days": is_metrics.get("num_trading_days", 0),
                "oos_num_trading_days": oos_metrics.get("num_trading_days", 0),
                "is_num_trades": is_metrics.get("num_trades", 0),
                "oos_num_trades": oos_metrics.get("num_trades", 0),
            },
        }
        
        # 打印报告
        logger.info("\n" + "=" * 80)
        logger.info("IS vs OOS PERFORMANCE COMPARISON")
        logger.info("=" * 80)
        
        header = f"{'Metric':<30} {'In-Sample':<20} {'Out-of-Sample':<20} {'Decay':<15}"
        logger.info(header)
        logger.info("-" * 80)
        
        rows = [
            ("Sharpe Ratio", f"{is_sharpe:.4f}", f"{oos_sharpe:.4f}", f"{sharpe_decay:.2%}"),
            ("Annualized Return", f"{is_return:.2%}", f"{oos_return:.2%}", f"{return_decay:.2%}"),
            ("Win Rate", f"{is_win_rate:.2%}", f"{oos_win_rate:.2%}", "-"),
            ("Profit Factor", f"{is_profit_factor:.2f}", f"{oos_profit_factor:.2f}", "-"),
            ("Max Drawdown", f"{is_metrics.get('max_drawdown', 0):.2%}", f"{oos_metrics.get('max_drawdown', 0):.2%}", "-"),
            ("Total Return", f"{is_metrics.get('total_return', 0):.2%}", f"{oos_metrics.get('total_return', 0):.2%}", "-"),
            ("Trading Days", f"{is_metrics.get('num_trading_days', 0)}", f"{oos_metrics.get('num_trading_days', 0)}", "-"),
            ("Total Trades", f"{is_metrics.get('num_trades', 0)}", f"{oos_metrics.get('num_trades', 0)}", "-"),
        ]
        
        for row_data in rows:
            logger.info(f"{row_data[0]:<30} {row_data[1]:<20} {row_data[2]:<20} {row_data[3]:<15}")
        
        logger.info("-" * 80)
        logger.info(f"Overfitting Status: {overfitting_status}")
        logger.info(f"Sharpe Decay Rate: {sharpe_decay:.2%} (threshold: 60%)")
        logger.info("=" * 80)
        
        # 保存报告
        report_path = self.output_dir / "oos_comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_path}")
        
        # 生成对比图
        self._plot_comparison(is_results["equity_curve"], oos_results["equity_curve"])
        
        return report
    
    def _plot_comparison(
        self,
        is_equity: pl.DataFrame,
        oos_equity: pl.DataFrame,
    ) -> None:
        """绘制 IS vs OOS 收益曲线对比图。"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # IS 曲线
            ax1 = axes[0]
            is_data = is_equity.sort("trade_date")
            ax1.plot(range(len(is_data)), is_data["portfolio_value"].to_list(), 'b-', linewidth=1)
            ax1.set_title(f"In-Sample Equity Curve ({self.IS_START_DATE} to {self.IS_END_DATE})", fontsize=12)
            ax1.set_xlabel("Trading Days")
            ax1.set_ylabel("Portfolio Value")
            ax1.grid(True, alpha=0.3)
            
            # OOS 曲线
            ax2 = axes[1]
            oos_data = oos_equity.sort("trade_date")
            ax2.plot(range(len(oos_data)), oos_data["portfolio_value"].to_list(), 'r-', linewidth=1)
            ax2.set_title(f"Out-of-Sample Equity Curve ({self.OOS_START_DATE} to {self.OOS_END_DATE})", fontsize=12)
            ax2.set_xlabel("Trading Days")
            ax2.set_ylabel("Portfolio Value")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = self.output_dir / "is_vs_oos_equity_curves.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to: {save_path}")
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate comparison plot: {e}")
    
    def run_full_validation(
        self,
        feature_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        运行完整验证流程。
        
        Args:
            feature_columns: 特征列列表
        
        Returns:
            dict: 验证结果
        """
        if feature_columns is None:
            feature_columns = [
                "momentum_5", "momentum_10", "momentum_20",
                "volatility_5", "volatility_20",
                "volume_ma_ratio_5", "volume_ma_ratio_20",
                "price_position_20", "price_position_60",
                "ma_deviation_5", "ma_deviation_20",
                "rsi_14", "mfi_14",
                "turnover_bias_20", "turnover_ma_ratio",
                "volume_price_divergence_5", "volume_price_divergence_20",
                "volume_price_correlation", "smart_money_flow",
                "volatility_contraction_10", "volume_shrink_ratio",
                "volume_price_stable", "accumulation_distribution_20",
            ]
        
        # Step 1: 加载并分割数据
        is_df, oos_df = self.load_and_split_data()
        
        if is_df.is_empty():
            logger.error("IS data is empty! Check date range.")
            return {}
        
        if oos_df.is_empty():
            logger.error("OOS data is empty! Check date range.")
            return {}
        
        # Step 2: 在 IS 上训练模型
        model, feature_importance = self.train_model_on_is(is_df, feature_columns)
        
        # 保存模型
        model_path = str(self.output_dir / "oos_trained_model.txt")
        model.save_model(model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Step 3: IS 回测
        is_results = self.run_is_backtest_for_comparison(is_df, model, feature_columns)
        
        # Step 4: OOS 回测
        oos_results = self.run_oos_backtest(oos_df, model, feature_columns)
        
        # Step 5: 生成对比报告
        report = self.generate_comparison_report(is_results, oos_results, feature_importance)
        
        return {
            "is_results": is_results,
            "oos_results": oos_results,
            "report": report,
            "model_path": model_path,
        }


def main() -> None:
    """主入口函数。"""
    parser = argparse.ArgumentParser(description="OOS Validator")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/parquet/features_latest.parquet",
        help="Path to features Parquet file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Prediction threshold",
    )
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=3,
        help="Minimum holding days",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/plots",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Out-of-Sample (OOS) Validation")
    logger.info("=" * 60)
    logger.info(f"Parameters:")
    logger.info(f"  Threshold: {args.threshold:.2%}")
    logger.info(f"  Min Hold Days: {args.min_hold_days}")
    logger.info(f"  Capital: ${args.capital:,.0f}")
    logger.info("=" * 60)
    
    validator = OOSValidator(
        parquet_path=args.parquet,
        initial_capital=args.capital,
        prediction_threshold=args.threshold,
        max_positions=10,
        min_hold_days=args.min_hold_days,
        use_dynamic_position=True,
        output_dir=args.output,
    )
    
    results = validator.run_full_validation()
    
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("OOS Validation Complete!")
        logger.info("=" * 60)
        
        report = results.get("report", {})
        status = report.get("overfitting_status", "Unknown")
        logger.info(f"Final Status: {status}")
        
        if status == "稳健策略":
            logger.info("✓ 策略通过 OOS 验证，可以进入模拟盘/实盘阶段")
        elif status == "轻度过拟合":
            logger.info("⚠ 策略存在轻度过拟合，建议进一步优化")
        else:
            logger.info("✗ 策略存在严重过拟合，不建议实盘")


if __name__ == "__main__":
    main()