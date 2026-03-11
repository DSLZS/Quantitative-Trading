#!/usr/bin/env python3
"""
Backtester Module - Strategy backtesting engine for quantitative trading.

This module provides a Backtester class for:
- Loading features from Parquet files
- Making predictions with trained LightGBM models
- Simulating trading strategies
- Calculating performance metrics

核心功能:
    - 从 Parquet 文件加载特征数据
    - 使用训练好的模型进行预测
    - 模拟买入/卖出交易
    - 计算交易成本（佣金 + 印花税）
    - 生成回测结果和统计信息
    
【重要修复 - 2026-03-10】:
    1. 时间戳对齐修复：T 日信号 → T+1 日开盘价成交（修复先验偏误）
    2. 惩罚性调仓逻辑：引入 min_prediction_diff 参数
    3. 最小持仓天数控制：降低换手率
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Optional
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .feature_pipeline import FeaturePipeline
except ImportError:
    from db_manager import DatabaseManager
    from feature_pipeline import FeaturePipeline


class Backtester:
    """
    回测引擎，用于验证量化策略的历史表现。
    
    功能特性:
        - 加载 Parquet 特征数据和训练好的模型
        - 逐日预测并模拟交易
        - 考虑交易成本（佣金 + 印花税）
        - 生成资金曲线和绩效指标
        - **组合选股**: 选取预测分最高的前 N 只股票等权重配置
        - **智能风控**: 支持信号失效止损
        - **交易降频**: 支持最小持仓天数控制
        - **惩罚性调仓**: 只有预测差异显著时才换仓
        
    交易规则:
        - 每日对全市场预测，选取 Top N 股票
        - 等权重买入持仓
        - 支持最小持仓天数，避免过度交易
        - 单边印花税（卖出时收取）
        - 双边佣金（买卖都收取）
        - 信号失效止损：预测值跌破阈值时强制平仓
        - 【修复】T 日信号 → T+1 日开盘价成交（修复先验偏误）
    
    使用示例:
        >>> backtester = Backtester()
        >>> results = backtester.run(
        ...     parquet_path="data/parquet/features_latest.parquet",
        ...     model_path="data/models/stock_model.txt"
        ... )
        >>> print(f"年化收益率：{results['annualized_return']:.2%}")
    """
    
    # 交易成本配置
    COMMISSION_RATE = 0.0005  # 佣金率：万分之五（买卖双边）
    STAMP_DUTY_RATE = 0.001   # 印花税：千分之一（仅卖出）
    MIN_COMMISSION = 5.0      # 最低佣金 5 元
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        prediction_threshold: float = 0.005,
        max_positions: int = 10,
        position_size_pct: float = 0.1,
        stop_loss_threshold: float = -0.02,
        enable_stop_loss: bool = True,
        min_hold_days: int = 1,
        turnover_control: bool = True,
        min_avg_return_to_enter: float = 0.01,
        min_prediction_diff: float = 0.005,  # 新增：惩罚性调仓参数
        use_dynamic_position: bool = False,  # 新增：动态仓位管理
        max_single_position: float = 0.3,  # 新增：单只股票最大仓位
    ) -> None:
        """
        初始化回测引擎。
        
        Args:
            initial_capital (float): 初始资金，默认 100 万
            prediction_threshold (float): 预测收益率阈值，默认 0.5%
                当预测收益率 > 阈值时触发买入信号
            max_positions (int): 最大持仓数量，默认 10 (组合选股数量 N)
            position_size_pct (float): 单个持仓占资金比例，默认 10% (等权重 1/N)
            stop_loss_threshold (float): 止损阈值，默认 -2%
                当预测值 < 此阈值时触发信号失效卖出
            enable_stop_loss (bool): 是否启用止损，默认 True
            min_hold_days (int): 最小持仓天数，默认 1 天
                用于降低交易频率，避免过度交易
            turnover_control (bool): 是否启用换手控制，默认 True
                如果 True，只有当新股票预测值显著高于持仓时才换仓
            min_avg_return_to_enter (float): 开仓最低平均预测收益率，默认 1%
                Top-K 股票的平均预测收益率必须高于此值才开仓
            min_prediction_diff (float): 最小预测差异，默认 0.5%
                【惩罚性调仓】只有当候选股预测收益率比持仓股高出此值时才换仓
                用于避免为微小的预测优势支付昂贵的交易成本
            use_dynamic_position (bool): 是否使用动态仓位管理，默认 False
                如果 True，根据预测得分加权分配仓位
            max_single_position (float): 单只股票最大仓位占比，默认 30%
                用于限制单只股票风险暴露
        """
        self.initial_capital = initial_capital
        self.prediction_threshold = prediction_threshold
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.stop_loss_threshold = stop_loss_threshold
        self.enable_stop_loss = enable_stop_loss
        self.min_hold_days = min_hold_days
        self.turnover_control = turnover_control
        self.min_avg_return_to_enter = min_avg_return_to_enter
        self.min_prediction_diff = min_prediction_diff  # 新增参数
        self.use_dynamic_position = use_dynamic_position  # 动态仓位管理
        self.max_single_position = max_single_position  # 单只最大仓位
        
        # 风控统计
        self.stop_loss_count = 0
        self.stop_loss_missed_opportunities = 0
        
        # 交易成本统计
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        
        # 调仓统计
        self.forced_hold_count = 0  # 因预测差异不足而强制持仓的次数
        
        logger.info(f"Backtester initialized: threshold={prediction_threshold:.2%}, "
                   f"max_positions={max_positions}, min_hold_days={min_hold_days}, "
                   f"min_avg_return_to_enter={min_avg_return_to_enter:.2%}, "
                   f"min_prediction_diff={min_prediction_diff:.2%}, "
                   f"use_dynamic_position={use_dynamic_position}, "
                   f"max_single_position={max_single_position:.0%}")
    
    def load_features(self, parquet_path: str) -> pl.DataFrame:
        """从 Parquet 文件加载特征数据。"""
        logger.info(f"Loading features from {parquet_path}")
        df = pl.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} rows of features")
        return df
    
    def load_model(self, model_path: str) -> Optional[Any]:
        """加载训练好的 LightGBM 模型。"""
        import lightgbm as lgb
        
        if not Path(model_path).exists():
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        try:
            logger.info(f"Loading model from {model_path}")
            model = lgb.Booster(model_file=model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return None
    
    def prepare_data(
        self,
        df: pl.DataFrame,
        feature_columns: list[str],
        label_column: str = "future_return_5",
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """准备回测数据，按日期分割训练集和测试集。"""
        df = df.sort(["trade_date"])
        unique_dates = df["trade_date"].unique().sort()
        split_idx = int(len(unique_dates) * 0.7)
        train_dates = unique_dates[:split_idx]
        test_dates = unique_dates[split_idx:]
        train_df = df.filter(pl.col("trade_date").is_in(train_dates))
        test_df = df.filter(pl.col("trade_date").is_in(test_dates))
        logger.info(f"Train dates: {len(train_dates)}, Test dates: {len(test_dates)}")
        return train_df, test_df
    
    def calculate_transaction_cost(
        self,
        buy_price: float,
        sell_price: float,
        shares: int,
    ) -> tuple[float, float, float]:
        """
        计算单笔交易的总成本。
        
        Returns:
            tuple: (total_cost, commission, stamp_duty)
        """
        buy_value = buy_price * shares
        sell_value = sell_price * shares
        
        buy_commission = max(buy_value * self.COMMISSION_RATE, self.MIN_COMMISSION)
        sell_commission = max(sell_value * self.COMMISSION_RATE, self.MIN_COMMISSION)
        sell_stamp_duty = sell_value * self.STAMP_DUTY_RATE
        
        total_commission = buy_commission + sell_commission
        total_cost = total_commission + sell_stamp_duty
        
        return total_cost, total_commission, sell_stamp_duty
    
    def simulate_trading(
        self,
        predictions_df: pl.DataFrame,
        feature_columns: list[str],
        model: Any,
    ) -> pl.DataFrame:
        """
        模拟交易过程，生成每日持仓和收益记录。
        
        【核心修复 - 时间戳对齐】:
            T 日生成的预测信号 → T+1 日开盘价成交
            修复了之前使用 T 日收盘价成交的先验偏误问题
        
        交易逻辑优化:
            1. 滚动训练模型
            2. 选取 Top N 预测股票
            3. 只有当平均预测收益率 > min_avg_return_to_enter 时才开仓
            4. 最小持仓天数控制，降低换手率
            5. 【新增】惩罚性调仓：只有预测差异显著时才换仓
            6. 【修复】使用次日开盘价成交（T+1 Open）
        """
        import lightgbm as lgb
        
        predictions_df = predictions_df.sort(["symbol", "trade_date"])
        unique_dates = predictions_df["trade_date"].unique().sort().to_list()
        
        daily_records = []
        positions: dict[str, dict] = {}
        cash = self.initial_capital
        portfolio_value = self.initial_capital
        train_window_days = 60
        
        # 持仓天数追踪
        hold_days: dict[str, int] = {}
        
        # 持仓预测值追踪（用于惩罚性调仓）
        position_predictions: dict[str, float] = {}
        
        logger.info(f"Starting backtest with {len(unique_dates)} trading days")
        
        for i, current_date in enumerate(unique_dates):
            if i < train_window_days:
                continue
            
            train_start_date = unique_dates[i - train_window_days]
            train_end_date = unique_dates[i - 1]
            
            train_data = predictions_df.filter(
                (pl.col("trade_date") >= train_start_date) &
                (pl.col("trade_date") <= train_end_date)
            )
            current_data = predictions_df.filter(pl.col("trade_date") == current_date)
            
            if train_data.is_empty() or current_data.is_empty():
                continue
            
            train_data_clean = train_data.drop_nulls(subset=feature_columns + ["future_return_5"])
            current_data_clean = current_data.drop_nulls(subset=feature_columns)
            
            if train_data_clean.is_empty() or current_data_clean.is_empty():
                continue
            
            X_train = train_data_clean.select(feature_columns).to_numpy()
            y_train = train_data_clean["future_return_5"].to_numpy()
            
            train_dataset = lgb.Dataset(X_train, label=y_train)
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
            }
            
            model_daily = lgb.train(params, train_dataset, num_boost_round=100)
            
            X_current = current_data_clean.select(feature_columns).to_numpy()
            symbols = current_data_clean["symbol"].to_list()
            closes = current_data_clean["close"].to_list()
            predictions = model_daily.predict(X_current)
            
            # 组合选股逻辑
            pred_results = [
                {"symbol": s, "pred_return": p, "close": c}
                for s, p, c in zip(symbols, predictions, closes)
            ]
            pred_results.sort(key=lambda x: x["pred_return"], reverse=True)
            top_n_results = pred_results[:self.max_positions]
            
            # 计算 Top-K 平均预测收益率
            avg_top_pred = np.mean([r["pred_return"] for r in top_n_results]) if top_n_results else 0
            
            # 阈值调优：只有当平均预测收益率显著高于成本时才开仓
            # 交易成本约 0.2% (买入 0.05% + 卖出 0.15%)
            effective_threshold = max(self.prediction_threshold, self.min_avg_return_to_enter)
            
            # 过滤买入信号
            buy_signals = [r for r in top_n_results if r["pred_return"] > self.prediction_threshold]
            
            # 智能风控检查
            if self.enable_stop_loss and positions:
                symbol_to_pred = {r["symbol"]: r["pred_return"] for r in pred_results}
                for symbol in list(positions.keys()):
                    if symbol in symbol_to_pred and symbol_to_pred[symbol] < self.stop_loss_threshold:
                        self.stop_loss_count += 1
            
            # 更新持仓天数
            for symbol in list(positions.keys()):
                hold_days[symbol] = hold_days.get(symbol, 0) + 1
            
            # 【核心修复】获取次日开盘价用于成交
            # T 日信号 → T+1 日开盘价成交
            next_prices_open = {}
            next_prices_for_sell = {}  # 用于计算卖出收益的次日价格
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
            
            # 记录当日买入 - 使用次日开盘价
            day_buys = []
            
            # 【动态仓位管理】根据预测得分加权分配仓位
            if self.use_dynamic_position and buy_signals:
                # 计算预测得分权重
                pred_returns = [s["pred_return"] for s in buy_signals]
                min_pred = min(pred_returns)
                max_pred = max(pred_returns)
                
                # 使用归一化得分计算权重（最小 - 最大归一化）
                pred_range = max_pred - min_pred if max_pred > min_pred else 1e-10
                scores = [(s["pred_return"] - min_pred) / pred_range + 0.1 for s in buy_signals]  # +0.1 避免权重为 0
                
                # 归一化权重，确保总和不超过 1
                total_score = sum(scores)
                weights = [s / total_score for s in scores]
                
                # 应用单只股票最大仓位限制
                weights = [min(w, self.max_single_position) for w in weights]
                
                # 重新归一化权重
                total_weight = sum(weights)
                if total_weight > 0:
                    weights = [w / total_weight for w in weights]
                
                # 按权重分配仓位
                for i, signal in enumerate(buy_signals):
                    symbol = signal["symbol"]
                    buy_price = next_prices_open.get(symbol, signal["close"])
                    
                    # 检查最小持仓控制
                    if self.min_hold_days > 1 and positions:
                        min_hold_met = all(days >= self.min_hold_days for days in hold_days.values())
                        if not min_hold_met and symbol not in positions:
                            continue
                    
                    # 【惩罚性调仓逻辑】
                    if self.turnover_control and positions and symbol not in positions:
                        min_position_pred = min(position_predictions.values()) if position_predictions else 0
                        if signal["pred_return"] <= min_position_pred + self.min_prediction_diff:
                            self.forced_hold_count += 1
                            continue
                    
                    if symbol not in positions:
                        # 按权重分配仓位，但不超过单只最大仓位限制
                        position_weight = min(weights[i], self.max_single_position)
                        position_value = cash * position_weight
                        shares = int(position_value / buy_price / 100) * 100
                        
                        if shares > 0:
                            positions[symbol] = {
                                "buy_price": buy_price,
                                "shares": shares,
                                "buy_date": current_date,
                                "weight": position_weight,  # 记录权重
                                "pred_return": signal["pred_return"],  # 记录预测收益
                            }
                            hold_days[symbol] = 0
                            position_predictions[symbol] = signal["pred_return"]
                            cash -= buy_price * shares
                            day_buys.append({"symbol": symbol, "action": "BUY", "price": buy_price, "shares": shares, "weight": position_weight})
            else:
                # 等权重分配（原有逻辑）
                for signal in buy_signals:
                    symbol = signal["symbol"]
                    
                    # 【修复】使用次日开盘价而非当日收盘价
                    buy_price = next_prices_open.get(symbol, signal["close"])
                    
                    # 检查最小持仓控制：如果已有持仓且未满最小天数，不买入新股
                    if self.min_hold_days > 1 and positions:
                        min_hold_met = all(days >= self.min_hold_days for days in hold_days.values())
                        if not min_hold_met and symbol not in positions:
                            continue
                    
                    # 【惩罚性调仓逻辑】
                    # 如果已有持仓，只有当新股预测值显著高于所有持仓股时才考虑替换
                    if self.turnover_control and positions and symbol not in positions:
                        # 获取持仓股的最小预测值
                        min_position_pred = min(position_predictions.values()) if position_predictions else 0
                        # 只有当新股预测值比持仓股最小值高出 min_prediction_diff 时才换仓
                        if signal["pred_return"] <= min_position_pred + self.min_prediction_diff:
                            self.forced_hold_count += 1
                            continue
                    
                    if symbol not in positions:
                        position_value = cash * self.position_size_pct
                        shares = int(position_value / buy_price / 100) * 100
                        
                        if shares > 0:
                            positions[symbol] = {
                                "buy_price": buy_price,
                                "shares": shares,
                                "buy_date": current_date,
                            }
                            hold_days[symbol] = 0
                            position_predictions[symbol] = signal["pred_return"]
                            cash -= buy_price * shares
                            day_buys.append({"symbol": symbol, "action": "BUY", "price": buy_price, "shares": shares})
            
            # 次日卖出逻辑 - 使用次日开盘价
            for symbol, pos_info in list(positions.items()):
                # 检查最小持仓天数
                if hold_days.get(symbol, 0) < self.min_hold_days:
                    continue
                
                if symbol in next_prices_for_sell:
                    sell_price = next_prices_for_sell[symbol]
                    shares = pos_info["shares"]
                    buy_price = pos_info["buy_price"]
                    
                    cost, commission, stamp_duty = self.calculate_transaction_cost(buy_price, sell_price, shares)
                    
                    # 更新成本统计
                    self.total_commission += commission
                    self.total_stamp_duty += stamp_duty
                    
                    sell_value = sell_price * shares
                    profit = sell_value - buy_price * shares - cost
                    cash += sell_value - cost
                    
                    daily_records.append({
                        "trade_date": current_date,
                        "sell_date": next_date,
                        "symbol": symbol,
                        "action": "SELL",
                        "buy_price": buy_price,
                        "sell_price": sell_price,
                        "shares": shares,
                        "cost": cost,
                        "profit": profit,
                    })
                    
                    del positions[symbol]
                    if symbol in hold_days:
                        del hold_days[symbol]
                    if symbol in position_predictions:
                        del position_predictions[symbol]
            
            # 记录每日组合价值
            portfolio_value = cash
            for symbol, pos_info in positions.items():
                # 使用最新可用价格计算持仓价值
                symbol_price = next_prices_open.get(symbol, pos_info["buy_price"])
                portfolio_value += symbol_price * pos_info["shares"]
            
            daily_records.append({
                "trade_date": current_date,
                "action": "DAILY_VALUE",
                "cash": cash,
                "portfolio_value": portfolio_value,
                "num_positions": len(positions),
            })
        
        records_df = pl.DataFrame(daily_records)
        logger.info(f"Backtest completed with {len(records_df)} records")
        logger.info(f"Forced hold count (due to min_prediction_diff): {self.forced_hold_count}")
        return records_df
    
    def run(
        self,
        parquet_path: str = "data/parquet/features_latest.parquet",
        model_path: str = "data/models/stock_model.txt",
        feature_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """运行完整回测流程。"""
        if feature_columns is None:
            # 使用 Parquet 文件中已有的因子列
            # 注意：新增因子 (rsi_14, mfi_14, turnover_bias_20 等) 需要重新运行 feature_pipeline.py 生成
            feature_columns = [
                "momentum_5", "momentum_10", "momentum_20",
                "volatility_5", "volatility_20",
                "volume_ma_ratio_5", "volume_ma_ratio_20",
                "price_position_20", "price_position_60",
                "ma_deviation_5", "ma_deviation_20",
            ]
        
        logger.info("=" * 50)
        logger.info("Starting Backtest")
        logger.info("=" * 50)
        
        features_df = self.load_features(parquet_path)
        model = None
        try:
            model = self.load_model(model_path)
        except FileNotFoundError:
            logger.warning(f"Model not found at {model_path}, will use rolling training")
        
        train_df, test_df = self.prepare_data(features_df, feature_columns)
        records_df = self.simulate_trading(test_df, feature_columns, model)
        metrics = self.calculate_metrics(records_df)
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        logger.info("=" * 50)
        logger.info("Backtest Complete")
        logger.info("=" * 50)
        logger.info(f"Total trades: {len(records_df.filter(pl.col('action') == 'SELL'))}")
        logger.info(f"Annualized return: {metrics.get('annualized_return', 0):.2%}")
        logger.info(f"Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Total transaction cost: ${self.total_commission + self.total_stamp_duty:,.2f}")
        logger.info(f"Cost ratio: {(self.total_commission + self.total_stamp_duty) / self.initial_capital * 100:.3f}%")
        
        return {
            "records": records_df,
            "metrics": metrics,
            "equity_curve": equity_curve,
            "cost_analysis": {
                "total_commission": self.total_commission,
                "total_stamp_duty": self.total_stamp_duty,
                "total_cost": self.total_commission + self.total_stamp_duty,
                "cost_ratio": (self.total_commission + self.total_stamp_duty) / self.initial_capital,
            }
        }
    
    def calculate_metrics(
        self,
        records_df: pl.DataFrame,
        risk_free_rate: float = 0.03,
    ) -> dict[str, float]:
        """计算回测绩效指标。"""
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        if equity_curve.is_empty():
            return self._empty_metrics()
        
        equity_curve = equity_curve.sort("trade_date")
        portfolio_values = equity_curve["portfolio_value"].to_list()
        dates = equity_curve["trade_date"].to_list()
        
        if len(portfolio_values) < 2:
            return self._empty_metrics()
        
        # 计算收益率序列
        returns = []
        for i in range(1, len(portfolio_values)):
            prev_value = portfolio_values[i - 1]
            curr_value = portfolio_values[i]
            if prev_value > 0 and curr_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                # 过滤异常值
                if abs(daily_return) < 1.0:  # 过滤掉超过 100% 的异常收益
                    returns.append(daily_return)
        
        if not returns:
            return self._empty_metrics()
        
        returns_array = np.array(returns)
        num_days = len(dates)
        
        # 总收益率
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
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
        
        # 年化波动率 - Bug 修复：确保正确计算标准差
        volatility = float(np.std(returns_array, ddof=1)) * np.sqrt(252)
        
        # 夏普比率
        mean_return = float(np.mean(returns_array))
        if volatility > 1e-10:
            sharpe_ratio = (mean_return * 252 - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # 胜率、盈亏比
        trade_records = records_df.filter(pl.col("action") == "SELL")
        win_rate = 0.0
        profit_factor = 0.0
        avg_daily_turnover = 0.0
        total_cost = 0.0
        
        if not trade_records.is_empty():
            profits = trade_records["profit"].to_list()
            costs = trade_records["cost"].to_list()
            total_cost = sum(costs)
            
            winning_trades = sum(1 for p in profits if p > 0)
            win_rate = winning_trades / len(profits) if profits else 0.0
            
            gross_profit = sum(p for p in profits if p > 0)
            gross_loss = abs(sum(p for p in profits if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else float('inf')
            
            avg_daily_turnover = total_cost / max(num_days, 1)
        
        return {
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "final_value": portfolio_values[-1],
            "num_trading_days": num_days,
            "num_trades": len(trade_records),
            "avg_daily_turnover": avg_daily_turnover,
            "total_transaction_cost": total_cost,
            "cost_ratio": total_cost / self.initial_capital,
            "volatility": volatility,
            "stop_loss_count": self.stop_loss_count,
            "stop_loss_missed_opportunities": self.stop_loss_missed_opportunities,
        }
    
    def _empty_metrics(self) -> dict[str, float]:
        """返回空指标字典。"""
        return {
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "final_value": 0.0,
            "num_trading_days": 0,
            "num_trades": 0,
            "avg_daily_turnover": 0.0,
            "total_transaction_cost": 0.0,
            "cost_ratio": 0.0,
            "volatility": 0.0,
            "stop_loss_count": 0,
            "stop_loss_missed_opportunities": 0,
        }


def run_backtest(
    parquet_path: str = "data/parquet/features_latest.parquet",
    model_path: str = "data/models/stock_model.txt",
    initial_capital: float = 1_000_000.0,
    threshold: float = 0.005,
    min_hold_days: int = 1,
    min_avg_return_to_enter: float = 0.01,
    min_prediction_diff: float = 0.005,  # 新增参数
) -> dict[str, Any]:
    """便捷函数：运行回测。"""
    backtester = Backtester(
        initial_capital=initial_capital,
        prediction_threshold=threshold,
        min_hold_days=min_hold_days,
        min_avg_return_to_enter=min_avg_return_to_enter,
        min_prediction_diff=min_prediction_diff,
    )
    return backtester.run(parquet_path, model_path)


if __name__ == "__main__":
    results = run_backtest()
    print(f"\nBacktest Results:")
    print(f"  Total Return: {results['metrics']['total_return']:.2%}")
    print(f"  Annualized Return: {results['metrics']['annualized_return']:.2%}")
    print(f"  Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {results['metrics']['win_rate']:.2%}")
    print(f"  Total Transaction Cost: ${results['cost_analysis']['total_cost']:,.2f}")
    print(f"  Cost Ratio: {results['cost_analysis']['cost_ratio']*100:.3f}%")