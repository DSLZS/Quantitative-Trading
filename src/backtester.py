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
        
    交易规则:
        - 每日对全市场预测，选取 Top N 股票
        - 等权重买入持仓
        - 次日开盘价卖出
        - 单边印花税（卖出时收取）
        - 双边佣金（买卖都收取）
        - 信号失效止损：预测值跌破阈值时强制平仓
    
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
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        prediction_threshold: float = 0.005,
        max_positions: int = 10,
        position_size_pct: float = 0.1,
        stop_loss_threshold: float = -0.02,
        enable_stop_loss: bool = True,
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
        """
        self.initial_capital = initial_capital
        self.prediction_threshold = prediction_threshold
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.stop_loss_threshold = stop_loss_threshold
        self.enable_stop_loss = enable_stop_loss
        
        # 风控统计
        self.stop_loss_count = 0  # 止损触发次数
        self.stop_loss_missed_opportunities = 0  # 止损后踏空次数
        
        logger.info(f"Backtester initialized with threshold={prediction_threshold:.2%}, "
                   f"max_positions={max_positions}, stop_loss={stop_loss_threshold:.2%}")
    
    def load_features(self, parquet_path: str) -> pl.DataFrame:
        """
        从 Parquet 文件加载特征数据。
        
        Args:
            parquet_path (str): Parquet 文件路径
            
        Returns:
            pl.DataFrame: 特征数据
        """
        logger.info(f"Loading features from {parquet_path}")
        df = pl.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} rows of features")
        return df
    
    def load_model(self, model_path: str) -> Optional[Any]:
        """
        加载训练好的 LightGBM 模型。
        
        Args:
            model_path (str): 模型文件路径
            
        Returns:
            LightGBM Booster: 加载的模型，如果文件不存在则返回 None
        """
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
        """
        准备回测数据，按日期分割训练集和测试集。
        
        Args:
            df (pl.DataFrame): 特征数据
            feature_columns (list[str]): 特征列名列表
            label_column (str): 标签列名
            
        Returns:
            tuple[pl.DataFrame, pl.DataFrame]: (训练集，测试集)
                训练集用于模拟历史预测，测试集用于回测
        """
        # 按日期排序
        df = df.sort(["trade_date"])
        
        # 获取唯一日期列表
        unique_dates = df["trade_date"].unique().sort()
        
        # 划分训练集和测试集（前 70% 训练，后 30% 测试）
        split_idx = int(len(unique_dates) * 0.7)
        train_dates = unique_dates[:split_idx]
        test_dates = unique_dates[split_idx:]
        
        train_df = df.filter(pl.col("trade_date").is_in(train_dates))
        test_df = df.filter(pl.col("trade_date").is_in(test_dates))
        
        logger.info(f"Train dates: {len(train_dates)}, Test dates: {len(test_dates)}")
        logger.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
        
        return train_df, test_df
    
    def calculate_transaction_cost(
        self,
        buy_price: float,
        sell_price: float,
        shares: int,
    ) -> float:
        """
        计算单笔交易的总成本。
        
        成本构成:
            - 买入佣金：buy_value * commission_rate
            - 卖出佣金：sell_value * commission_rate
            - 卖出印花税：sell_value * stamp_duty_rate
        
        Args:
            buy_price (float): 买入价格
            sell_price (float): 卖出价格
            shares (int): 股数
            
        Returns:
            float: 总交易成本
        """
        buy_value = buy_price * shares
        sell_value = sell_price * shares
        
        # 买入佣金（最低 5 元）
        buy_commission = max(buy_value * self.COMMISSION_RATE, 5.0)
        
        # 卖出佣金（最低 5 元）
        sell_commission = max(sell_value * self.COMMISSION_RATE, 5.0)
        
        # 卖出印花税（无最低限制）
        sell_stamp_duty = sell_value * self.STAMP_DUTY_RATE
        
        total_cost = buy_commission + sell_commission + sell_stamp_duty
        return total_cost
    
    def simulate_trading(
        self,
        predictions_df: pl.DataFrame,
        feature_columns: list[str],
        model: Any,
    ) -> pl.DataFrame:
        """
        模拟交易过程，生成每日持仓和收益记录。
        
        交易逻辑:
            1. 每日使用截至前一日的训练数据重新训练模型（滚动训练）
            2. 对当日数据做预测
            3. 选择预测收益率 > 阈值的股票买入
            4. 次日开盘卖出所有持仓
        
        Args:
            predictions_df (pl.DataFrame): 包含特征和标签的数据
            feature_columns (list[str]): 特征列名列表
            model (Any): LightGBM 模型（用于参考参数）
            
        Returns:
            pl.DataFrame: 每日交易记录和收益
        """
        import lightgbm as lgb
        
        # 按日期排序
        predictions_df = predictions_df.sort(["symbol", "trade_date"])
        
        # 获取唯一日期列表
        unique_dates = predictions_df["trade_date"].unique().sort().to_list()
        
        # 初始化结果记录
        daily_records = []
        
        # 每日持仓记录：{symbol: {"buy_price": price, "shares": shares}}
        positions: dict[str, dict] = {}
        
        # 资金记录
        cash = self.initial_capital
        portfolio_value = self.initial_capital
        
        # 滚动训练窗口（使用过去的数据训练）
        train_window_days = 60  # 使用 60 天数据训练
        
        logger.info(f"Starting backtest with {len(unique_dates)} trading days")
        
        for i, current_date in enumerate(unique_dates):
            # 跳过前 train_window_days 天（没有足够的训练数据）
            if i < train_window_days:
                continue
            
            # 获取训练数据（过去 train_window_days 天的数据）
            train_start_date = unique_dates[i - train_window_days]
            train_end_date = unique_dates[i - 1]  # 不包括当天
            
            train_data = predictions_df.filter(
                (pl.col("trade_date") >= train_start_date) &
                (pl.col("trade_date") <= train_end_date)
            )
            
            # 获取当日数据用于预测
            current_data = predictions_df.filter(pl.col("trade_date") == current_date)
            
            if train_data.is_empty() or current_data.is_empty():
                continue
            
            # 训练数据准备
            train_data_clean = train_data.drop_nulls(subset=feature_columns + ["future_return_5"])
            current_data_clean = current_data.drop_nulls(subset=feature_columns)
            
            if train_data_clean.is_empty() or current_data_clean.is_empty():
                continue
            
            # 训练模型（使用简化的参数快速训练）
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
            
            model_daily = lgb.train(
                params,
                train_dataset,
                num_boost_round=100,  # 简化训练
            )
            
            # 对当日数据做预测
            X_current = current_data_clean.select(feature_columns).to_numpy()
            symbols = current_data_clean["symbol"].to_list()
            closes = current_data_clean["close"].to_list()
            
            predictions = model_daily.predict(X_current)
            
            # 组合选股逻辑：选取预测分最高的前 N 名
            # 创建预测结果列表
            pred_results = []
            for j, (symbol, pred, close) in enumerate(zip(symbols, predictions, closes)):
                pred_results.append({
                    "symbol": symbol,
                    "pred_return": pred,
                    "close": close,
                })
            
            # 按预测收益率排序，选取 Top N
            pred_results.sort(key=lambda x: x["pred_return"], reverse=True)
            top_n_results = pred_results[:self.max_positions]
            
            # 过滤买入信号：只保留预测收益率 > 阈值的
            buy_signals = [r for r in top_n_results if r["pred_return"] > self.prediction_threshold]
            
            # 智能风控：检查持仓是否需要止损
            if self.enable_stop_loss and positions:
                # 获取当前持仓的预测值
                symbol_to_pred = {r["symbol"]: r["pred_return"] for r in pred_results}
                
                for symbol in list(positions.keys()):
                    if symbol in symbol_to_pred:
                        current_pred = symbol_to_pred[symbol]
                        # 信号失效止损：预测值跌破止损阈值
                        if current_pred < self.stop_loss_threshold:
                            # 触发止损，记录但不在这里卖出（会在次日卖出逻辑中处理）
                            self.stop_loss_count += 1
                            logger.debug(f"Stop loss triggered for {symbol}: pred={current_pred:.4f} < threshold={self.stop_loss_threshold:.4f}")
            
            # 记录当日买入交易
            day_buys = []
            for signal in buy_signals:
                symbol = signal["symbol"]
                buy_price = signal["close"]  # 使用收盘价作为买入价（简化）
                
                # 计算可买股数
                position_value = cash * self.position_size_pct
                shares = int(position_value / buy_price / 100) * 100  # 100 股的整数倍
                
                if shares > 0 and symbol not in positions:
                    positions[symbol] = {
                        "buy_price": buy_price,
                        "shares": shares,
                        "buy_date": current_date,
                    }
                    cash -= buy_price * shares
                    day_buys.append({
                        "symbol": symbol,
                        "action": "BUY",
                        "price": buy_price,
                        "shares": shares,
                    })
            
            # 次日卖出逻辑（在下一个交易日卖出所有持仓）
            if i + 1 < len(unique_dates):
                next_date = unique_dates[i + 1]
                next_data = predictions_df.filter(pl.col("trade_date") == next_date)
                
                if not next_data.is_empty():
                    next_prices = {
                        row[0]: row[1]
                        for row in next_data.select(["symbol", "open"]).iter_rows()
                    }
                    
                    # 卖出所有持仓
                    for symbol, pos_info in list(positions.items()):
                        if symbol in next_prices:
                            sell_price = next_prices[symbol]
                            shares = pos_info["shares"]
                            buy_price = pos_info["buy_price"]
                            
                            # 计算交易成本
                            cost = self.calculate_transaction_cost(buy_price, sell_price, shares)
                            
                            # 计算收益
                            sell_value = sell_price * shares
                            profit = sell_value - buy_price * shares - cost
                            
                            # 更新资金
                            cash += sell_value - cost
                            
                            # 记录交易
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
                            
                            # 移除持仓
                            del positions[symbol]
            
            # 记录每日组合价值
            portfolio_value = cash
            for symbol, pos_info in positions.items():
                # 估算持仓价值（使用当日收盘价）
                symbol_close = next_prices.get(symbol, pos_info["buy_price"]) if 'next_prices' in dir() else pos_info["buy_price"]
                portfolio_value += symbol_close * pos_info["shares"]
            
            daily_records.append({
                "trade_date": current_date,
                "action": "DAILY_VALUE",
                "cash": cash,
                "portfolio_value": portfolio_value,
                "num_positions": len(positions),
            })
        
        # 转换为 DataFrame
        records_df = pl.DataFrame(daily_records)
        
        logger.info(f"Backtest completed with {len(records_df)} records")
        return records_df
    
    def run(
        self,
        parquet_path: str = "data/parquet/features_latest.parquet",
        model_path: str = "data/models/stock_model.txt",
        feature_columns: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        运行完整回测流程。
        
        Args:
            parquet_path (str): 特征文件路径
            model_path (str): 模型文件路径
            feature_columns (list[str], optional): 特征列名列表
            
        Returns:
            dict[str, Any]: 回测结果，包含:
                - "records": 交易记录 DataFrame
                - "metrics": 绩效指标字典
                - "equity_curve": 资金曲线数据
        """
        # 默认特征列
        if feature_columns is None:
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
        
        # Step 1: 加载特征数据
        features_df = self.load_features(parquet_path)
        
        # Step 2: 尝试加载模型（如果不存在则使用默认参数）
        model = None
        try:
            model = self.load_model(model_path)
        except FileNotFoundError:
            logger.warning(f"Model not found at {model_path}, will use rolling training")
        
        # Step 3: 准备数据
        train_df, test_df = self.prepare_data(features_df, feature_columns)
        
        # Step 4: 模拟交易
        records_df = self.simulate_trading(test_df, feature_columns, model)
        
        # Step 5: 计算绩效指标
        metrics = self.calculate_metrics(records_df)
        
        # Step 6: 提取资金曲线
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        logger.info("=" * 50)
        logger.info("Backtest Complete")
        logger.info("=" * 50)
        logger.info(f"Total trades: {len(records_df.filter(pl.col('action') == 'SELL'))}")
        logger.info(f"Annualized return: {metrics.get('annualized_return', 0):.2%}")
        logger.info(f"Max drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        
        return {
            "records": records_df,
            "metrics": metrics,
            "equity_curve": equity_curve,
        }
    
    def calculate_metrics(
        self,
        records_df: pl.DataFrame,
        risk_free_rate: float = 0.03,
    ) -> dict[str, float]:
        """
        计算回测绩效指标。
        
        Args:
            records_df (pl.DataFrame): 交易记录
            risk_free_rate (float): 无风险利率，默认 3%
            
        Returns:
            dict[str, float]: 绩效指标字典
                - annualized_return: 年化收益率
                - max_drawdown: 最大回撤
                - sharpe_ratio: 夏普比率
                - total_return: 总收益率
                - win_rate: 胜率
                - profit_factor: 盈亏比
        """
        # 提取资金曲线
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        if equity_curve.is_empty():
            return {
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }
        
        # 按日期排序
        equity_curve = equity_curve.sort("trade_date")
        
        # 获取组合价值序列
        portfolio_values = equity_curve["portfolio_value"].to_list()
        dates = equity_curve["trade_date"].to_list()
        
        # 计算收益率序列
        returns = []
        for i in range(1, len(portfolio_values)):
            if portfolio_values[i - 1] > 0:
                daily_return = portfolio_values[i] / portfolio_values[i - 1] - 1
                returns.append(daily_return)
        
        if not returns:
            return {
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }
        
        # 总收益率
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        num_days = len(dates)
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
        
        # 夏普比率
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        if std_return > 0:
            daily_sharpe = (mean_return - risk_free_rate / 252) / std_return
            sharpe_ratio = daily_sharpe * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # 胜率、盈亏比、换手率统计
        trade_records = records_df.filter(pl.col("action") == "SELL")
        if not trade_records.is_empty():
            profits = trade_records["profit"].to_list()
            winning_trades = sum(1 for p in profits if p > 0)
            win_rate = winning_trades / len(profits) if profits else 0.0
            
            # 盈亏比
            gross_profit = sum(p for p in profits if p > 0)
            gross_loss = abs(sum(p for p in profits if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # 计算日均换手率
            total_turnover = gross_profit + gross_loss  # 总交易额（绝对值）
            avg_daily_turnover = total_turnover / max(num_days, 1)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            avg_daily_turnover = 0.0
        
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
            # 风控统计
            "stop_loss_count": self.stop_loss_count,
            "stop_loss_missed_opportunities": self.stop_loss_missed_opportunities,
        }


def run_backtest(
    parquet_path: str = "data/parquet/features_latest.parquet",
    model_path: str = "data/models/stock_model.txt",
    initial_capital: float = 1_000_000.0,
    threshold: float = 0.005,
) -> dict[str, Any]:
    """
    便捷函数：运行回测。
    
    Args:
        parquet_path (str): 特征文件路径
        model_path (str): 模型文件路径
        initial_capital (float): 初始资金
        threshold (float): 预测阈值
        
    Returns:
        dict[str, Any]: 回测结果
    """
    backtester = Backtester(
        initial_capital=initial_capital,
        prediction_threshold=threshold,
    )
    return backtester.run(parquet_path, model_path)


if __name__ == "__main__":
    # 运行回测
    results = run_backtest()
    print(f"\nBacktest Results:")
    print(f"  Total Return: {results['metrics']['total_return']:.2%}")
    print(f"  Annualized Return: {results['metrics']['annualized_return']:.2%}")
    print(f"  Max Drawdown: {results['metrics']['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {results['metrics']['win_rate']:.2%}")