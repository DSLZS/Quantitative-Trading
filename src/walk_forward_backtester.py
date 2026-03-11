#!/usr/bin/env python3
"""
Walk-Forward Backtester - Rolling window retraining for quantitative trading.

This module implements Walk-Forward Analysis (WFA) for:
- Initial training with 12 months of data
- Predicting the next month
- Rolling the window forward by 1 month
- Retraining with new data included

核心功能:
    - 滚动窗口重训（Walk-Forward Analysis）
    - 初始使用 12 个月数据训练，预测下一个月
    - 窗口向后滑动 1 个月，加入新数据重新训练
    - 覆盖 2025 年至今的数据
    - 验证模型抵御市场风格漂移的能力

【新增 - 2026-03-11】:
    防御性重训方案：通过"增强正则化"和"滚动重训"双管齐下，
    寻找策略在样本外的真正生命力。
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Optional
from loguru import logger
from datetime import datetime, timedelta
import lightgbm as lgb

try:
    from .model_trainer import ModelTrainer
    from .db_manager import DatabaseManager
except ImportError:
    from model_trainer import ModelTrainer
    from db_manager import DatabaseManager


class WalkForwardBacktester:
    """
    滚动窗口回测器，实现 Walk-Forward Analysis。
    
    工作原理:
        1. 初始训练窗口：12 个月数据
        2. 预测窗口：接下来 1 个月
        3. 滚动步长：1 个月
        4. 每次滚动后重新训练模型
        
    目的:
        - 验证模型在不断吸收新信息的情况下能否抵御市场风格漂移
        - 评估策略在样本外的真实表现
        
    使用示例:
        >>> wfa = WalkForwardBacktester()
        >>> results = wfa.run(
        ...     parquet_path="data/parquet/features_latest.parquet",
        ...     start_date="2025-01-01"
        ... )
        >>> print(f"OOS Sharpe: {results['oos_sharpe']:.2f}")
    """
    
    # 交易成本配置
    COMMISSION_RATE = 0.0005  # 佣金率：万分之五
    STAMP_DUTY_RATE = 0.001   # 印花税：千分之一（仅卖出）
    MIN_COMMISSION = 5.0      # 最低佣金 5 元
    
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        train_window_months: int = 12,
        predict_window_months: int = 1,
        roll_step_months: int = 1,
        max_positions: int = 10,
        position_size_pct: float = 0.1,
        prediction_threshold: float = 0.005,
        min_hold_days: int = 1,
        min_prediction_diff: float = 0.005,
        use_defensive_params: bool = True,
        use_shuffle_importance: bool = True,
        shuffle_importance_threshold: float = 0.0001,
    ) -> None:
        """
        初始化滚动窗口回测器。
        
        Args:
            initial_capital (float): 初始资金，默认 100 万
            train_window_months (int): 训练窗口月数，默认 12 个月
            predict_window_months (int): 预测窗口月数，默认 1 个月
            roll_step_months (int): 滚动步长，默认 1 个月
            max_positions (int): 最大持仓数量，默认 10
            position_size_pct (float): 单只股票仓位比例，默认 10%
            prediction_threshold (float): 预测收益率阈值，默认 0.5%
            min_hold_days (int): 最小持仓天数，默认 1 天
            min_prediction_diff (float): 最小预测差异，默认 0.5%
            use_defensive_params (bool): 是否使用防御性参数，默认 True
            use_shuffle_importance (bool): 是否使用排列重要性剔除因子，默认 True
            shuffle_importance_threshold (float): 排列重要性阈值，默认 0.0001
                低于此值的因子将被剔除
        """
        self.initial_capital = initial_capital
        self.train_window_months = train_window_months
        self.predict_window_months = predict_window_months
        self.roll_step_months = roll_step_months
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.prediction_threshold = prediction_threshold
        self.min_hold_days = min_hold_days
        self.min_prediction_diff = min_prediction_diff
        self.use_defensive_params = use_defensive_params
        self.use_shuffle_importance = use_shuffle_importance
        self.shuffle_importance_threshold = shuffle_importance_threshold
        
        # 防御性参数配置
        if self.use_defensive_params:
            self.max_depth = 4  # 限制在 3-5 层
            self.num_leaves = 18  # 降至 15-20
            self.subsample = 0.8  # 采样扰动
            self.colsample_bytree = 0.8  # 采样扰动
            self.lambda_l1 = 0.1  # 增加正则化
            self.lambda_l2 = 0.1  # 增加正则化
        else:
            self.max_depth = 6
            self.num_leaves = 31
            self.subsample = 1.0
            self.colsample_bytree = 1.0
            self.lambda_l1 = 0.0
            self.lambda_l2 = 0.0
        
        # 交易成本统计
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        
        # 调仓统计
        self.forced_hold_count = 0
        
        logger.info(f"WalkForwardBacktester initialized:")
        logger.info(f"  Train window: {train_window_months} months")
        logger.info(f"  Predict window: {predict_window_months} months")
        logger.info(f"  Roll step: {roll_step_months} month")
        logger.info(f"  Defensive params: {use_defensive_params}")
        logger.info(f"  Shuffle importance: {use_shuffle_importance}")
    
    def get_monthly_dates(self, df: pl.DataFrame) -> list[str]:
        """获取 DataFrame 中所有唯一的月份（格式：YYYY-MM）。"""
        dates = df["trade_date"].to_numpy()
        months = []
        for d in dates:
            if isinstance(d, str):
                month = d[:7]  # YYYY-MM
            else:
                month = str(d)[:7]
            if month not in months:
                months.append(month)
        return sorted(months)
    
    def filter_by_month_range(
        self,
        df: pl.DataFrame,
        start_month: str,
        end_month: str,
    ) -> pl.DataFrame:
        """过滤出指定月份范围内的数据。"""
        # 修复日期类型比较 - 将字符串转换为日期
        start_date = pl.lit(start_month + "-01").cast(pl.Date)
        end_date = pl.lit(end_month + "-01").cast(pl.Date)
        return df.filter(
            (pl.col("trade_date") >= start_date) &
            (pl.col("trade_date") < end_date)
        )
    
    def calculate_shuffle_importance(
        self,
        model: lgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: list[str],
        n_repeats: int = 3,
    ) -> dict[str, float]:
        """
        计算排列重要性（Shuffle Importance）。
        
        通过随机打乱特征值来评估特征重要性：
        1. 计算原始模型的基准得分（MSE）
        2. 对每个特征，随机打乱其值
        3. 用打乱后的数据计算新得分
        4. 重要性 = 新得分 - 基准得分
        5. 重复多次取平均
        
        Returns:
            dict[str, float]: 各特征的排列重要性字典
        """
        logger.info(f"Calculating Shuffle Importance for {len(feature_columns)} features...")
        
        np.random.seed(42)
        
        # 计算基准得分
        baseline_pred = model.predict(X)
        baseline_mse = np.mean((baseline_pred - y) ** 2)
        logger.info(f"Baseline MSE: {baseline_mse:.6f}")
        
        importance_dict = {}
        
        for col_idx, col in enumerate(feature_columns):
            importance_scores = []
            
            for repeat in range(n_repeats):
                # 复制数据
                X_shuffled = X.copy()
                
                # 随机打乱当前特征
                shuffled_indices = np.random.permutation(len(y))
                X_shuffled[:, col_idx] = X[:, col_idx][shuffled_indices]
                
                # 计算打乱后的得分
                shuffled_pred = model.predict(X_shuffled)
                shuffled_mse = np.mean((shuffled_pred - y) ** 2)
                
                # 重要性 = 得分下降量
                importance_scores.append(shuffled_mse - baseline_mse)
            
            # 取平均
            importance_dict[col] = np.mean(importance_scores)
        
        # 识别干扰因子
        noise_features = [
            name for name, imp in importance_dict.items()
            if imp < self.shuffle_importance_threshold
        ]
        
        if noise_features:
            logger.info(f"Noise features (shuffle importance < {self.shuffle_importance_threshold}): {len(noise_features)}")
            for name in noise_features[:10]:
                logger.info(f"  - {name}")
        
        return importance_dict
    
    def get_significant_features(
        self,
        model: lgb.Booster,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: list[str],
    ) -> list[str]:
        """
        使用排列重要性剔除"干扰因子"，返回显著特征列表。
        
        Returns:
            list[str]: 显著特征列表
        """
        if not self.use_shuffle_importance:
            return feature_columns
        
        importance_dict = self.calculate_shuffle_importance(model, X, y, feature_columns)
        
        # 保留重要性高于阈值的特征
        significant_features = [
            name for name, imp in importance_dict.items()
            if imp >= self.shuffle_importance_threshold
        ]
        
        logger.info(f"Significant features after shuffle importance filtering: {len(significant_features)}/{len(feature_columns)}")
        
        return significant_features
    
    def train_model(
        self,
        train_df: pl.DataFrame,
        feature_columns: list[str],
        label_column: str = "future_return_5",
    ) -> tuple[lgb.Booster, list[str]]:
        """
        训练 LightGBM 模型。
        
        Returns:
            tuple: (训练好的模型，使用的特征列表)
        """
        # 清理数据
        train_clean = train_df.drop_nulls(subset=feature_columns + [label_column])
        
        if len(train_clean) < 100:
            raise ValueError(f"Insufficient training data: {len(train_clean)} rows")
        
        X_train = train_clean.select(feature_columns).to_numpy()
        y_train = train_clean[label_column].to_numpy()
        
        # 防御性参数配置
        params = {
            "objective": "regression",
            "metric": "mse",
            "boosting_type": "gbdt",
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "learning_rate": 0.05,
            "min_child_samples": 100,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
        }
        
        logger.info(f"Training on {len(train_clean)} samples with defensive params")
        logger.info(f"  max_depth={self.max_depth}, num_leaves={self.num_leaves}")
        logger.info(f"  lambda_l1={self.lambda_l1}, lambda_l2={self.lambda_l2}")
        logger.info(f"  subsample={self.subsample}, colsample_bytree={self.colsample_bytree}")
        
        train_dataset = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_dataset, num_boost_round=500)
        
        logger.info(f"Training complete, best iteration: {model.best_iteration}")
        
        # 使用排列重要性剔除干扰因子
        significant_features = self.get_significant_features(model, X_train, y_train, feature_columns)
        
        return model, significant_features
    
    def calculate_transaction_cost(
        self,
        buy_price: float,
        sell_price: float,
        shares: int,
    ) -> tuple[float, float, float]:
        """计算交易成本。"""
        buy_value = buy_price * shares
        sell_value = sell_price * shares
        
        buy_commission = max(buy_value * self.COMMISSION_RATE, self.MIN_COMMISSION)
        sell_commission = max(sell_value * self.COMMISSION_RATE, self.MIN_COMMISSION)
        sell_stamp_duty = sell_value * self.STAMP_DUTY_RATE
        
        total_commission = buy_commission + sell_commission
        total_cost = total_commission + sell_stamp_duty
        
        return total_cost, total_commission, sell_stamp_duty
    
    def run(
        self,
        parquet_path: str = "data/parquet/features_latest.parquet",
        feature_columns: Optional[list[str]] = None,
        label_column: str = "future_return_5",
        start_date: str = "2025-01-01",
    ) -> dict[str, Any]:
        """
        运行完整的 Walk-Forward 回测。
        
        Args:
            parquet_path (str): Parquet 文件路径
            feature_columns (list[str], optional): 特征列名列表
            label_column (str): 标签列名
            start_date (str): 回测开始日期
            
        Returns:
            dict[str, Any]: 回测结果
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
                "volume_price_stable",
            ]
        
        logger.info("=" * 60)
        logger.info("Walk-Forward Backtest Started")
        logger.info("=" * 60)
        logger.info(f"  Data source: {parquet_path}")
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Start date: {start_date}")
        logger.info(f"  Train window: {self.train_window_months} months")
        logger.info(f"  Predict window: {self.predict_window_months} month")
        logger.info(f"  Roll step: {self.roll_step_months} month")
        logger.info("=" * 60)
        
        # 加载数据
        logger.info("Loading data...")
        df = pl.read_parquet(parquet_path)
        df = df.sort("trade_date")
        
        # 过滤 2025 年以后的数据 - 修复日期类型比较
        start_date_obj = pl.lit(start_date).cast(pl.Date)
        df = df.filter(pl.col("trade_date") >= start_date_obj)
        
        if len(df) == 0:
            raise ValueError(f"No data found after {start_date}")
        
        logger.info(f"Loaded {len(df)} rows from {start_date}")
        
        # 获取所有月份
        all_months = self.get_monthly_dates(df)
        logger.info(f"Available months: {len(all_months)}")
        logger.info(f"  Range: {all_months[0]} to {all_months[-1]}")
        
        # 初始化结果记录
        daily_records = []
        window_results = []
        
        cash = self.initial_capital
        positions: dict[str, dict] = {}
        hold_days: dict[str, int] = {}
        position_predictions: dict[str, float] = {}
        
        # Walk-Forward 循环
        logger.info("\n" + "=" * 60)
        logger.info("Starting Walk-Forward Loop")
        logger.info("=" * 60)
        
        for window_idx in range(len(all_months) - self.train_window_months):
            # 计算训练窗口和预测窗口
            train_start_month = all_months[window_idx]
            train_end_month = all_months[window_idx + self.train_window_months]
            predict_month = all_months[window_idx + self.train_window_months]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Window {window_idx + 1}: Train [{train_start_month}, {train_end_month}) -> Predict {predict_month}")
            logger.info(f"{'='*60}")
            
            # 获取训练数据
            train_df = self.filter_by_month_range(df, train_start_month, train_end_month)
            # 修复：将日期列转换为字符串后再匹配
            predict_df = df.filter(
                pl.col("trade_date").cast(pl.String).str.starts_with(predict_month)
            )
            
            if len(train_df) < 100:
                logger.warning(f"Insufficient training data for window {window_idx + 1}, skipping...")
                continue
            
            if len(predict_df) == 0:
                logger.warning(f"No prediction data for {predict_month}, skipping...")
                continue
            
            # 训练模型
            try:
                model, significant_features = self.train_model(train_df, feature_columns, label_column)
            except Exception as e:
                logger.error(f"Failed to train model for window {window_idx + 1}: {e}")
                continue
            
            # 记录窗口结果（包含 Shuffle Importance 分析结果）
            window_results.append({
                "window": window_idx + 1,
                "train_start": train_start_month,
                "train_end": train_end_month,
                "predict_month": predict_month,
                "train_samples": len(train_df),
                "predict_samples": len(predict_df),
                "significant_features": len(significant_features),
                "total_features": len(feature_columns),
            })
            
            # 在预测窗口进行回测 - 使用原始特征列而非剔除后的
            # 注意：Shuffle Importance 仅用于分析，不用于预测时的特征剔除
            # 因为模型是用所有特征训练的，预测时也必须使用相同的特征
            daily_results = self._backtest_window(
                predict_df=predict_df,
                model=model,
                feature_columns=feature_columns,  # 使用原始特征列
                cash=cash,
                positions=positions,
                hold_days=hold_days,
                position_predictions=position_predictions,
            )
            
            # 更新结果和状态
            daily_records.extend(daily_results["records"])
            cash = daily_results["cash"]
            positions = daily_results["positions"]
            hold_days = daily_results["hold_days"]
            position_predictions = daily_results["position_predictions"]
        
        # 合并结果
        if not daily_records:
            return self._empty_results()
        
        records_df = pl.DataFrame(daily_records)
        
        # 计算绩效指标
        metrics = self._calculate_metrics(records_df)
        
        # 计算 OOS 夏普比率（每个预测窗口的表现）
        oos_sharpe = self._calculate_oos_sharpe(window_results, records_df)
        
        logger.info("\n" + "=" * 60)
        logger.info("Walk-Forward Backtest Complete")
        logger.info("=" * 60)
        logger.info(f"  Total windows: {len(window_results)}")
        logger.info(f"  Total records: {len(records_df)}")
        logger.info(f"  Final value: ${cash:,.2f}")
        logger.info(f"  Total return: {(cash - self.initial_capital) / self.initial_capital:.2%}")
        logger.info(f"  OOS Sharpe: {oos_sharpe:.2f}")
        logger.info(f"  Forced hold count: {self.forced_hold_count}")
        logger.info(f"  Total transaction cost: ${self.total_commission + self.total_stamp_duty:,.2f}")
        
        return {
            "records": records_df,
            "metrics": metrics,
            "window_results": window_results,
            "oos_sharpe": oos_sharpe,
            "final_value": cash,
            "total_return": (cash - self.initial_capital) / self.initial_capital,
            "cost_analysis": {
                "total_commission": self.total_commission,
                "total_stamp_duty": self.total_stamp_duty,
                "total_cost": self.total_commission + self.total_stamp_duty,
            },
        }
    
    def _backtest_window(
        self,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: list[str],
        cash: float,
        positions: dict[str, dict],
        hold_days: dict[str, int],
        position_predictions: dict[str, float],
    ) -> dict[str, Any]:
        """
        在单个预测窗口内进行回测。
        
        Returns:
            dict: 包含记录、现金、持仓状态的字典
        """
        records = []
        predict_df = predict_df.sort(["symbol", "trade_date"])
        unique_dates = predict_df["trade_date"].unique().sort().to_list()
        
        for current_date in unique_dates:
            # 获取当日数据
            day_data = predict_df.filter(pl.col("trade_date") == current_date)
            
            if day_data.is_empty():
                continue
            
            # 生成预测
            day_clean = day_data.drop_nulls(subset=feature_columns)
            if day_clean.is_empty():
                continue
            
            X_day = day_clean.select(feature_columns).to_numpy()
            symbols = day_clean["symbol"].to_list()
            closes = day_clean["close"].to_list()
            opens = day_clean["open"].to_list() if "open" in day_clean.columns else closes
            
            predictions = model.predict(X_day)
            
            # 获取次日价格（用于成交）
            next_prices = {}
            current_idx = unique_dates.index(current_date)
            if current_idx + 1 < len(unique_dates):
                next_date = unique_dates[current_idx + 1]
                next_data = predict_df.filter(pl.col("trade_date") == next_date)
                if not next_data.is_empty():
                    next_prices = {
                        row[0]: row[1]
                        for row in next_data.select(["symbol", "open"]).iter_rows()
                    }
            
            # 选股逻辑
            pred_results = [
                {"symbol": s, "pred_return": p, "close": c, "open": o}
                for s, p, c, o in zip(symbols, predictions, closes, opens)
            ]
            pred_results.sort(key=lambda x: x["pred_return"], reverse=True)
            top_n_results = pred_results[:self.max_positions]
            
            # 更新持仓天数
            for symbol in list(positions.keys()):
                hold_days[symbol] = hold_days.get(symbol, 0) + 1
            
            # 买入逻辑
            buy_signals = [r for r in top_n_results if r["pred_return"] > self.prediction_threshold]
            
            for signal in buy_signals:
                symbol = signal["symbol"]
                buy_price = next_prices.get(symbol, signal["close"])
                
                # 惩罚性调仓逻辑
                if positions and symbol not in positions:
                    min_position_pred = min(position_predictions.values()) if position_predictions else 0
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
            
            # 卖出逻辑（使用次日开盘价）
            for symbol, pos_info in list(positions.items()):
                if hold_days.get(symbol, 0) < self.min_hold_days:
                    continue
                
                if symbol in next_prices:
                    sell_price = next_prices[symbol]
                    shares = pos_info["shares"]
                    buy_price = pos_info["buy_price"]
                    
                    cost, commission, stamp_duty = self.calculate_transaction_cost(
                        buy_price, sell_price, shares
                    )
                    
                    self.total_commission += commission
                    self.total_stamp_duty += stamp_duty
                    
                    sell_value = sell_price * shares
                    profit = sell_value - buy_price * shares - cost
                    cash += sell_value - cost
                    
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
                    del position_predictions[symbol]
            
            # 记录每日组合价值
            portfolio_value = cash
            for symbol, pos_info in positions.items():
                portfolio_value += pos_info["buy_price"] * pos_info["shares"]
            
            records.append({
                "trade_date": current_date,
                "action": "DAILY_VALUE",
                "cash": cash,
                "portfolio_value": portfolio_value,
                "num_positions": len(positions),
            })
        
        return {
            "records": records,
            "cash": cash,
            "positions": positions,
            "hold_days": hold_days,
            "position_predictions": position_predictions,
        }
    
    def _calculate_metrics(
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
                if abs(daily_return) < 1.0:
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
        
        # 波动率和夏普比率
        volatility = float(np.std(returns_array, ddof=1)) * np.sqrt(252)
        mean_return = float(np.mean(returns_array))
        
        if volatility > 1e-10:
            sharpe_ratio = (mean_return * 252 - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
        # 胜率
        trade_records = records_df.filter(pl.col("action") == "SELL")
        win_rate = 0.0
        profit_factor = 0.0
        
        if not trade_records.is_empty():
            profits = trade_records["profit"].to_list()
            winning_trades = sum(1 for p in profits if p > 0)
            win_rate = winning_trades / len(profits) if profits else 0.0
            
            gross_profit = sum(p for p in profits if p > 0)
            gross_loss = abs(sum(p for p in profits if p < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else float('inf')
        
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
            "volatility": volatility,
        }
    
    def _calculate_oos_sharpe(
        self,
        window_results: list[dict],
        records_df: pl.DataFrame,
    ) -> float:
        """
        计算 OOS 夏普比率（基于每个窗口的表现）。
        
        将每个预测窗口视为一个独立的 OOS 样本，
        计算窗口间收益率的夏普比率。
        """
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        if equity_curve.is_empty() or len(window_results) < 2:
            return 0.0
        
        # 计算每个窗口的收益率
        window_returns = []
        
        for i, window in enumerate(window_results):
            predict_month = window["predict_month"]
            
            # 获取窗口开始和结束时的组合价值 - 修复日期类型比较
            month_records = equity_curve.filter(
                pl.col("trade_date").cast(pl.String).str.starts_with(predict_month)
            )
            
            if len(month_records) < 2:
                continue
            
            month_records = month_records.sort("trade_date")
            values = month_records["portfolio_value"].to_list()
            
            if values[0] > 0:
                month_return = (values[-1] - values[0]) / values[0]
                window_returns.append(month_return)
        
        if len(window_returns) < 2:
            return 0.0
        
        # 计算 OOS 夏普比率
        returns_array = np.array(window_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        # 年化
        annualized_return = mean_return * 12
        annualized_volatility = std_return * np.sqrt(12)
        
        if annualized_volatility > 1e-10:
            oos_sharpe = (annualized_return - 0.03) / annualized_volatility
        else:
            oos_sharpe = 0.0
        
        logger.info(f"OOS Sharpe calculation: {len(window_returns)} windows, mean={mean_return:.4f}, std={std_return:.4f}")
        
        return oos_sharpe
    
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
            "volatility": 0.0,
        }
    
    def _empty_results(self) -> dict[str, Any]:
        """返回空结果字典。"""
        return {
            "records": pl.DataFrame(),
            "metrics": self._empty_metrics(),
            "window_results": [],
            "oos_sharpe": 0.0,
            "final_value": 0.0,
            "total_return": 0.0,
            "cost_analysis": {
                "total_commission": 0.0,
                "total_stamp_duty": 0.0,
                "total_cost": 0.0,
            },
        }
    
    def plot_equity_curve(
        self,
        results: dict[str, Any],
        output_path: str = "data/plots/walk_forward_equity_curve.png",
    ) -> None:
        """绘制权益曲线并保存。"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        
        records_df = results.get("records")
        if records_df is None or records_df.is_empty():
            logger.warning("No data to plot")
            return
        
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        if equity_curve.is_empty():
            return
        
        equity_curve = equity_curve.sort("trade_date")
        dates = equity_curve["trade_date"].to_list()
        values = equity_curve["portfolio_value"].to_list()
        
        # 转换日期
        date_objects = []
        for d in dates:
            if isinstance(d, str):
                try:
                    date_objects.append(datetime.strptime(d, "%Y-%m-%d"))
                except ValueError:
                    date_objects.append(datetime.strptime(d[:10], "%Y-%m-%d"))
            else:
                date_objects.append(d)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(date_objects, values, linewidth=1.5, color='#2E86AB')
        
        # 格式化
        ax.set_title("Walk-Forward Backtest - Equity Curve", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        # 添加统计信息
        stats_text = (
            f"Total Return: {results['total_return']:.2%}\n"
            f"OOS Sharpe: {results['oos_sharpe']:.2f}\n"
            f"Final Value: ${results['final_value']:,.2f}"
        )
        plt.figtext(0.99, 0.01, stats_text, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Equity curve saved to {output_path}")


def run_walk_forward_backtest(
    parquet_path: str = "data/parquet/features_latest.parquet",
    initial_capital: float = 1_000_000.0,
    use_defensive_params: bool = True,
    use_shuffle_importance: bool = True,
    start_date: str = "2025-01-01",
) -> dict[str, Any]:
    """便捷函数：运行 Walk-Forward 回测。"""
    backtester = WalkForwardBacktester(
        initial_capital=initial_capital,
        use_defensive_params=use_defensive_params,
        use_shuffle_importance=use_shuffle_importance,
    )
    return backtester.run(parquet_path, start_date=start_date)


if __name__ == "__main__":
    results = run_walk_forward_backtest()
    
    print("\n" + "=" * 60)
    print("Walk-Forward Backtest Results")
    print("=" * 60)
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  OOS Sharpe: {results['oos_sharpe']:.2f}")
    print(f"  Final Value: ${results['final_value']:,.2f}")
    print(f"  Total Windows: {len(results['window_results'])}")
    print(f"  Total Transaction Cost: ${results['cost_analysis']['total_cost']:,.2f}")