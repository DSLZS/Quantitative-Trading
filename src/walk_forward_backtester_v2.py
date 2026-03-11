#!/usr/bin/env python3
"""
Walk-Forward Backtester V2 - Portfolio Optimization Version.

在保持现有防御性参数的基础上，通过动态选股逻辑和仓位优化提升策略的绝对收益能力。

核心优化:
    1. 分位数选股逻辑 (Quantile-based Selection)
       - 取消固定的 prediction_threshold
       - 实现 top_k 逻辑：每日选取预测得分最高的 Top 5 只股票
    
    2. 波动率自适应仓位 (Volatility Scaling)
       - 计算每只入选股票的过去 20 日波动率
       - 使用逆波动率加权分配资金
    
    3. 市场状态开关 (Regime Switch)
       - 增加大盘 20 日均线状态
       - 防守模式：仓位上限降低 50%
    
    4. 损益归因分析 (Profit Attribution)
       - 计算策略收益 vs 等权重全市场收益
       - 分解 Beta 和 Alpha 贡献

使用示例:
    >>> from src.walk_forward_backtester_v2 import WalkForwardBacktesterV2
    >>> backtester = WalkForwardBacktesterV2()
    >>> results = backtester.run()
    >>> print(f"Alpha: {results['alpha']:.2%}, Beta: {results['beta']:.2%}")
"""

import polars as pl
import numpy as np
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
from loguru import logger
from datetime import datetime, timedelta
import lightgbm as lgb

try:
    from .model_trainer import ModelTrainer
except ImportError:
    from model_trainer import ModelTrainer


class WalkForwardBacktesterV2:
    """
    滚动窗口回测器 V2 - 组合优化版本。
    
    优化功能:
        1. 分位数选股 - Top K 逻辑替代固定阈值
        2. 逆波动率加权 - 波动率小的股票多买
        3. 市场状态开关 - 20 日均线判断防守/进攻模式
        4. 损益归因 - Alpha/Beta 分解
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
        top_k: int = 5,              # 【优化 1】Top K 选股数量
        use_volatility_scaling: bool = True,  # 【优化 2】波动率自适应仓位
        volatility_window: int = 20,          # 波动率计算窗口
        use_regime_switch: bool = True,       # 【优化 3】市场状态开关
        market_benchmark: str = "000300.SH",  # 市场基准代码
        ma_window: int = 20,                  # 均线窗口
        defensive_position_limit: float = 0.5, # 防守模式仓位上限
        min_hold_days: int = 1,
        use_defensive_params: bool = True,
    ) -> None:
        """
        初始化滚动窗口回测器 V2。
        
        Args:
            initial_capital (float): 初始资金，默认 100 万
            train_window_months (int): 训练窗口月数，默认 12 个月
            predict_window_months (int): 预测窗口月数，默认 1 个月
            roll_step_months (int): 滚动步长，默认 1 个月
            top_k (int): Top K 选股数量，默认 5 只
            use_volatility_scaling (bool): 是否使用波动率自适应仓位，默认 True
            volatility_window (int): 波动率计算窗口（日数），默认 20 日
            use_regime_switch (bool): 是否使用市场状态开关，默认 True
            market_benchmark (str): 市场基准代码，默认沪深 300
            ma_window (int): 均线窗口，默认 20 日
            defensive_position_limit (float): 防守模式仓位上限比例，默认 50%
            min_hold_days (int): 最小持仓天数，默认 1 天
            use_defensive_params (bool): 是否使用防御性模型参数，默认 True
        """
        self.initial_capital = initial_capital
        self.train_window_months = train_window_months
        self.predict_window_months = predict_window_months
        self.roll_step_months = roll_step_months
        
        # 【优化 1】Top K 选股
        self.top_k = top_k
        
        # 【优化 2】波动率自适应仓位
        self.use_volatility_scaling = use_volatility_scaling
        self.volatility_window = volatility_window
        
        # 【优化 3】市场状态开关
        self.use_regime_switch = use_regime_switch
        self.market_benchmark = market_benchmark
        self.ma_window = ma_window
        self.defensive_position_limit = defensive_position_limit
        self.current_regime = "normal"  # 当前市场状态：normal/defensive
        self.is_above_ma = True  # 是否在均线上方
        
        self.min_hold_days = min_hold_days
        self.use_defensive_params = use_defensive_params
        self.position_size_pct = 0.1  # 单只股票仓位比例 10%
        
        # 防御性模型参数
        if self.use_defensive_params:
            self.max_depth = 4
            self.num_leaves = 18
            self.subsample = 0.8
            self.colsample_bytree = 0.8
            self.lambda_l1 = 0.1
            self.lambda_l2 = 0.1
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
        
        # 归因分析数据
        self.benchmark_returns: List[float] = []
        self.alpha_returns: List[float] = []
        
        logger.info(f"WalkForwardBacktesterV2 initialized:")
        logger.info(f"  Top K: {top_k} stocks")
        logger.info(f"  Volatility Scaling: {use_volatility_scaling}")
        logger.info(f"  Regime Switch: {use_regime_switch}")
        logger.info(f"  Defensive Params: {use_defensive_params}")
    
    def get_monthly_dates(self, df: pl.DataFrame) -> List[str]:
        """获取 DataFrame 中所有唯一的月份（格式：YYYY-MM）。"""
        dates = df["trade_date"].to_numpy()
        months = []
        for d in dates:
            if isinstance(d, str):
                month = d[:7]
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
        start_date = pl.lit(start_month + "-01").cast(pl.Date)
        end_date = pl.lit(end_month + "-01").cast(pl.Date)
        return df.filter(
            (pl.col("trade_date") >= start_date) &
            (pl.col("trade_date") < end_date)
        )
    
    def calculate_volatility(
        self,
        returns: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """
        计算滚动波动率。
        
        Args:
            returns: 收益率序列
            window: 滚动窗口大小
        
        Returns:
            滚动波动率序列
        """
        if len(returns) < window:
            return np.full_like(returns, np.std(returns) + 1e-6)
        
        volatilities = np.zeros_like(returns)
        for i in range(len(returns)):
            if i < window - 1:
                volatilities[i] = np.std(returns[:i+1]) + 1e-6
            else:
                volatilities[i] = np.std(returns[i-window+1:i+1]) + 1e-6
        
        return volatilities
    
    def calculate_inverse_volatility_weights(
        self,
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """
        计算逆波动率权重。
        
        Args:
            volatilities: {symbol: volatility} 字典
        
        Returns:
            {symbol: weight} 字典，权重和为 1
        """
        inv_vol = {s: 1.0 / max(v, 1e-6) for s, v in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        weights = {s: iv / total_inv_vol for s, iv in inv_vol.items()}
        return weights
    
    def check_market_regime(
        self,
        market_data: pl.DataFrame,
        current_date: datetime,
    ) -> Tuple[str, bool]:
        """
        判断当前市场状态（进攻/防守模式）。
        
        Args:
            market_data: 市场基准数据（包含 trade_date, close）
            current_date: 当前日期
        
        Returns:
            (regime, is_above_ma): 市场状态和是否在均线上方
        """
        if not self.use_regime_switch:
            return "normal", True
        
        # 获取当前日期前的数据
        cutoff = pl.lit(current_date.strftime("%Y-%m-%d")).cast(pl.Date)
        historical = market_data.filter(pl.col("trade_date") <= cutoff)
        
        if len(historical) < self.ma_window:
            return "normal", True
        
        # 计算 20 日均线
        ma20 = historical.tail(self.ma_window)["close"].mean()
        current_price = historical.tail(1)["close"][0]
        
        is_above_ma = current_price > ma20
        regime = "defensive" if not is_above_ma else "normal"
        
        return regime, is_above_ma
    
    def train_model(
        self,
        train_df: pl.DataFrame,
        feature_columns: List[str],
        label_column: str = "future_return_5",
    ) -> lgb.Booster:
        """训练 LightGBM 模型。"""
        train_clean = train_df.drop_nulls(subset=feature_columns + [label_column])
        
        if len(train_clean) < 100:
            raise ValueError(f"Insufficient training data: {len(train_clean)} rows")
        
        X_train = train_clean.select(feature_columns).to_numpy()
        y_train = train_clean[label_column].to_numpy()
        
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
        
        train_dataset = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_dataset, num_boost_round=500)
        
        logger.info(f"Training complete, best iteration: {model.best_iteration}")
        return model
    
    def calculate_transaction_cost(
        self,
        buy_price: float,
        sell_price: float,
        shares: int,
    ) -> Tuple[float, float, float]:
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
        market_parquet_path: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        label_column: str = "future_return_5",
        start_date: str = "2025-01-01",
    ) -> Dict[str, Any]:
        """
        运行完整的 Walk-Forward 回测（优化版）。
        
        Args:
            parquet_path (str): 个股特征 Parquet 文件路径
            market_parquet_path (str, optional): 市场基准 Parquet 文件路径
            feature_columns (list, optional): 特征列名列表
            label_column (str): 标签列名
            start_date (str): 回测开始日期
        
        Returns:
            dict: 回测结果，包含:
                - records: 每日交易记录
                - metrics: 绩效指标
                - attribution: 损益归因分析
                - regime_analysis: 市场状态分析
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
        logger.info("Walk-Forward Backtest V2 (Portfolio Optimization)")
        logger.info("=" * 60)
        logger.info(f"  Data source: {parquet_path}")
        logger.info(f"  Features: {len(feature_columns)}")
        logger.info(f"  Top K Selection: {self.top_k} stocks")
        logger.info(f"  Volatility Scaling: {self.use_volatility_scaling}")
        logger.info(f"  Regime Switch: {self.use_regime_switch}")
        logger.info(f"  Start date: {start_date}")
        logger.info("=" * 60)
        
        # 加载个股数据
        logger.info("Loading data...")
        df = pl.read_parquet(parquet_path)
        df = df.sort("trade_date")
        
        start_date_obj = pl.lit(start_date).cast(pl.Date)
        df = df.filter(pl.col("trade_date") >= start_date_obj)
        
        if len(df) == 0:
            raise ValueError(f"No data found after {start_date}")
        
        logger.info(f"Loaded {len(df)} rows from {start_date}")
        
        # 加载市场基准数据（用于市场状态判断）
        market_data = None
        if self.use_regime_switch and market_parquet_path:
            try:
                market_data = pl.read_parquet(market_parquet_path)
                market_data = market_data.sort("trade_date")
                market_data = market_data.filter(pl.col("trade_date") >= start_date_obj)
                logger.info(f"Loaded market data: {len(market_data)} rows")
            except Exception as e:
                logger.warning(f"Failed to load market data: {e}. Disabling regime switch.")
                self.use_regime_switch = False
        
        # 获取所有月份
        all_months = self.get_monthly_dates(df)
        logger.info(f"Available months: {len(all_months)}")
        logger.info(f"  Range: {all_months[0]} to {all_months[-1]}")
        
        # 初始化结果记录
        daily_records = []
        window_results = []
        regime_records = []  # 市场状态记录
        
        cash = self.initial_capital
        positions: Dict[str, Dict] = {}
        hold_days: Dict[str, int] = {}
        position_predictions: Dict[str, float] = {}
        position_volatilities: Dict[str, float] = {}
        
        # 计算个股历史波动率（用于逆波动率加权）
        stock_volatilities = self._calculate_stock_volatilities(df, feature_columns)
        
        # Walk-Forward 循环
        logger.info("\n" + "=" * 60)
        logger.info("Starting Walk-Forward Loop")
        logger.info("=" * 60)
        
        for window_idx in range(len(all_months) - self.train_window_months):
            train_start_month = all_months[window_idx]
            train_end_month = all_months[window_idx + self.train_window_months]
            predict_month = all_months[window_idx + self.train_window_months]
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Window {window_idx + 1}: Train [{train_start_month}, {train_end_month}) -> Predict {predict_month}")
            logger.info(f"{'='*60}")
            
            # 获取训练数据
            train_df = self.filter_by_month_range(df, train_start_month, train_end_month)
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
                model = self.train_model(train_df, feature_columns, label_column)
            except Exception as e:
                logger.error(f"Failed to train model for window {window_idx + 1}: {e}")
                continue
            
            # 记录窗口结果
            window_results.append({
                "window": window_idx + 1,
                "train_start": train_start_month,
                "train_end": train_end_month,
                "predict_month": predict_month,
                "train_samples": len(train_df),
                "predict_samples": len(predict_df),
            })
            
            # 在预测窗口进行回测
            daily_results = self._backtest_window_v2(
                predict_df=predict_df,
                model=model,
                feature_columns=feature_columns,
                market_data=market_data,
                stock_volatilities=stock_volatilities,
                cash=cash,
                positions=positions,
                hold_days=hold_days,
                position_predictions=position_predictions,
                position_volatilities=position_volatilities,
            )
            
            # 更新结果和状态
            daily_records.extend(daily_results["records"])
            cash = daily_results["cash"]
            positions = daily_results["positions"]
            hold_days = daily_results["hold_days"]
            position_predictions = daily_results["position_predictions"]
            position_volatilities = daily_results["position_volatilities"]
            
            # 记录市场状态
            if "regime_records" in daily_results:
                regime_records.extend(daily_results["regime_records"])
        
        # 合并结果
        if not daily_records:
            return self._empty_results()
        
        records_df = pl.DataFrame(daily_records)
        
        # 计算绩效指标
        metrics = self._calculate_metrics(records_df)
        
        # 损益归因分析
        attribution = self._calculate_attribution(records_df, df, feature_columns)
        
        # 市场状态分析
        regime_analysis = self._analyze_regime_performance(regime_records, records_df)
        
        # 计算 OOS 夏普比率
        oos_sharpe = self._calculate_oos_sharpe(window_results, records_df)
        
        logger.info("\n" + "=" * 60)
        logger.info("Walk-Forward Backtest Complete")
        logger.info("=" * 60)
        logger.info(f"  Total windows: {len(window_results)}")
        logger.info(f"  Total records: {len(records_df)}")
        logger.info(f"  Final value: ${cash:,.2f}")
        logger.info(f"  Total return: {(cash - self.initial_capital) / self.initial_capital:.2%}")
        logger.info(f"  OOS Sharpe: {oos_sharpe:.2f}")
        logger.info(f"  Alpha: {attribution['alpha']:.2%}")
        logger.info(f"  Beta: {attribution['beta']:.2%}")
        logger.info(f"  Total transaction cost: ${self.total_commission + self.total_stamp_duty:,.2f}")
        
        return {
            "records": records_df,
            "metrics": metrics,
            "window_results": window_results,
            "oos_sharpe": oos_sharpe,
            "final_value": cash,
            "total_return": (cash - self.initial_capital) / self.initial_capital,
            "attribution": attribution,
            "regime_analysis": regime_analysis,
            "cost_analysis": {
                "total_commission": self.total_commission,
                "total_stamp_duty": self.total_stamp_duty,
                "total_cost": self.total_commission + self.total_stamp_duty,
            },
        }
    
    def _calculate_stock_volatilities(
        self,
        df: pl.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, float]:
        """
        预计算每只股票的历史波动率。
        
        Returns:
            {symbol: avg_volatility} 字典
        """
        volatilities = {}
        
        # 按股票分组计算收益率和波动率
        for symbol in df["symbol"].unique().to_list():
            symbol_data = df.filter(pl.col("symbol") == symbol).sort("trade_date")
            
            if len(symbol_data) < self.volatility_window:
                continue
            
            # 计算收益率
            closes = symbol_data["close"].to_numpy()
            returns = np.diff(closes) / closes[:-1]
            
            # 计算波动率
            vol = np.std(returns) * np.sqrt(252)  # 年化波动率
            volatilities[symbol] = vol
        
        return volatilities
    
    def _backtest_window_v2(
        self,
        predict_df: pl.DataFrame,
        model: lgb.Booster,
        feature_columns: List[str],
        market_data: Optional[pl.DataFrame],
        stock_volatilities: Dict[str, float],
        cash: float,
        positions: Dict[str, Dict],
        hold_days: Dict[str, int],
        position_predictions: Dict[str, float],
        position_volatilities: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        在单个预测窗口内进行回测（优化版）。
        
        优化点:
            1. Top K 选股替代固定阈值
            2. 逆波动率加权分配资金
            3. 市场状态开关控制仓位
        """
        records = []
        regime_records = []
        predict_df = predict_df.sort(["symbol", "trade_date"])
        unique_dates = predict_df["trade_date"].unique().sort().to_list()
        
        # 计算市场基准收益率（用于归因）
        benchmark_returns = {}
        if market_data is not None:
            for date in unique_dates:
                date_str = date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date)[:10]
                market_day = market_data.filter(pl.col("trade_date") == date)
                if not market_day.is_empty():
                    close = market_day["close"][0]
                    if len(benchmark_returns) > 0:
                        prev_close = list(benchmark_returns.values())[-1]
                        benchmark_returns[date_str] = (close - prev_close) / prev_close
                    else:
                        benchmark_returns[date_str] = 0.0
        
        for current_date in unique_dates:
            date_str = current_date.strftime("%Y-%m-%d") if hasattr(current_date, 'strftime') else str(current_date)[:10]
            
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
            
            # 【优化 1】Top K 选股逻辑 - 替代固定阈值
            pred_results = [
                {"symbol": s, "pred_return": p, "close": c, "open": o}
                for s, p, c, o in zip(symbols, predictions, closes, opens)
            ]
            pred_results.sort(key=lambda x: x["pred_return"], reverse=True)
            
            # 选取 Top K 股票
            top_k_results = pred_results[:self.top_k]
            
            # 【优化 3】判断市场状态
            regime = "normal"
            is_above_ma = True
            if market_data is not None and self.use_regime_switch:
                regime, is_above_ma = self.check_market_regime(market_data, current_date)
            
            regime_records.append({
                "trade_date": date_str,
                "regime": regime,
                "is_above_ma": is_above_ma,
            })
            
            # 记录市场状态
            self.current_regime = regime
            self.is_above_ma = is_above_ma
            
            # 更新持仓天数
            for symbol in list(positions.keys()):
                hold_days[symbol] = hold_days.get(symbol, 0) + 1
            
            # 【优化 2】计算逆波动率权重
            selected_symbols = [r["symbol"] for r in top_k_results]
            selected_volatilities = {
                s: stock_volatilities.get(s, 0.3) for s in selected_symbols
            }
            
            if self.use_volatility_scaling:
                weights = self.calculate_inverse_volatility_weights(selected_volatilities)
            else:
                # 等权重
                weights = {s: 1.0 / len(selected_symbols) for s in selected_symbols}
            
            # 【优化 3】防守模式降低仓位上限
            position_limit = 1.0
            if regime == "defensive":
                position_limit = self.defensive_position_limit
            
            # 买入逻辑
            available_cash = cash * position_limit  # 可用资金（受市场状态限制）
            
            for signal in top_k_results:
                symbol = signal["symbol"]
                buy_price = next_prices.get(symbol, signal["close"])
                
                if symbol not in positions:
                    # 根据权重分配资金
                    weight = weights.get(symbol, 1.0 / len(top_k_results))
                    # 修复：直接使用可用资金 * 权重计算仓位
                    position_value = available_cash * weight
                    
                    shares = int(position_value / buy_price / 100) * 100
                    
                    if shares > 0 and len(positions) < self.top_k:
                        positions[symbol] = {
                            "buy_price": buy_price,
                            "shares": shares,
                            "buy_date": current_date,
                            "weight": weight,
                        }
                        hold_days[symbol] = 0
                        position_predictions[symbol] = signal["pred_return"]
                        position_volatilities[symbol] = selected_volatilities.get(symbol, 0.3)
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
                        "weight": pos_info.get("weight", 0),
                    })
                    
                    del positions[symbol]
                    del hold_days[symbol]
                    del position_predictions[symbol]
                    del position_volatilities[symbol]
            
            # 记录每日组合价值
            portfolio_value = cash
            for symbol, pos_info in positions.items():
                portfolio_value += pos_info["buy_price"] * pos_info["shares"]
            
            # 记录基准收益率
            benchmark_return = benchmark_returns.get(date_str, 0.0)
            self.benchmark_returns.append(benchmark_return)
            
            records.append({
                "trade_date": current_date,
                "action": "DAILY_VALUE",
                "cash": cash,
                "portfolio_value": portfolio_value,
                "num_positions": len(positions),
                "regime": regime,
                "benchmark_return": benchmark_return,
            })
        
        return {
            "records": records,
            "cash": cash,
            "positions": positions,
            "hold_days": hold_days,
            "position_predictions": position_predictions,
            "position_volatilities": position_volatilities,
            "regime_records": regime_records,
        }
    
    def _calculate_attribution(
        self,
        records_df: pl.DataFrame,
        original_df: pl.DataFrame,
        feature_columns: List[str],
    ) -> Dict[str, float]:
        """
        损益归因分析 - 分解 Alpha 和 Beta。
        
        计算方法:
            1. 计算等权重全市场收益率（Beta 基准）
            2. 策略收益率 - Beta = Alpha
        """
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        if equity_curve.is_empty():
            return {"alpha": 0.0, "beta": 0.0, "total_return": 0.0}
        
        equity_curve = equity_curve.sort("trade_date")
        portfolio_values = equity_curve["portfolio_value"].to_list()
        
        if len(portfolio_values) < 2:
            return {"alpha": 0.0, "beta": 0.0, "total_return": 0.0}
        
        # 策略收益率
        strategy_returns = []
        for i in range(1, len(portfolio_values)):
            prev_value = portfolio_values[i - 1]
            curr_value = portfolio_values[i]
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                if abs(ret) < 1.0:
                    strategy_returns.append(ret)
        
        if not strategy_returns:
            return {"alpha": 0.0, "beta": 0.0, "total_return": 0.0}
        
        # 计算等权重全市场收益率（Beta）
        benchmark_returns_list = equity_curve["benchmark_return"].to_list()[1:]
        
        if not benchmark_returns_list:
            # 如果没有基准数据，假设 Beta=0
            total_strategy_return = sum(strategy_returns)
            return {
                "alpha": total_strategy_return,
                "beta": 0.0,
                "total_return": total_strategy_return,
            }
        
        # 计算累计收益
        total_strategy_return = 1.0
        for r in strategy_returns:
            total_strategy_return *= (1 + r)
        total_strategy_return -= 1
        
        total_beta_return = 1.0
        for r in benchmark_returns_list:
            total_beta_return *= (1 + r)
        total_beta_return -= 1
        
        # Alpha = 策略收益 - Beta 收益
        alpha = total_strategy_return - total_beta_return
        
        logger.info(f"Attribution Analysis: Alpha={alpha:.2%}, Beta={total_beta_return:.2%}")
        
        return {
            "alpha": alpha,
            "beta": total_beta_return,
            "total_return": total_strategy_return,
        }
    
    def _analyze_regime_performance(
        self,
        regime_records: List[Dict],
        records_df: pl.DataFrame,
    ) -> Dict[str, Any]:
        """分析不同市场状态下的策略表现。"""
        if not regime_records:
            return {"normal": {}, "defensive": {}}
        
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE").sort("trade_date")
        
        normal_returns = []
        defensive_returns = []
        
        # 将 equity_curve 转换为 pandas 以便索引
        equity_pdf = equity_curve.to_pandas()
        
        for i, record in enumerate(regime_records):
            if i + 1 >= len(equity_pdf):
                continue
            
            regime = record["regime"]
            curr_value = float(equity_pdf.iloc[i]["portfolio_value"])
            next_value = float(equity_pdf.iloc[i + 1]["portfolio_value"])
            
            if curr_value > 0:
                daily_return = (next_value - curr_value) / curr_value
                if regime == "normal":
                    normal_returns.append(daily_return)
                else:
                    defensive_returns.append(daily_return)
        
        def calc_stats(returns):
            if not returns:
                return {"mean_return": 0.0, "volatility": 0.0, "sharpe": 0.0}
            mean_ret = np.mean(returns)
            vol = np.std(returns) * np.sqrt(252)
            sharpe = (mean_ret * 252 - 0.03) / vol if vol > 1e-6 else 0.0
            return {"mean_return": mean_ret, "volatility": vol, "sharpe": sharpe}
        
        return {
            "normal": calc_stats(normal_returns),
            "defensive": calc_stats(defensive_returns),
            "normal_days": len(normal_returns),
            "defensive_days": len(defensive_returns),
        }
    
    def _calculate_metrics(
        self,
        records_df: pl.DataFrame,
        risk_free_rate: float = 0.03,
    ) -> Dict[str, float]:
        """计算回测绩效指标。"""
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        if equity_curve.is_empty():
            return self._empty_metrics()
        
        equity_curve = equity_curve.sort("trade_date")
        portfolio_values = equity_curve["portfolio_value"].to_list()
        dates = equity_curve["trade_date"].to_list()
        
        if len(portfolio_values) < 2:
            return self._empty_metrics()
        
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
        
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / max(num_days, 1)) - 1
        
        peak = portfolio_values[0]
        max_drawdown = 0.0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        volatility = float(np.std(returns_array, ddof=1)) * np.sqrt(252)
        mean_return = float(np.mean(returns_array))
        
        if volatility > 1e-10:
            sharpe_ratio = (mean_return * 252 - risk_free_rate) / volatility
        else:
            sharpe_ratio = 0.0
        
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
        window_results: List[Dict],
        records_df: pl.DataFrame,
    ) -> float:
        """计算 OOS 夏普比率。"""
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE")
        
        if equity_curve.is_empty() or len(window_results) < 2:
            return 0.0
        
        window_returns = []
        
        for window in window_results:
            predict_month = window["predict_month"]
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
        
        returns_array = np.array(window_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array, ddof=1)
        
        annualized_return = mean_return * 12
        annualized_volatility = std_return * np.sqrt(12)
        
        if annualized_volatility > 1e-10:
            oos_sharpe = (annualized_return - 0.03) / annualized_volatility
        else:
            oos_sharpe = 0.0
        
        logger.info(f"OOS Sharpe calculation: {len(window_returns)} windows, mean={mean_return:.4f}, std={std_return:.4f}")
        return oos_sharpe
    
    def _empty_metrics(self) -> Dict[str, float]:
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
    
    def _empty_results(self) -> Dict[str, Any]:
        """返回空结果字典。"""
        return {
            "records": pl.DataFrame(),
            "metrics": self._empty_metrics(),
            "window_results": [],
            "oos_sharpe": 0.0,
            "final_value": 0.0,
            "total_return": 0.0,
            "attribution": {"alpha": 0.0, "beta": 0.0, "total_return": 0.0},
            "regime_analysis": {"normal": {}, "defensive": {}},
            "cost_analysis": {
                "total_commission": 0.0,
                "total_stamp_duty": 0.0,
                "total_cost": 0.0,
            },
        }
    
    def plot_comparison(
        self,
        results_v2: Dict[str, Any],
        results_v1: Optional[Dict[str, Any]] = None,
        output_path: str = "data/plots/optimization_comparison.png",
    ) -> None:
        """绘制优化前后对比图。"""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        records_df = results_v2.get("records")
        if records_df is None or records_df.is_empty():
            logger.warning("No data to plot")
            return
        
        equity_curve = records_df.filter(pl.col("action") == "DAILY_VALUE").sort("trade_date")
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
        
        # 创建双图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 图 1：权益曲线
        ax1.plot(date_objects, values, linewidth=1.5, color='#2E86AB', label='Strategy')
        ax1.set_title("Portfolio Optimization - Equity Curve", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 添加统计信息
        stats_text = (
            f"Total Return: {results_v2['total_return']:.2%}\n"
            f"OOS Sharpe: {results_v2['oos_sharpe']:.2f}\n"
            f"Alpha: {results_v2['attribution']['alpha']:.2%}\n"
            f"Beta: {results_v2['attribution']['beta']:.2%}"
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 图 2：市场状态分布
        regime_data = equity_curve.select(["trade_date", "regime"]).to_pandas()
        if not regime_data.empty and "regime" in regime_data.columns:
            regime_counts = regime_data["regime"].value_counts()
            colors = ['#4CAF50' if r == "normal" else '#FF5722' for r in regime_counts.index]
            ax2.bar(regime_counts.index, regime_counts.values, color=colors, alpha=0.7)
            ax2.set_title("Market Regime Distribution", fontsize=12)
            ax2.set_xlabel("Regime")
            ax2.set_ylabel("Days")
        
        plt.tight_layout()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison plot saved to {output_path}")


def run_walk_forward_backtest_v2(
    parquet_path: str = "data/parquet/features_latest.parquet",
    market_parquet_path: Optional[str] = None,
    initial_capital: float = 1_000_000.0,
    top_k: int = 5,
    use_volatility_scaling: bool = True,
    use_regime_switch: bool = True,
    use_defensive_params: bool = True,
    start_date: str = "2025-01-01",
) -> Dict[str, Any]:
    """便捷函数：运行优化版 Walk-Forward 回测。"""
    backtester = WalkForwardBacktesterV2(
        initial_capital=initial_capital,
        top_k=top_k,
        use_volatility_scaling=use_volatility_scaling,
        use_regime_switch=use_regime_switch,
        use_defensive_params=use_defensive_params,
    )
    return backtester.run(parquet_path, market_parquet_path, start_date=start_date)


if __name__ == "__main__":
    results = run_walk_forward_backtest_v2()
    
    print("\n" + "=" * 60)
    print("Walk-Forward Backtest V2 Results")
    print("=" * 60)
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  OOS Sharpe: {results['oos_sharpe']:.2f}")
    print(f"  Final Value: ${results['final_value']:,.2f}")
    print(f"  Alpha: {results['attribution']['alpha']:.2%}")
    print(f"  Beta: {results['attribution']['beta']:.2%}")
    print(f"  Total Windows: {len(results['window_results'])}")
    print(f"  Total Transaction Cost: ${results['cost_analysis']['total_cost']:,.2f}")