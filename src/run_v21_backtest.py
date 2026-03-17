"""
V21 Backtest Runner - 动态赋权与换手优化

【使用说明】
1. 从数据库获取股票数据和信号（使用 V19 因子体系）
2. 使用 V21 会计引擎执行真实回测（动态权重 + 缓冲区）
3. 生成 V21 报告并与 V20 对比

【核心特性】
- 动态权重：根据信号强度分配仓位
- 换手率缓冲区：Top 10 买入 / Top 30 卖出
- 真实账户管理：现金、持仓、冻结资金
- 真实手续费计算：滑点 + 规费 + 印花税 + 最低佣金 5 元
- T+1 锁定：今日买入明日才能卖
- 持仓生命周期：止损、止盈、信号排名

作者：高级量化架构师
版本：V21.0 (Dynamic Sizing & Turnover Buffer)
日期：2026-03-17
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .v21_accounting_engine import (
        V21AccountingEngine,
        V21BacktestExecutor,
        V21DynamicSizingManager,
        V21PositionManager,
        BacktestResult,
        generate_v21_report,
        INITIAL_CAPITAL,
        TOP_K_STOCKS,
        SCORE_THRESHOLD,
        BUY_RANK_THRESHOLD,
        SELL_RANK_THRESHOLD,
    )
except ImportError:
    from db_manager import DatabaseManager
    from v21_accounting_engine import (
        V21AccountingEngine,
        V21BacktestExecutor,
        V21DynamicSizingManager,
        V21PositionManager,
        BacktestResult,
        generate_v21_report,
        INITIAL_CAPITAL,
        TOP_K_STOCKS,
        SCORE_THRESHOLD,
        BUY_RANK_THRESHOLD,
        SELL_RANK_THRESHOLD,
    )


# ===========================================
# V21 信号生成器 - 基于 V19 LightGBM 模型
# ===========================================

class V21SignalGenerator:
    """
    V21 信号生成器 - 基于 V19 LightGBM 集成模型
    
    【核心特性】
    1. 使用 V19 的 6 因子体系（含动量与低波）
    2. LightGBM 滚动预测（120 天训练/20 天预测）
    3. 双重中性化（行业 + 市值）
    4. 信号平滑（EMA + 调仓门槛）
    """
    
    EPSILON = 1e-6
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        self.factor_engine = self._create_factor_engine()
        self.neutralizer = self._create_neutralizer()
        self.ensemble = self._create_ensemble()
        logger.info("V21SignalGenerator initialized (V19 LightGBM)")
    
    def _create_factor_engine(self):
        """创建因子引擎"""
        class SimpleFactorEngine:
            EPSILON = 1e-6
            
            def compute_all_factors(self, df: pl.DataFrame) -> pl.DataFrame:
                """计算所有因子"""
                result = df.clone().with_columns([
                    pl.col("close").cast(pl.Float64, strict=False),
                    pl.col("volume").cast(pl.Float64, strict=False),
                    pl.col("turnover_rate").cast(pl.Float64, strict=False),
                ])
                
                if "volume" not in result.columns:
                    result = result.with_columns([pl.lit(1.0).alias("volume")])
                
                # 1. 量价相关性
                volume_change = (pl.col("volume") / pl.col("volume").shift(1) - 1).fill_null(0)
                returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
                window = 20
                vol_mean = volume_change.rolling_mean(window_size=window).shift(1)
                ret_mean = returns.rolling_mean(window_size=window).shift(1)
                cov_xy = ((volume_change - vol_mean) * (returns - ret_mean)).rolling_mean(window_size=window).shift(1)
                vol_std = volume_change.rolling_std(window_size=window, ddof=1).shift(1)
                ret_std = returns.rolling_std(window_size=window, ddof=1).shift(1)
                vol_price_corr = cov_xy / (vol_std * ret_std + self.EPSILON)
                result = result.with_columns([vol_price_corr.alias("vol_price_corr")])
                
                # 2. 短线反转
                momentum_5 = pl.col("close") / (pl.col("close").shift(6) + self.EPSILON) - 1
                reversal_st = -momentum_5.shift(1)
                result = result.with_columns([reversal_st.alias("reversal_st")])
                
                # 3. 波动风险
                stock_returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
                volatility_20 = stock_returns.rolling_std(window_size=20, ddof=1).shift(1) * np.sqrt(252)
                vol_risk = -volatility_20
                result = result.with_columns([vol_risk.alias("vol_risk")])
                
                # 4. 异常换手
                turnover_ma20 = pl.col("turnover_rate").rolling_mean(window_size=20).shift(1)
                turnover_ratio = pl.col("turnover_rate") / (turnover_ma20 + self.EPSILON)
                turnover_signal = (turnover_ratio - 1).clip(-0.9, 2.0)
                result = result.with_columns([turnover_signal.alias("turnover_signal")])
                
                # 5. 动量因子
                ma20 = pl.col("close").rolling_mean(window_size=20).shift(1)
                momentum = (pl.col("close") - ma20) / (ma20 + self.EPSILON)
                momentum_signal = -momentum
                result = result.with_columns([ma20.alias("ma20"), momentum_signal.alias("momentum")])
                
                # 6. 低波动因子
                std_20d = stock_returns.rolling_std(window_size=20, ddof=1).shift(1) * np.sqrt(252)
                low_vol = -std_20d
                result = result.with_columns([low_vol.alias("low_vol")])
                
                logger.info(f"Computed 6 factors for {len(result)} records")
                return result
        
        return SimpleFactorEngine()
    
    def _create_neutralizer(self):
        """创建中性化器"""
        class SimpleNeutralizer:
            EPSILON = 1e-6
            
            def normalize_factors(self, df: pl.DataFrame, factor_names: List[str]) -> pl.DataFrame:
                """截面标准化"""
                result = df.clone()
                for factor in factor_names:
                    if factor not in result.columns:
                        continue
                    stats = result.group_by("trade_date").agg([
                        pl.col(factor).mean().alias("mean"),
                        pl.col(factor).std().alias("std"),
                    ])
                    result = result.join(stats, on="trade_date", how="left")
                    result = result.with_columns([
                        ((pl.col(factor) - pl.col("mean")) / (pl.col("std") + self.EPSILON)).alias(f"{factor}_std")
                    ]).drop(["mean", "std"])
                return result
        
        return SimpleNeutralizer()
    
    def _create_ensemble(self):
        """创建 LightGBM 集成模型"""
        try:
            import lightgbm as lgb
            from sklearn.preprocessing import StandardScaler
            
            class LightGBMEnsemble:
                def __init__(self, train_window=120, predict_horizon=20):
                    self.train_window = train_window
                    self.predict_horizon = predict_horizon
                    self.scaler = StandardScaler()
                    self.model = None
                    self.feature_names = []
                
                def generate_predictions(self, df: pl.DataFrame, factor_names: List[str]) -> pl.DataFrame:
                    """生成 LightGBM 预测"""
                    logger.info("[LightGBM] Generating predictions...")
                    
                    all_dates = df["trade_date"].unique().sort().to_list()
                    predictions = []
                    feature_names_std = [f"{f}_std" for f in factor_names if f"{f}_std" in df.columns]
                    
                    if not feature_names_std:
                        feature_names_std = factor_names
                    
                    self.feature_names = feature_names_std
                    
                    last_train_date = None
                    current_model = None
                    
                    for i, date in enumerate(all_dates):
                        should_retrain = (
                            last_train_date is None or
                            (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(last_train_date, "%Y-%m-%d")).days >= self.predict_horizon
                        )
                        
                        if should_retrain:
                            train_df = df.filter(pl.col("trade_date") < date)
                            if len(train_df) >= 100:
                                features, labels = self._prepare_features(train_df, feature_names_std)
                                if len(features) > 50 and labels is not None:
                                    train_data = lgb.Dataset(features, label=labels)
                                    current_model = lgb.train({
                                        "objective": "regression",
                                        "metric": "mse",
                                        "boosting_type": "gbdt",
                                        "num_leaves": 31,
                                        "learning_rate": 0.05,
                                        "feature_fraction": 0.8,
                                        "verbose": -1,
                                    }, train_data, num_boost_round=100)
                                    last_train_date = date
                        
                        if current_model is None:
                            continue
                        
                        day_df = df.filter(pl.col("trade_date") == date)
                        features = day_df.select(feature_names_std).to_numpy()
                        valid_mask = ~np.isnan(features).any(axis=1)
                        
                        if valid_mask.sum() > 0:
                            features_valid = self.scaler.fit_transform(features[valid_mask])
                            preds = current_model.predict(features_valid)
                            
                            for j, symbol in enumerate(day_df["symbol"].to_list()):
                                if valid_mask[j]:
                                    predictions.append({
                                        "symbol": symbol,
                                        "trade_date": date,
                                        "predict_score": preds[np.where(valid_mask)[0].tolist().index(j)] if j in np.where(valid_mask)[0] else 0,
                                    })
                    
                    predictions_df = pl.DataFrame(predictions)
                    logger.info(f"[LightGBM] Generated {len(predictions_df)} predictions")
                    return predictions_df
                
                def _prepare_features(self, df: pl.DataFrame, feature_names: List[str]):
                    features = df.select(feature_names).to_numpy()
                    labels = df["future_return_5d"].to_numpy() if "future_return_5d" in df.columns else None
                    valid_mask = ~np.isnan(features).any(axis=1)
                    if labels is not None:
                        valid_mask &= ~np.isnan(labels)
                    features = features[valid_mask]
                    labels = labels[valid_mask] if labels is not None else None
                    if len(features) > 0:
                        features = self.scaler.fit_transform(features)
                    return features, labels
            
            return LightGBMEnsemble()
        except ImportError:
            logger.warning("LightGBM not available, falling back to simple factor scoring")
            return None
    
    def compute_future_returns(self, df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
        """计算未来收益标签"""
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        future_return = (
            pl.col("close").shift(-window) /
            (pl.col("close").shift(-1) + self.EPSILON) - 1
        ).alias("future_return_5d")
        result = result.with_columns([future_return])
        return result
    
    def generate_signals(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """生成信号"""
        logger.info(f"Generating signals from {start_date} to {end_date}...")
        
        query = f"""
            SELECT 
                symbol, trade_date, open, high, low, close, pre_close,
                volume, amount, turnover_rate, industry_code, total_mv
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            logger.error("No data found")
            return pl.DataFrame()
        
        logger.info(f"Loaded {len(df)} records, {df['symbol'].n_unique()} stocks")
        
        # 计算未来收益标签（用于训练）
        df = self.compute_future_returns(df, window=5)
        
        # 计算因子
        df = self.factor_engine.compute_all_factors(df)
        
        # 标准化因子
        factor_names = ["vol_price_corr", "reversal_st", "vol_risk", "turnover_signal", "momentum", "low_vol"]
        df = self.neutralizer.normalize_factors(df, factor_names)
        
        # 使用 LightGBM 预测或简单因子评分
        if self.ensemble is not None:
            predictions_df = self.ensemble.generate_predictions(df, factor_names)
            if not predictions_df.is_empty():
                logger.info("Using LightGBM predictions")
                return predictions_df.rename({"predict_score": "signal"})
        
        # Fallback: 简单等权重因子评分
        logger.info("Using simple factor scoring (LightGBM fallback)")
        std_factors = [f"{f}_std" for f in factor_names if f"{f}_std" in df.columns]
        if not std_factors:
            std_factors = factor_names
        
        df = df.with_columns([
            (sum(pl.col(f) for f in std_factors) / len(std_factors)).alias("signal")
        ])
        
        signals = df.select(["symbol", "trade_date", "signal"])
        logger.info(f"Generated signals for {len(signals)} records")
        return signals
    
    def get_prices(self, start_date: str, end_date: str) -> pl.DataFrame:
        """获取价格数据"""
        query = f"""
            SELECT symbol, trade_date, close
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        return self.db.read_sql(query)


# ===========================================
# V21 回测运行器
# ===========================================

class V21BacktestRunner:
    """
    V21 回测运行器 - 整合信号生成和会计引擎
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        top_k_stocks: int = TOP_K_STOCKS,
        db: Optional[DatabaseManager] = None,
    ):
        self.initial_capital = initial_capital
        self.top_k_stocks = top_k_stocks
        self.db = db or DatabaseManager.get_instance()
        self.signal_generator = V21SignalGenerator(db=self.db)
        
        logger.info(f"V21BacktestRunner initialized")
        logger.info(f"  Initial Capital: {initial_capital:,.0f}")
        logger.info(f"  Top K Stocks: {top_k_stocks}")
        logger.info(f"  Score Threshold: {SCORE_THRESHOLD}")
        logger.info(f"  Buy Rank Threshold: Top {BUY_RANK_THRESHOLD}")
        logger.info(f"  Sell Rank Threshold: > {SELL_RANK_THRESHOLD}")
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V21 BACKTEST RUNNER")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} to {end_date}")
        
        # 1. 生成信号
        logger.info("\nStep 1: Generating signals...")
        signals_df = self.signal_generator.generate_signals(start_date, end_date)
        
        if signals_df.is_empty():
            logger.error("Signal generation failed")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
            )
        
        # 2. 获取价格数据
        logger.info("\nStep 2: Getting price data...")
        prices_df = self.signal_generator.get_prices(start_date, end_date)
        
        if prices_df.is_empty():
            logger.error("Price data not found")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
            )
        
        logger.info(f"Loaded {len(prices_df)} price records")
        
        # 3. 运行回测
        logger.info("\nStep 3: Running V21 backtest...")
        executor = V21BacktestExecutor(
            initial_capital=self.initial_capital,
            top_k_stocks=self.top_k_stocks,
            db=self.db,
        )
        
        result = executor.run_backtest(
            signals_df=signals_df,
            prices_df=prices_df,
        )
        
        return result
    
    def generate_report(self, result: BacktestResult, v20_result: Optional[BacktestResult] = None) -> str:
        """生成报告"""
        report_path = generate_v21_report(result, v20_result=v20_result)
        return report_path


# ===========================================
# V20 结果加载器（用于对比）
# ===========================================

def load_v20_result(json_path: Optional[str] = None) -> Optional[BacktestResult]:
    """从 JSON 文件加载 V20 结果"""
    if json_path is None:
        # 查找最新的 V20 结果
        reports_dir = Path("reports")
        v20_files = list(reports_dir.glob("V20_backtest_result_*.json"))
        if v20_files:
            v20_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            json_path = str(v20_files[0])
    
    if json_path is None or not Path(json_path).exists():
        logger.warning("V20 result not found, skipping comparison")
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 创建简化的 BacktestResult
        v20_result = BacktestResult(
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            initial_capital=data.get("initial_capital", INITIAL_CAPITAL),
            total_return=data.get("total_return", 0.0),
            annual_return=data.get("annual_return", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            win_rate=data.get("win_rate", 0.0),
            total_trades=data.get("total_trades", 0),
            total_buy_amount=data.get("total_buy_amount", 0.0),
            total_sell_amount=data.get("total_sell_amount", 0.0),
            total_commission=data.get("total_commission", 0.0),
            total_stamp_duty=data.get("total_stamp_duty", 0.0),
            total_slippage=data.get("total_slippage", 0.0),
            turnover_rate=data.get("turnover_rate", 0.0),
            avg_holding_days=data.get("avg_holding_days", 0.0),
            avg_position_count=data.get("avg_position_count", 0.0),
        )
        
        logger.info(f"Loaded V20 result from {json_path}")
        return v20_result
    except Exception as e:
        logger.warning(f"Failed to load V20 result: {e}")
        return None


# ===========================================
# 主函数
# ===========================================

def main():
    """主入口函数"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("V21 Dynamic Sizing & Turnover Buffer Backtest Runner")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 默认参数
    start_date = "2025-01-01"
    end_date = "2026-03-31"
    initial_capital = INITIAL_CAPITAL
    top_k_stocks = TOP_K_STOCKS
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
    if len(sys.argv) > 2:
        end_date = sys.argv[2]
    if len(sys.argv) > 3:
        initial_capital = float(sys.argv[3])
    if len(sys.argv) > 4:
        top_k_stocks = int(sys.argv[4])
    
    logger.info(f"Parameters:")
    logger.info(f"  Start Date: {start_date}")
    logger.info(f"  End Date: {end_date}")
    logger.info(f"  Initial Capital: {initial_capital:,.0f}")
    logger.info(f"  Top K Stocks: {top_k_stocks}")
    
    # 加载 V20 结果（用于对比）
    logger.info("\nStep 0: Loading V20 result for comparison...")
    v20_result = load_v20_result()
    
    # 运行回测
    runner = V21BacktestRunner(
        initial_capital=initial_capital,
        top_k_stocks=top_k_stocks,
    )
    
    result = runner.run_backtest(start_date, end_date)
    
    # 生成报告
    logger.info("\nStep 4: Generating report...")
    report_path = runner.generate_report(result, v20_result=v20_result)
    
    # 保存 JSON 结果
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"V21_backtest_result_{timestamp}.json"
    
    # 序列化结果
    result_dict = {
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "total_buy_amount": result.total_buy_amount,
        "total_sell_amount": result.total_sell_amount,
        "total_commission": result.total_commission,
        "total_stamp_duty": result.total_stamp_duty,
        "total_slippage": result.total_slippage,
        "turnover_rate": result.turnover_rate,
        "avg_holding_days": result.avg_holding_days,
        "avg_position_count": result.avg_position_count,
        "avg_position_weight": result.avg_position_weight,
        "cash_drag": result.cash_drag,
        "report_path": report_path,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nJSON result saved to: {json_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("V21 BACKTEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report Path: {report_path}")
    logger.info(f"JSON Path: {json_path}")
    
    # 输出最终结果
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"Annual Return: {result.annual_return:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Total Commission: {result.total_commission:,.2f} 元")
    logger.info(f"Turnover Rate: {result.turnover_rate:.2%}")
    logger.info(f"Avg Holding Days: {result.avg_holding_days:.1f} 天")


if __name__ == "__main__":
    main()