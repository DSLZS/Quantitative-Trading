"""
Predict Next Day Module - Daily inference for trading signals.

This module provides functionality for:
- Loading latest market data from database
- Computing features using the feature pipeline
- Making predictions with trained model
- Generating trading signals (Buy/Hold/Sell)

核心功能:
    - 读取数据库最新行情数据
    - 自动生成特征
    - 模型预测
    - 输出交易建议（买入/持有/卖出）

使用示例:
    >>> python src/predict_next_day.py
    # 输出明天的交易建议
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional

import polars as pl
from loguru import logger

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db_manager import DatabaseManager
from feature_pipeline import FeaturePipeline
from factor_engine import FactorEngine


# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class NextDayPredictor:
    """
    次日预测器，用于生成交易信号。
    
    功能特性:
        - 从数据库读取最新行情数据
        - 使用 FeaturePipeline 计算特征
        - 加载训练好的模型进行预测
        - 生成买入/持有/卖出信号
    
    使用示例:
        >>> predictor = NextDayPredictor()
        >>> signals = predictor.predict()
        >>> print(signals)
    """
    
    # 预测阈值配置
    BUY_THRESHOLD = 0.005      # 预测收益率 > 0.5% 买入
    SELL_THRESHOLD = -0.005    # 预测收益率 < -0.5% 卖出
    
    def __init__(
        self,
        config_path: str = "config/factors.yaml",
        model_path: str = "data/models/stock_model.txt",
        symbols: Optional[list[str]] = None,
        lookback_days: int = 60,
    ) -> None:
        """
        初始化预测器。
        
        Args:
            config_path (str): 因子配置文件路径
            model_path (str): 模型文件路径
            symbols (list[str], optional): 要预测的股票列表，默认读取所有
            lookback_days (int): 回溯天数，用于计算特征，默认 60 天
        """
        self.config_path = config_path
        self.model_path = model_path
        self.symbols = symbols
        self.lookback_days = lookback_days
        
        self.db = DatabaseManager.get_instance()
        self.factor_engine = FactorEngine(config_path)
        
        logger.info(f"NextDayPredictor initialized")
        logger.info(f"Config: {config_path}, Model: {model_path}")
    
    def load_latest_data(
        self,
        table_name: str = "stock_daily",
    ) -> pl.DataFrame:
        """
        从数据库读取最新的行情数据。
        
        Args:
            table_name (str): 数据表名
            
        Returns:
            pl.DataFrame: 最新的行情数据
        """
        # 获取最新日期
        date_query = f"""
            SELECT MAX(trade_date) as latest_date 
            FROM {table_name}
        """
        date_result = self.db.read_sql(date_query)
        
        if date_result.is_empty():
            raise ValueError("No data found in database")
        
        latest_date = date_result["latest_date"][0]
        
        # 计算开始日期（回溯 lookback_days 天）
        if isinstance(latest_date, datetime):
            latest_date = latest_date.date()
        
        start_date = latest_date - timedelta(days=self.lookback_days + 30)  # 额外缓冲
        
        logger.info(f"Latest trade date: {latest_date}")
        logger.info(f"Fetching data from {start_date} to {latest_date}")
        
        # 构建查询
        if self.symbols:
            symbols_str = "', '".join(self.symbols)
            query = f"""
                SELECT * FROM {table_name}
                WHERE symbol IN ('{symbols_str}')
                AND trade_date >= '{start_date}'
                AND trade_date <= '{latest_date}'
                ORDER BY symbol, trade_date
            """
        else:
            query = f"""
                SELECT * FROM {table_name}
                WHERE trade_date >= '{start_date}'
                AND trade_date <= '{latest_date}'
                ORDER BY symbol, trade_date
            """
        
        df = self.db.read_sql(query)
        logger.info(f"Loaded {len(df)} rows of data")
        
        return df
    
    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算特征因子。
        
        Args:
            df (pl.DataFrame): 原始行情数据
            
        Returns:
            pl.DataFrame: 包含特征的数据
        """
        logger.info("Computing features...")
        
        # 准备数据
        df = self._prepare_data(df)
        
        # 计算因子
        df_with_factors = self.factor_engine.compute_factors(df)
        
        logger.info(f"Computed {len(self.factor_engine.factors)} factors")
        
        return df_with_factors
    
    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        准备数据用于因子计算。
        
        Args:
            df (pl.DataFrame): 原始数据
            
        Returns:
            pl.DataFrame: 准备好的数据
        """
        # 确保必要的列
        required_columns = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 转换数值列为 Float64
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "adj_factor"]
        for col in numeric_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        # 计算 pct_change
        if "pct_change" not in df.columns:
            df = df.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1)
                .over("symbol")
                .alias("pct_change")
            )
        
        # 排序
        df = df.sort(["symbol", "trade_date"])
        
        return df
    
    def load_model(self) -> Any:
        """
        加载训练好的 LightGBM 模型。
        
        Returns:
            LightGBM Booster: 加载的模型
            
        Raises:
            FileNotFoundError: 如果模型文件不存在
        """
        import lightgbm as lgb
        
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading model from {self.model_path}")
        model = lgb.Booster(model_file=str(model_path))
        logger.info("Model loaded successfully")
        
        return model
    
    def predict(
        self,
        feature_columns: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        执行预测并生成交易信号。
        
        Args:
            feature_columns (list[str], optional): 特征列名列表
            
        Returns:
            pl.DataFrame: 预测结果和交易信号
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
        logger.info("Starting Next Day Prediction")
        logger.info("=" * 50)
        
        # Step 1: 加载最新数据
        df = self.load_latest_data()
        
        if df.is_empty():
            logger.warning("No data loaded")
            return pl.DataFrame()
        
        # Step 2: 计算特征
        df_with_factors = self.compute_features(df)
        
        # Step 3: 获取每个股票最新一天的数据
        latest_df = df_with_factors.sort("trade_date").group_by("symbol").last()
        
        # Step 4: 加载模型并预测
        try:
            model = self.load_model()
            
            # 准备预测数据
            X_pred = latest_df.select(feature_columns)
            
            # 删除有空值的行
            valid_mask = ~X_pred.select(pl.any_horizontal(pl.col(feature_columns).is_null()))
            latest_df_valid = latest_df.filter(valid_mask)
            X_pred_valid = X_pred.filter(valid_mask)
            
            if X_pred_valid.is_empty():
                logger.warning("No valid data for prediction after removing nulls")
                return pl.DataFrame()
            
            # 预测
            predictions = model.predict(X_pred_valid.to_numpy())
            
            # 添加预测结果
            latest_df_valid = latest_df_valid.with_columns(
                pl.Series("prediction", predictions)
            )
            
        except FileNotFoundError:
            logger.warning(f"Model not found, cannot make predictions")
            latest_df_valid = latest_df.with_columns(
                pl.lit(None).alias("prediction")
            )
        
        # Step 5: 生成交易信号
        latest_df_valid = self._generate_signals(latest_df_valid)
        
        # Step 6: 选择输出列
        output_columns = [
            "symbol", "trade_date", "close", "prediction",
            "signal", "signal_strength"
        ]
        
        result = latest_df_valid.select(output_columns)
        
        logger.info("=" * 50)
        logger.info("Prediction Complete")
        logger.info("=" * 50)
        
        # 打印信号统计
        if not result.is_empty():
            buy_count = len(result.filter(pl.col("signal") == "BUY"))
            sell_count = len(result.filter(pl.col("signal") == "SELL"))
            hold_count = len(result.filter(pl.col("signal") == "HOLD"))
            
            logger.info(f"Signals: BUY={buy_count}, HOLD={hold_count}, SELL={sell_count}")
        
        return result
    
    def _generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        根据预测值生成交易信号。
        
        Args:
            df (pl.DataFrame): 包含预测值的数据
            
        Returns:
            pl.DataFrame: 添加了交易信号的数据
        """
        # 检查 prediction 列是否存在以及是否有有效值
        if "prediction" not in df.columns:
            df = df.with_columns(
                pl.lit(None).alias("prediction")
            )
        
        # 初始化信号列
        df = df.with_columns(
            pl.lit("HOLD").alias("signal"),
            pl.lit(0.0).alias("signal_strength")
        )
        
        # 过滤出有有效预测值的行
        valid_preds = df.filter(pl.col("prediction").is_not_null())
        null_preds = df.filter(pl.col("prediction").is_null())
        
        if not valid_preds.is_empty():
            # 生成信号
            valid_preds = valid_preds.with_columns(
                pl.when(pl.col("prediction") > self.BUY_THRESHOLD)
                .then(pl.lit("BUY"))
                .when(pl.col("prediction") < self.SELL_THRESHOLD)
                .then(pl.lit("SELL"))
                .otherwise(pl.col("signal"))
                .alias("signal")
            )
            
            # 计算信号强度（预测值的绝对值）
            valid_preds = valid_preds.with_columns(
                pl.col("prediction").abs().alias("signal_strength")
            )
            
            # 合并结果
            df = pl.concat([valid_preds, null_preds])
        
        return df
    
    def get_top_signals(
        self,
        result: pl.DataFrame,
        top_n: int = 10,
    ) -> pl.DataFrame:
        """
        获取最强的买入信号。
        
        Args:
            result (pl.DataFrame): 预测结果
            top_n (int): 返回前 N 个信号
            
        Returns:
            pl.DataFrame: 最强的买入信号
        """
        if result.is_empty():
            return pl.DataFrame()
        
        # 过滤买入信号并按预测值排序
        buy_signals = result.filter(pl.col("signal") == "BUY")
        
        if buy_signals.is_empty():
            logger.info("No BUY signals found")
            return pl.DataFrame()
        
        top_signals = buy_signals.sort("prediction", descending=True).head(top_n)
        
        return top_signals


def run_prediction(
    symbols: Optional[list[str]] = None,
    model_path: str = "data/models/stock_model.txt",
    top_n: int = 10,
) -> dict[str, Any]:
    """
    便捷函数：运行次日预测。
    
    Args:
        symbols (list[str], optional): 要预测的股票列表
        model_path (str): 模型文件路径
        top_n (int): 返回前 N 个买入信号
        
    Returns:
        dict[str, Any]: 预测结果
    """
    predictor = NextDayPredictor(
        symbols=symbols,
        model_path=model_path,
    )
    
    # 执行预测
    result = predictor.predict()
    
    # 获取最强信号
    top_signals = predictor.get_top_signals(result, top_n)
    
    return {
        "all_signals": result,
        "top_signals": top_signals,
    }


def print_prediction_summary(result: pl.DataFrame) -> None:
    """
    打印预测结果摘要。
    
    Args:
        result (pl.DataFrame): 预测结果
    """
    if result.is_empty():
        print("\nNo prediction results available")
        return
    
    print("\n" + "=" * 60)
    print("NEXT DAY TRADING SIGNALS")
    print("=" * 60)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total stocks analyzed: {len(result)}")
    
    # 信号统计
    buy_count = len(result.filter(pl.col("signal") == "BUY"))
    sell_count = len(result.filter(pl.col("signal") == "SELL"))
    hold_count = len(result.filter(pl.col("signal") == "HOLD"))
    
    print(f"\nSignal Summary:")
    print(f"  BUY:  {buy_count} stocks")
    print(f"  HOLD: {hold_count} stocks")
    print(f"  SELL: {sell_count} stocks")
    
    # 显示最强的买入信号
    buy_signals = result.filter(pl.col("signal") == "BUY").sort("prediction", descending=True)
    
    if not buy_signals.is_empty():
        print(f"\nTop BUY Signals:")
        print("-" * 60)
        print(f"{'Symbol':<15} {'Close':>10} {'Prediction':>12} {'Strength':>10}")
        print("-" * 60)
        
        for row in buy_signals.head(10).iter_rows():
            symbol = row[0]
            close = row[2]
            pred = row[3]
            strength = row[5]
            print(f"{symbol:<15} {close:>10.2f} {pred:>12.4f} {strength:>10.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # 运行预测
    try:
        results = run_prediction()
        
        # 打印摘要
        print_prediction_summary(results["all_signals"])
        
        # 显示最强信号
        if not results["top_signals"].is_empty():
            print("\nRecommended stocks to watch tomorrow:")
            for row in results["top_signals"].iter_rows():
                print(f"  - {row[0]}: predicted return {row[3]:.2%}")
                
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.info("Please train the model first using model_trainer.py")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise