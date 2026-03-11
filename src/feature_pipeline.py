"""
Feature Pipeline Module - ETL pipeline for computing and storing features.

This module provides a FeaturePipeline class for:
- Reading price data from MySQL database
- Computing factors using FactorEngine
- Data cleaning (null handling)
- Persisting features to Parquet format

核心功能:
    - 从 MySQL 读取股票日线数据
    - 使用 FactorEngine 计算因子
    - 自动处理空值 (forward fill)
    - 保存为压缩 Parquet 文件
"""

import polars as pl
from pathlib import Path
from typing import Optional
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .factor_engine import FactorEngine
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine


class FeaturePipeline:
    """
    特征计算管道，用于从原始数据生成机器学习特征。
    
    功能特性:
        - 从 MySQL 数据库读取股票日线数据
        - 使用 FactorEngine 计算配置的因子
        - 自动处理空值 (前向填充)
        - 保存为压缩 Parquet 文件
        
    使用示例:
        >>> pipeline = FeaturePipeline("config/factors.yaml")
        >>> pipeline.run(output_path="data/parquet/features_latest.parquet")
    """
    
    def __init__(
        self,
        config_path: str = "config/factors.yaml",
        fill_null_strategy: str = "forward",
    ) -> None:
        """
        初始化特征管道。
        
        Args:
            config_path (str): 因子配置文件路径，默认 "config/factors.yaml"
            fill_null_strategy (str): 空值填充策略，默认 "forward" (前向填充)
                支持："forward", "backward", "zero", "mean"
        """
        self.config_path = config_path
        self.fill_null_strategy = fill_null_strategy
        self.db = DatabaseManager.get_instance()
        self.factor_engine = FactorEngine(config_path)
        
        logger.info(f"FeaturePipeline initialized with config: {config_path}")
    
    def read_data_from_db(
        self,
        table_name: str = "stock_daily",
        symbols: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        从 MySQL 数据库读取股票日线数据。
        
        Args:
            table_name (str): 数据表名，默认 "stock_daily"
            symbols (list[str], optional): 股票代码列表，默认读取所有
                示例：["000001.SZ", "600519.SH"]
            start_date (str, optional): 开始日期，格式 "YYYY-MM-DD"
            end_date (str, optional): 结束日期，格式 "YYYY-MM-DD"
        
        Returns:
            pl.DataFrame: 包含日线数据的 DataFrame
                列：symbol, trade_date, open, high, low, close, volume, amount, adj_factor
        
        注意:
            - 数据按 symbol 和 trade_date 排序
            - 使用分块读取避免内存溢出
        """
        # 构建 WHERE 子句
        conditions = []
        
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        
        if start_date:
            conditions.append(f"trade_date >= '{start_date}'")
        
        if end_date:
            conditions.append(f"trade_date <= '{end_date}'")
        
        # 构建查询
        query = f"SELECT * FROM {table_name}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY symbol, trade_date"
        
        logger.info(f"Reading data from {table_name}...")
        df = self.db.read_sql(query)
        
        logger.info(f"Loaded {len(df)} rows from database")
        return df
    
    def prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        准备数据用于因子计算。
        
        此方法执行:
        1. 确保必要的列存在
        2. 转换数值列为 Float64 (修复 MySQL decimal 类型问题)
        3. 计算 pct_change (如果不存在)
        4. 按 symbol 分组排序
        
        Args:
            df (pl.DataFrame): 原始日线数据
        
        Returns:
            pl.DataFrame: 准备好的数据
        """
        # 确保必要的列
        required_columns = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 转换数值列为 Float64 (修复 MySQL decimal 类型不支持 rolling 操作的问题)
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "adj_factor", "turnover_rate", "pre_close", "change", "pct_chg", "vol_ratio"]
        for col in numeric_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        # 计算 pct_change (涨跌幅) 如果不存在
        if "pct_change" not in df.columns:
            df = df.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1)
                .over("symbol")
                .alias("pct_change")
            )
            logger.debug("Computed pct_change column")
        
        # 按 symbol 和 trade_date 排序
        df = df.sort(["symbol", "trade_date"])
        
        return df
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算所有配置的因子。
        
        Args:
            df (pl.DataFrame): 准备好的日线数据
        
        Returns:
            pl.DataFrame: 添加了因子列的 DataFrame
        """
        logger.info(f"Computing {len(self.factor_engine.factors)} factors...")
        
        # 使用 FactorEngine 计算因子
        df_with_factors = self.factor_engine.compute_factors(df)
        
        # 计算标签 (如果配置了)
        if self.factor_engine.label_config:
            df_with_factors = self.factor_engine.compute_label(df_with_factors)
            logger.info(f"Computed label: {self.factor_engine.label_config['name']}")
        
        logger.info(f"Computed factors on {len(df_with_factors)} rows")
        return df_with_factors
    
    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        清洗数据，处理空值。
        
        空值处理策略:
        - forward: 前向填充 (使用前一天的值), 然后反向填充处理前导 null
        - backward: 后向填充
        - zero: 填充为 0
        - mean: 填充为列均值
        
        Args:
            df (pl.DataFrame): 包含因子的数据
        
        Returns:
            pl.DataFrame: 清洗后的数据
        
        注意:
            - 标签列 (如 future_return_5) 使用 shift(-n) 会产生尾部 null
            - 这些 null 无法通过 forward fill 填充，需要单独处理
            - 因子列的前导 null (窗口期) 使用 forward + backward 组合填充
        """
        logger.info(f"Cleaning data with strategy: {self.fill_null_strategy}")
        
        # 获取因子列 (排除标识列、标签列和原始数据列)
        # 标签列使用 shift(-n) 会产生尾部 null，无法通过 forward/backward fill 填充
        exclude_columns = {
            "symbol", "trade_date", "Date",
            # 排除原始数据列 (不需要填充)
            "open", "high", "low", "close", "volume", "amount",
            "adj_factor", "turnover_rate", "pct_change"
        }
        label_columns = set()
        if self.factor_engine.label_config:
            label_columns.add(self.factor_engine.label_config["name"])
        
        # 特征列 (需要填充的因子列)
        feature_columns = [
            col for col in df.columns 
            if col not in exclude_columns and col not in label_columns
        ]
        
        logger.debug(f"Feature columns to clean: {feature_columns}")
        
        # 按 symbol 分组进行填充
        if self.fill_null_strategy == "forward":
            # 前向填充 + 后向填充组合，处理前导和尾随 null
            for col in feature_columns:
                df = df.with_columns(
                    pl.col(col).fill_null(strategy="forward").over("symbol")
                )
                df = df.with_columns(
                    pl.col(col).fill_null(strategy="backward").over("symbol")
                )
        elif self.fill_null_strategy == "backward":
            for col in feature_columns:
                df = df.with_columns(
                    pl.col(col).fill_null(strategy="backward").over("symbol")
                )
                df = df.with_columns(
                    pl.col(col).fill_null(strategy="forward").over("symbol")
                )
        elif self.fill_null_strategy == "zero":
            df = df.with_columns(
                [pl.col(col).fill_null(0) for col in feature_columns]
            )
        elif self.fill_null_strategy == "mean":
            df = df.with_columns(
                [pl.col(col).fill_null(pl.col(col).mean().over("symbol")) 
                 for col in feature_columns]
            )
        
        # 删除特征列仍有空值的行 (窗口期外的数据)
        # 标签列的 null 保留，后续由训练逻辑处理
        # 先统计每列的空值数量
        null_counts = {col: df[col].null_count() for col in feature_columns}
        logger.debug(f"Null counts before drop: {null_counts}")
        
        # 只删除所有特征列都为空的行，而不是任何特征列为空的行
        # 这样可以保留更多数据
        df = df.filter(
            ~pl.all_horizontal([pl.col(col).is_null() for col in feature_columns])
        )
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def save_to_parquet(
        self,
        df: pl.DataFrame,
        output_path: str,
        compression: str = "zstd",
    ) -> None:
        """
        将数据保存为 Parquet 文件。
        
        Args:
            df (pl.DataFrame): 要保存的 DataFrame
            output_path (str): 输出文件路径
            compression (str): 压缩算法，默认 "zstd"
                支持："zstd", "snappy", "gzip", "lz4", "none"
        
        注意:
            - 自动创建输出目录
            - 使用 zstd 压缩平衡文件大小和读取速度
        """
        # 确保目录存在
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to Parquet: {output_path} (compression: {compression})")
        
        # 保存为 Parquet
        df.write_parquet(
            output_file,
            compression=compression,
        )
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {len(df)} rows to {output_path} ({file_size_mb:.2f} MB)")
    
    def run(
        self,
        output_path: str = "data/parquet/features_latest.parquet",
        symbols: Optional[list[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        compression: str = "zstd",
    ) -> pl.DataFrame:
        """
        运行完整的特征计算管道。
        
        执行流程:
        1. 从数据库读取数据
        2. 准备数据 (计算 pct_change 等)
        3. 计算因子
        4. 清洗数据 (处理空值)
        5. 保存为 Parquet
        
        Args:
            output_path (str): 输出 Parquet 文件路径
            symbols (list[str], optional): 股票代码列表
            start_date (str, optional): 开始日期
            end_date (str, optional): 结束日期
            compression (str): Parquet 压缩算法
        
        Returns:
            pl.DataFrame: 计算好的特征数据
        """
        logger.info("=" * 50)
        logger.info("Starting Feature Pipeline")
        logger.info("=" * 50)
        
        # Step 1: 读取数据
        df = self.read_data_from_db(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
        )
        
        if df.is_empty():
            logger.warning("No data read from database")
            return df
        
        # Step 2: 准备数据
        df = self.prepare_data(df)
        
        # Step 3: 计算因子
        df = self.compute_factors(df)
        
        # Step 4: 清洗数据
        df = self.clean_data(df)
        
        # Step 5: 保存到 Parquet
        self.save_to_parquet(df, output_path, compression)
        
        logger.info("=" * 50)
        logger.info("Feature Pipeline complete!")
        logger.info("=" * 50)
        
        return df


def run_pipeline(
    output_path: str = "data/parquet/features_latest.parquet",
    config_path: str = "config/factors.yaml",
) -> pl.DataFrame:
    """
    便捷函数：运行特征管道。
    
    Args:
        output_path (str): 输出 Parquet 文件路径
        config_path (str): 因子配置文件路径
    
    Returns:
        pl.DataFrame: 计算好的特征数据
    """
    pipeline = FeaturePipeline(config_path)
    return pipeline.run(output_path)


if __name__ == "__main__":
    # 运行特征管道
    df = run_pipeline()
    print(f"Generated {len(df)} rows of features")
    print(f"Columns: {df.columns}")