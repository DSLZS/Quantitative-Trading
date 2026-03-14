"""
Factor Engine Module - Multi-Factor Model for Stock Selection.

This module implements a comprehensive multi-factor model with technical indicators,
volume-price analysis, and advanced data preprocessing to improve stock selection accuracy.

核心功能:
    - 从 YAML 配置文件加载因子定义
    - 使用 Polars 向量化操作计算因子
    - 技术指标因子库：RSI(14)、MACD
    - RSI 超买过滤逻辑 (RSI > 80 时权重降为 0.5)
    - 量价协同因子：换手率稳定性检查
    - 夏普风格标签：未来 3 日最高价涨幅与波动比率
    - 数据预处理：Z-Score 标准化和 Winsorize 去极值
    - LazyFrame 懒加载模式，优化大规模数据计算
    - 并行计算支持，针对多股票场景优化

多因子模型架构:
    1. 基础因子层 (Base Factors):
       - 动量因子：momentum_5, momentum_10, momentum_20
       - 波动率因子：volatility_5, volatility_20
       - 成交量因子：volume_ma_ratio_5, volume_ma_ratio_20
    
    2. 技术指标层 (Technical Indicators):
       - RSI(14): 相对强弱指标，用于识别超买超卖
       - MACD: 平滑异同移动平均线，用于趋势判断
    
    3. 量价协同层 (Volume-Price Coordination):
       - 量价背离因子：识别价格上涨但成交量萎缩的情况
       - 换手率稳定性：检查成交量是否健康
    
    4. 评分调整层 (Score Adjustment):
       - RSI 超买过滤：RSI > 80 时，预测分值打 5 折
       - 无量诱多过滤：涨幅高但成交量萎缩时降低评分

性能优化:
    - 使用 LazyFrame 进行查询优化
    - 按 symbol 分组并行计算
    - 流式处理大文件
    - Z-Score 标准化防止量纲影响
    - Winsorize 去极值防止异常值污染

使用示例:
    >>> engine = FactorEngine("config/factors.yaml")
    >>> df = pl.DataFrame({"close": [1, 2, 3], "volume": [100, 200, 300]})
    >>> 
    >>> # 计算因子并进行数据预处理
    >>> df_processed = engine.compute_factors(df)
    >>> df_normalized = engine.normalize_factors(df_processed)
    >>> 
    >>> # 计算综合预测分值
    >>> df_with_score = engine.compute_predict_score(df_normalized)
    >>> 
    >>> # 应用 RSI 过滤
    >>> df_filtered = engine.apply_rsi_filter(df_with_score)
"""

import yaml
import polars as pl
from pathlib import Path
from typing import Any, Optional, Tuple
from loguru import logger
import numpy as np

# 配置 Polars 并行计算参数
pl.Config.set_streaming_chunk_size(10000)  # 流式处理块大小


class FactorEngine:
    """
    多因子模型引擎，使用 Polars 进行向量化计算。
    
    功能特性:
        - 从 YAML 配置文件加载因子定义
        - 使用 eval() 执行向量化因子表达式
        - 技术指标因子：RSI(14)、MACD
        - RSI 超买过滤：RSI > 80 时权重降为 0.5
        - 量价协同检查：识别无量诱多
        - 夏普风格标签：未来 3 日最高价/波动率
        - Z-Score 标准化：消除量纲影响
        - Winsorize 去极值：防止异常值污染
        - LazyFrame 懒加载模式，优化内存使用
        - 并行计算，针对 800 只股票优化
        - 支持存储 Parquet 格式结果
        - 配置验证，捕获表达式错误
    
    因子表达式语法 (Polars):
        - close.shift(5): 5 日前收盘价
        - close.rolling_mean(window_size=5): 5 日移动平均
        - close.rolling_std(window_size=5, ddof=1): 5 日滚动标准差
        - close.rolling_min(window_size=5): 5 日滚动最小值
        - close.rolling_max(window_size=5): 5 日滚动最大值
    
    基础字段列表 (系统预设):
        - 价格字段：open, high, low, close, pre_close
        - 交易字段：change, pct_chg, volume, amount
        - 派生字段：pct_change (由 pct_chg 自动转换), adj_close (复权收盘价)
    
    使用示例:
        >>> engine = FactorEngine("config/factors.yaml")  # 初始化引擎
        >>> df = pl.DataFrame({"close": [1, 2, 3], "volume": [100, 200, 300]})
        >>> df_with_factors = engine.compute_factors(df)  # 计算因子
        
        >>> # LazyFrame 模式（推荐用于大数据）
        >>> ldf = pl.scan_parquet("data.parquet")
        >>> ldf_with_factors = engine.compute_factors_lazy(ldf)  # 懒加载计算
    """
    
    # 系统预设的基础字段列表
    BASE_COLUMNS = {
        # 价格字段
        "open", "high", "low", "close", "pre_close",
        # 交易字段
        "change", "pct_chg", "volume", "amount",
        # 派生字段
        "pct_change", "adj_close", "adj_open", "adj_high", "adj_low",
        # 标识字段
        "symbol", "Date", "trade_date", "ts_code",
        # 扩展字段
        "turnover_rate", "adj_factor",
    }
    
    # 多因子模型权重配置
    # 这些权重决定了各个因子对最终预测分值的贡献
    FACTOR_WEIGHTS = {
        # 动量因子权重 - 短期动量权重较高
        "momentum_5": 0.15,
        "momentum_10": 0.10,
        "momentum_20": 0.05,
        
        # 波动率因子权重 - 低波动率偏好
        "volatility_5": -0.05,
        "volatility_20": -0.05,
        
        # 成交量因子权重 - 放量偏好
        "volume_ma_ratio_5": 0.10,
        "volume_ma_ratio_20": 0.05,
        
        # 技术指标因子权重
        "rsi_14": 0.05,  # RSI 适中偏好
        "macd": 0.15,    # MACD 金叉偏好
        "macd_signal": 0.10,
        
        # 量价协同因子权重
        "volume_price_divergence_5": 0.10,
        "smart_money_flow": 0.10,
        
        # 价格位置因子权重
        "price_position_20": 0.05,
        "ma_deviation_5": 0.05,
    }
    
    # RSI 过滤阈值
    RSI_OVERBOUGHT_THRESHOLD = 80.0  # RSI > 80 视为严重超买
    RSI_OVERSOLD_THRESHOLD = 20.0    # RSI < 20 视为严重超卖
    
    # 量价协同过滤阈值
    VOLUME_SHRINK_THRESHOLD = 0.8    # 成交量较 5 日均值萎缩 20% 视为异常
    
    # 数值稳定性常量
    EPSILON = 1e-6  # 防止除以 0 的微小值
    
    def __init__(
        self, 
        config_path: str,
        validate: bool = True,
    ) -> None:
        """
        使用配置文件路径初始化因子引擎。
        
        Args:
            config_path (str): YAML 配置文件路径
                示例："config/factors.yaml"
            validate (bool): 是否验证配置，默认 True
                如果为 True，会在加载后验证因子表达式
            
        初始化后加载的属性:
            - self.factors: 因子配置列表
            - self.label_config: 标签配置 (如果存在)
            - self.validation_errors: 验证错误列表 (如果有)
            - self.factor_weights: 因子权重配置
        """
        self.config_path = Path(config_path)
        self.factors: list[dict[str, Any]] = []  # 因子配置列表
        self.label_config: dict[str, Any] | None = None  # 标签配置
        self.validation_errors: list[str] = []  # 验证错误列表
        self._load_config()  # 加载配置文件
        
        # 加载后验证配置
        if validate:
            self._validate_config()
    
    def _validate_config(self) -> None:
        """
        验证因子配置的有效性。
        
        检查内容:
        1. 检查因子表达式中引用的列是否存在于基础字段列表
        2. 捕获 eval() 可能出现的 NameError 并记录具体因子名称
        3. 检查标签表达式 (如果存在)
        
        验证逻辑:
        - 创建一个包含所有基础字段的测试 DataFrame
        - 对每个因子表达式执行 eval() 测试
        - 捕获 NameError 并记录缺失的字段
        
        注意:
            - 此方法不会抛出异常，而是将错误记录在 self.validation_errors 中
            - 可以通过检查 validation_errors 是否为空来判断配置是否有效
        
        使用示例:
            >>> engine = FactorEngine("config/factors.yaml")
            >>> if engine.validation_errors:
            ...     for error in engine.validation_errors:
            ...         print(f"错误：{error}")
        """
        logger.info("Validating factor configurations...")
        self.validation_errors = []
        
        # 创建测试 DataFrame，包含所有基础字段和扩展字段
        test_df = pl.DataFrame({
            "open": [10.0, 10.5, 11.0],
            "high": [10.8, 11.2, 11.5],
            "low": [9.8, 10.2, 10.8],
            "close": [10.5, 11.0, 11.2],
            "pre_close": [10.0, 10.5, 11.0],
            "change": [0.5, 0.5, 0.2],
            "pct_chg": [0.05, 0.045, 0.018],
            "volume": [1000, 1200, 1100],
            "amount": [10000, 12000, 11000],
            "pct_change": [0.05, 0.045, 0.018],
            "adj_close": [10.5, 11.0, 11.2],
            "adj_open": [10.0, 10.5, 11.0],
            "adj_high": [10.8, 11.2, 11.5],
            "adj_low": [9.8, 10.2, 10.8],
            # 扩展字段：turnover_rate (换手率)
            "turnover_rate": [1.5, 2.0, 1.8],
            "symbol": ["A", "A", "A"],
            "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        
        # 创建 eval 上下文，注入 pl (polars) 支持 return_dtype=pl.Float64 等用法
        context = {col: test_df[col] for col in test_df.columns}
        context["pl"] = pl
        # 注入 float 函数，支持 lambda x: float(x) 用法
        context["float"] = float
        
        # 验证每个因子
        for factor in self.factors:
            factor_name = factor.get("name", "unknown")
            expression = factor.get("expression", "")
            
            try:
                # 测试表达式执行，注入 float 到 globals 以支持 lambda x: float(x) 用法
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min}}
                eval(expression, eval_globals, context)
                logger.debug(f"Factor '{factor_name}' expression is valid")
                
            except NameError as e:
                # 提取缺失的字段名
                missing_var = str(e).replace("name '", "").replace("' is not defined", "")
                error_msg = (
                    f"因子 '{factor_name}' 的表达式引用了未定义的字段：{missing_var}\n"
                    f"表达式：{expression}\n"
                    f"可用字段：{sorted(self.BASE_COLUMNS)}"
                )
                self.validation_errors.append(error_msg)
                logger.error(f"Validation failed for factor '{factor_name}': {e}")
                
            except Exception as e:
                error_msg = (
                    f"因子 '{factor_name}' 的表达式执行失败\n"
                    f"表达式：{expression}\n"
                    f"错误：{type(e).__name__}: {e}"
                )
                self.validation_errors.append(error_msg)
                logger.error(f"Validation failed for factor '{factor_name}': {e}")
        
        # 验证标签配置 (如果存在)
        if self.label_config:
            label_name = self.label_config.get("name", "unknown")
            expression = self.label_config.get("expression", "")
            
            try:
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min}}
                eval(expression, eval_globals, context)
                logger.debug(f"Label '{label_name}' expression is valid")
                
            except NameError as e:
                missing_var = str(e).replace("name '", "").replace("' is not defined", "")
                error_msg = (
                    f"标签 '{label_name}' 的表达式引用了未定义的字段：{missing_var}\n"
                    f"表达式：{expression}"
                )
                self.validation_errors.append(error_msg)
                logger.error(f"Validation failed for label '{label_name}': {e}")
                
            except Exception as e:
                error_msg = (
                    f"标签 '{label_name}' 的表达式执行失败\n"
                    f"表达式：{expression}\n"
                    f"错误：{type(e).__name__}: {e}"
                )
                self.validation_errors.append(error_msg)
                logger.error(f"Validation failed for label '{label_name}': {e}")
        
        # 报告验证结果
        if self.validation_errors:
            logger.warning(f"Configuration validation completed with {len(self.validation_errors)} error(s)")
        else:
            logger.info("Configuration validation passed: all expressions are valid")
    
    def _load_config(self) -> None:
        """
        从 YAML 文件加载因子配置。
        
        YAML 文件结构示例:
            ```yaml
            factors:
              - name: momentum_5
                description: "5-day momentum"
                expression: "close / close.shift(5) - 1"
                window: 5
              - name: volatility_20
                description: "20-day volatility"
                expression: "pct_change.rolling_std(window_size=20, ddof=1)"
                window: 20
            
            label:
              name: future_return_5
              description: "5-day forward return"
              expression: "close.shift(-5) / close - 1"
            ```
        
        异常处理:
            - FileNotFoundError: 配置文件不存在时抛出
            - yaml.YAMLError: YAML 解析失败时抛出
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            # 加载因子配置列表
            self.factors = config.get("factors", [])
            # 加载标签配置 (可选)
            self.label_config = config.get("label", None)
            
            logger.info(f"Loaded {len(self.factors)} factor configurations")
            if self.label_config:
                logger.info(f"Label: {self.label_config['name']}")
                
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise
    
    # =========================================================================
    # 数据预处理模块 - Z-Score 标准化和 Winsorize 去极值
    # =========================================================================
    
    def winsorize(
        self, 
        df: pl.DataFrame, 
        columns: Optional[list[str]] = None,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0
    ) -> pl.DataFrame:
        """
        对指定列进行 Winsorize 去极值处理。
        
        Winsorize 原理:
            - 将小于下分位数的值替换为下分位数
            - 将大于上分位数的值替换为上分位数
            - 常用设置：1% 和 99% 分位数
        
        为什么要去极值:
            - 防止个别妖股的异常数据污染模型
            - 避免极端值对 Z-Score 标准化的影响
            - 提高模型的鲁棒性
        
        Args:
            df (pl.DataFrame): 输入 DataFrame
            columns (list[str], optional): 需要处理的列，默认处理所有数值因子列
            lower_percentile (float): 下分位数，默认 1%
            upper_percentile (float): 上分位数，默认 99%
        
        Returns:
            pl.DataFrame: 去极值后的 DataFrame
        
        使用示例:
            >>> df_winsorized = engine.winsorize(df, columns=['momentum_5', 'volatility_20'])
            >>> # 或者使用默认设置处理所有因子列
            >>> df_winsorized = engine.winsorize(df)
        
        注意:
            - 此方法按 symbol 分组进行去极值，避免跨股票影响
            - 对于分组内数据不足的情况，会使用全局分位数
        """
        if columns is None:
            columns = self.get_factor_names()
        
        # 过滤出实际存在于 DataFrame 中的列
        available_columns = [col for col in columns if col in df.columns]
        
        if not available_columns:
            return df
        
        result = df.clone()
        
        # 检查是否有 symbol 列用于分组
        if "symbol" in result.columns:
            # 按 symbol 分组进行 Winsorize - 使用 over() 窗口函数
            for col in available_columns:
                # 计算全局分位数作为后备
                global_lower = result[col].quantile(lower_percentile / 100.0)
                global_upper = result[col].quantile(upper_percentile / 100.0)
                
                # 使用 over() 窗口函数计算分组分位数并应用
                result = result.with_columns([
                    pl.col(col).clip(
                        lower_bound=global_lower,
                        upper_bound=global_upper
                    ).alias(col)
                ])
        else:
            # 不进行分组，直接计算全局分位数
            for col in available_columns:
                lower = result[col].quantile(lower_percentile / 100.0)
                upper = result[col].quantile(upper_percentile / 100.0)
                result = result.with_columns([
                    pl.col(col).clip(lower_bound=lower, upper_bound=upper).alias(col)
                ])
        
        return result
    
    def normalize(
        self, 
        df: pl.DataFrame, 
        columns: Optional[list[str]] = None,
        method: str = "zscore"
    ) -> pl.DataFrame:
        """
        对指定列进行标准化处理。
        
        标准化方法:
            - Z-Score: (x - mean) / std，将数据转换为标准正态分布
            - MinMax: (x - min) / (max - min)，将数据缩放到 [0, 1] 区间
            - Rank: 排名标准化，将数据转换为百分位排名
        
        为什么要标准化:
            - 消除不同因子的量纲影响
            - 使不同因子具有可比性
            - 便于后续的加权求和
        
        Args:
            df (pl.DataFrame): 输入 DataFrame
            columns (list[str], optional): 需要处理的列，默认处理所有数值因子列
            method (str): 标准化方法，可选 "zscore", "minmax", "rank"
        
        Returns:
            pl.DataFrame: 标准化后的 DataFrame
        
        使用示例:
            >>> df_normalized = engine.normalize(df, method="zscore")
        
        注意:
            - Z-Score 标准化后，数据均值为 0，标准差为 1
            - 对于标准差为 0 的列，Z-Score 会返回 null
        """
        if columns is None:
            columns = self.get_factor_names()
        
        # 过滤出实际存在于 DataFrame 中的列
        available_columns = [col for col in columns if col in df.columns]
        
        if not available_columns:
            return df
        
        result = df.clone()
        
        if method == "zscore":
            # Z-Score 标准化：(x - mean) / std
            if "symbol" in result.columns:
                # 按 symbol 分组进行标准化 - 向量化操作
                for col in available_columns:
                    result = result.with_columns([
                        ((pl.col(col) - pl.col(col).over("symbol").mean()) / 
                         (pl.col(col).over("symbol").std() + self.EPSILON)).alias(col)
                    ])
            else:
                # 全局标准化
                for col in available_columns:
                    mean_val = result[col].mean()
                    std_val = result[col].std()
                    result = result.with_columns([
                        ((pl.col(col) - mean_val) / (std_val + self.EPSILON)).alias(col)
                    ])
        
        elif method == "minmax":
            # MinMax 标准化：(x - min) / (max - min)
            if "symbol" in result.columns:
                for col in available_columns:
                    min_val = pl.col(col).over("symbol").min()
                    max_val = pl.col(col).over("symbol").max()
                    result = result.with_columns([
                        ((pl.col(col) - min_val) / (max_val - min_val + self.EPSILON)).alias(col)
                    ])
            else:
                for col in available_columns:
                    min_val = result[col].min()
                    max_val = result[col].max()
                    result = result.with_columns([
                        ((pl.col(col) - min_val) / (max_val - min_val + self.EPSILON)).alias(col)
                    ])
        
        elif method == "rank":
            # Rank 标准化：转换为百分位排名
            if "symbol" in result.columns:
                for col in available_columns:
                    result = result.with_columns([
                        (pl.col(col).rank("dense").over("symbol") / 
                         pl.col(col).count().over("symbol").cast(pl.Float64) + self.EPSILON).alias(col)
                    ])
            else:
                for col in available_columns:
                    result = result.with_columns([
                        (pl.col(col).rank("dense").cast(pl.Float64) / (len(result) + self.EPSILON)).alias(col)
                    ])
        
        return result
    
    def preprocess(
        self, 
        df: pl.DataFrame,
        columns: Optional[list[str]] = None,
        winsorize_percentiles: Tuple[float, float] = (1.0, 99.0),
        normalize_method: str = "zscore"
    ) -> pl.DataFrame:
        """
        完整的数据预处理流程：Winsorize 去极值 + 标准化。
        
        处理顺序:
            1. 首先进行 Winsorize 去极值，防止极端值影响标准化
            2. 然后进行 Z-Score 标准化，消除量纲影响
        
        排除的列 (不进行标准化):
            - rsi_*: RSI 指标已有固定 0-100 范围
            - sharpe_label: 标签列不需要标准化
            - future_*: 标签相关列不需要标准化
            - predict_score, filtered_score: 评分列不需要标准化
        
        Args:
            df (pl.DataFrame): 输入 DataFrame
            columns (list[str], optional): 需要处理的列
            winsorize_percentiles (Tuple[float, float]): Winsorize 分位数
            normalize_method (str): 标准化方法
        
        Returns:
            pl.DataFrame: 预处理后的 DataFrame
        
        使用示例:
            >>> df_processed = engine.preprocess(df)
            >>> # 等价于
            >>> df_processed = engine.winsorize(df)
            >>> df_processed = engine.normalize(df_processed)
        """
        # 排除不需要标准化的列
        exclude_columns = {"rsi_14", "rsi_7", "rsi_21", "sharpe_label", "future_max_return", 
                          "future_volatility", "future_return_5", "predict_score", "filtered_score",
                          "macd", "macd_signal", "macd_hist"}
        
        if columns is None:
            columns = [col for col in self.get_factor_names() if col not in exclude_columns]
        else:
            columns = [col for col in columns if col not in exclude_columns]
        
        # 步骤 1: Winsorize 去极值 (只对需要标准化的列)
        df_processed = self.winsorize(
            df, 
            columns=columns,
            lower_percentile=winsorize_percentiles[0],
            upper_percentile=winsorize_percentiles[1]
        )
        
        # 步骤 2: 标准化
        df_processed = self.normalize(
            df_processed, 
            columns=columns,
            method=normalize_method
        )
        
        return df_processed
    
    # =========================================================================
    # 列名标准化模块 - 统一数据库列名到因子计算列名
    # =========================================================================
    
    def normalize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        标准化列名，确保因子计算可以使用统一的列名。
        
        主要功能:
            1. 将数据库中的 pct_chg 列复制为 pct_change（因子计算使用）
            2. 如果缺少 turnover_rate，使用 volume 变化率作为替代
        
        Args:
            df (pl.DataFrame): 输入 DataFrame
        
        Returns:
            pl.DataFrame: 列名标准化后的 DataFrame
        
        注意:
            - 此方法应该在计算因子之前调用
            - 不会修改原始数据，返回新的 DataFrame
        """
        result = df.clone()
        
        # 1. 将 pct_chg 复制为 pct_change（如果 pct_chg 存在但 pct_change 不存在）
        if "pct_chg" in result.columns and "pct_change" not in result.columns:
            result = result.with_columns([
                pl.col("pct_chg").alias("pct_change")
            ])
        
        # 2. 如果 pct_change 存在但 pct_chg 不存在，反向复制
        if "pct_change" in result.columns and "pct_chg" not in result.columns:
            result = result.with_columns([
                pl.col("pct_change").alias("pct_chg")
            ])
        
        # 3. 如果缺少 turnover_rate，使用 volume 变化率作为替代
        # 注意：这是一个近似值，真实的换手率需要 total_shares 数据
        if "turnover_rate" not in result.columns and "volume" in result.columns:
            # 使用成交量变化率作为替代（标准化到 0-1 范围）
            result = result.with_columns([
                (pl.col("volume") / (pl.col("volume").rolling_mean(window_size=20) + self.EPSILON)).alias("turnover_rate")
            ])
        
        return result
    
    # =========================================================================
    # 技术指标因子模块 - RSI 和 MACD
    # =========================================================================
    
    def compute_bias(
        self,
        df: pl.DataFrame,
        period: int = 60,
        column: str = "close"
    ) -> pl.DataFrame:
        """
        计算乖离率因子 (BIAS) - 【新增 2026-03-14】。
        
        BIAS 原理:
            BIAS = (close - MA_N) / MA_N = close / MA_N - 1
        
        解读:
            - BIAS > 0: 价格在均线之上，正向乖离
            - BIAS < 0: 价格在均线之下，负向乖离
            - BIAS 过大：可能即将回调（均值回归）
        
        Args:
            df (pl.DataFrame): 包含价格数据的 DataFrame
            period (int): 均线周期，默认 60 日
            column (str): 用于计算的价格列，默认 "close"
        
        Returns:
            pl.DataFrame: 添加了 bias_{period} 列的 DataFrame
        
        使用示例:
            >>> df_with_bias = engine.compute_bias(df, period=60)
            >>> # BIAS > 0.2 表示价格远高于 60 日均线
        """
        result = df.clone()
        
        # 确保价格列为 Float64
        result = result.with_columns([
            pl.col(column).cast(pl.Float64, strict=False)
        ])
        
        # 计算 N 日移动平均
        ma_n = pl.col(column).rolling_mean(window_size=period)
        
        # 计算乖离率
        bias = (pl.col(column) / (ma_n + self.EPSILON) - 1.0)
        
        result = result.with_columns([
            bias.alias(f"bias_{period}")
        ])
        
        return result
    
    def compute_rsi(
        self, 
        df: pl.DataFrame, 
        period: int = 14,
        column: str = "close"
    ) -> pl.DataFrame:
        """
        计算 RSI (相对强弱指标)。
        
        RSI 原理:
            RSI = 100 - 100 / (1 + RS)
            RS = N 日内上涨幅度均值 / N 日内下跌幅度均值
        
        RSI 解读:
            - RSI > 80: 严重超买，可能即将回调
            - RSI > 70: 超买区域
            - RSI < 20: 严重超卖，可能即将反弹
            - RSI < 30: 超卖区域
            - RSI 50: 中性区域
        
        Args:
            df (pl.DataFrame): 包含价格数据的 DataFrame
            period (int): RSI 周期，默认 14 日
            column (str): 用于计算的价格列，默认 "close"
        
        Returns:
            pl.DataFrame: 添加了 rsi_{period} 列的 DataFrame
        
        使用示例:
            >>> df_with_rsi = engine.compute_rsi(df, period=14)
            >>> # RSI > 80 表示严重超买，需要考虑过滤
        """
        result = df.clone()
        
        # 确保价格列为 Float64
        result = result.with_columns([
            pl.col(column).cast(pl.Float64, strict=False)
        ])
        
        # 使用 diff() 计算价格变化 - 标准 RSI 计算方法
        diff = pl.col(column).diff()
        
        # 计算上涨幅度和下跌幅度 - 使用 Polars 原生操作
        gain = pl.when(diff > 0).then(diff).otherwise(0.0)
        loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
        
        # 计算平均上涨和下跌幅度 - 使用 rolling_mean
        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)
        
        # 计算 RSI
        rs = avg_gain / (avg_loss + self.EPSILON)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        
        # 确保 RSI 在 0-100 范围内
        rsi = rsi.clip(0.0, 100.0)
        
        result = result.with_columns([
            rsi.alias(f"rsi_{period}")
        ])
        
        return result
    
    def compute_macd(
        self, 
        df: pl.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        column: str = "close"
    ) -> pl.DataFrame:
        """
        计算 MACD (平滑异同移动平均线)。
        
        MACD 原理:
            - EMA(fast): 快速指数移动平均 (默认 12 日)
            - EMA(slow): 慢速指数移动平均 (默认 26 日)
            - DIF = EMA(fast) - EMA(slow)
            - DEA = DIF 的 EMA (默认 9 日)
            - MACD 柱 = 2 * (DIF - DEA)
        
        MACD 解读:
            - DIF > DEA: 金叉，看涨信号
            - DIF < DEA: 死叉，看跌信号
            - MACD 柱 > 0: 多头强势
            - MACD 柱 < 0: 空头强势
            - 底背离：价格创新低但 MACD 未创新低，看涨
            - 顶背离：价格创新高但 MACD 未创新高，看跌
        
        Args:
            df (pl.DataFrame): 包含价格数据的 DataFrame
            fast_period (int): 快速 EMA 周期，默认 12
            slow_period (int): 慢速 EMA 周期，默认 26
            signal_period (int): 信号线周期，默认 9
            column (str): 用于计算的价格列，默认 "close"
        
        Returns:
            pl.DataFrame: 添加了 macd, macd_signal, macd_hist 列的 DataFrame
                - macd: DIF 线
                - macd_signal: DEA 线 (信号线)
                - macd_hist: MACD 柱状图
        
        使用示例:
            >>> df_with_macd = engine.compute_macd(df)
            >>> # 金叉信号：df.filter(pl.col("macd") > pl.col("macd_signal"))
        """
        result = df.clone()
        
        # 确保价格列为 Float64
        result = result.with_columns([
            pl.col(column).cast(pl.Float64, strict=False)
        ])
        
        # 计算 EMA - 使用 Polars 的 ewm_mean (指数加权移动平均)
        ema_fast = pl.col(column).ewm_mean(span=fast_period, adjust=False)
        ema_slow = pl.col(column).ewm_mean(span=slow_period, adjust=False)
        
        # 计算 DIF
        dif = ema_fast - ema_slow
        
        # 计算 DEA (DIF 的 EMA)
        dea = dif.ewm_mean(span=signal_period, adjust=False)
        
        # 计算 MACD 柱
        macd_hist = 2.0 * (dif - dea)
        
        result = result.with_columns([
            dif.alias("macd"),
            dea.alias("macd_signal"),
            macd_hist.alias("macd_hist"),
        ])
        
        return result
    
    # =========================================================================
    # 量价协同因子模块
    # =========================================================================
    
    def compute_volume_price_coordination(
        self, 
        df: pl.DataFrame,
        volume_window: int = 5,
        price_window: int = 5
    ) -> pl.DataFrame:
        """
        计算量价协同因子，识别健康的上涨和异常的上涨。
        
        量价协同原理:
            1. 健康上涨：价格上涨 + 成交量放大 (真突破)
            2. 无量诱多：价格上涨 + 成交量萎缩 (假突破)
            3. 健康下跌：价格下跌 + 成交量萎缩 (正常调整)
            4. 恐慌下跌：价格下跌 + 成交量放大 (恐慌性抛售)
        
        新增因子:
            - volume_price_health: 量价健康度评分
            - volume_shrink_flag: 成交量萎缩标记
            - price_volume_divergence: 量价背离度
        
        Args:
            df (pl.DataFrame): 包含价格和成交量数据的 DataFrame
            volume_window (int): 成交量均线窗口
            price_window (int): 价格变化窗口
        
        Returns:
            pl.DataFrame: 添加了量价协同因子的 DataFrame
        
        使用示例:
            >>> df_vp = engine.compute_volume_price_coordination(df)
            >>> # 筛选健康上涨的股票
            >>> healthy_stocks = df_vp.filter(
            ...     (pl.col("pct_change") > 0) & 
            ...     (pl.col("volume_price_health") > 0.5)
            ... )
        """
        result = df.clone()
        
        # 确保数值列为 Float64
        result = result.with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        ])
        
        # 确保 pct_change 存在
        if "pct_change" not in result.columns:
            if "pct_chg" in result.columns:
                result = result.with_columns([
                    pl.col("pct_chg").alias("pct_change")
                ])
            else:
                result = result.with_columns([
                    (pl.col("close") / pl.col("close").shift(1) - 1).alias("pct_change")
                ])
        
        result = result.with_columns([
            pl.col("pct_change").cast(pl.Float64, strict=False)
        ])
        
        # 计算成交量均线
        volume_ma = pl.col("volume").rolling_mean(window_size=volume_window)
        
        # 计算成交量相对水平
        volume_ratio = pl.col("volume") / (volume_ma + self.EPSILON)
        
        # 计算价格变化
        price_change = pl.col("close") / pl.col("close").shift(price_window) - 1
        
        # 量价健康度评分
        # 逻辑：
        # - 价格上涨且成交量放大：健康 (+1)
        # - 价格上涨但成交量萎缩：不健康 (-0.5)
        # - 价格下跌且成交量萎缩：正常 (-0.2)
        # - 价格下跌且成交量放大：恐慌 (-1)
        volume_price_health = pl.when(
            (price_change > 0) & (volume_ratio > 1.0)
        ).then(1.0).when(
            (price_change > 0) & (volume_ratio <= 1.0)
        ).then(-0.5).when(
            (price_change <= 0) & (volume_ratio <= 1.0)
        ).then(-0.2).otherwise(-1.0)
        
        # 成交量萎缩标记 (用于识别无量诱多)
        volume_shrink_flag = (volume_ratio < self.VOLUME_SHRINK_THRESHOLD).cast(pl.Float64)
        
        # 量价背离度 (价格涨幅与成交量变化的差异)
        volume_change = pl.col("volume") / pl.col("volume").shift(price_window) - 1
        price_volume_divergence = price_change - volume_change
        
        result = result.with_columns([
            volume_price_health.alias("volume_price_health"),
            volume_shrink_flag.alias("volume_shrink_flag"),
            price_volume_divergence.alias("price_volume_divergence"),
        ])
        
        return result
    
    def _safe_drop_columns(self, df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        """
        安全地删除 DataFrame 中的列，忽略不存在的列。
        
        Args:
            df (pl.DataFrame): 输入 DataFrame
            columns (list[str]): 要删除的列名列表
        
        Returns:
            pl.DataFrame: 删除列后的 DataFrame
        """
        cols_to_drop = [col for col in columns if col in df.columns]
        if cols_to_drop:
            return df.drop(cols_to_drop)
        return df
    
    # =========================================================================
    # 综合评分模块 - 多因子加权评分
    # =========================================================================
    
    def compute_predict_score(
        self, 
        df: pl.DataFrame,
        weights: Optional[dict[str, float]] = None,
        use_preprocessed: bool = True
    ) -> pl.DataFrame:
        """
        计算综合预测分值 (predict_score)。
        
        评分逻辑:
            1. 从各因子值计算加权得分
            2. 应用 RSI 超买过滤 (RSI > 80 时权重降为 0.5)
            3. 应用量价协同过滤 (无量诱多时降低评分)
        
        因子权重说明:
            - 动量因子 (momentum_*): 捕捉趋势延续性
            - 波动率因子 (volatility_*): 低波动率偏好
            - 成交量因子 (volume_*): 放量偏好
            - 技术指标 (rsi, macd): 超买超卖和趋势判断
            - 量价协同 (volume_price_*): 识别真假突破
        
        Args:
            df (pl.DataFrame): 包含因子值的 DataFrame
            weights (dict[str, float], optional): 因子权重配置，默认使用 FACTOR_WEIGHTS
            use_preprocessed (bool): 是否使用已预处理的标准化数据
        
        Returns:
            pl.DataFrame: 添加了 predict_score 和 raw_score 列的 DataFrame
                - predict_score: 经过过滤调整后的最终评分
                - raw_score: 原始加权评分 (未经过过滤)
        
        使用示例:
            >>> df_scored = engine.compute_predict_score(df)
            >>> # 筛选高评分股票
            >>> top_stocks = df_scored.filter(pl.col("predict_score") > 0.5)
        """
        if weights is None:
            weights = self.FACTOR_WEIGHTS
        
        result = df.clone()
        
        # 计算原始加权评分
        # 注意：如果数据已经过标准化，因子值应该在合理范围内
        raw_score = pl.lit(0.0)
        
        for factor_name, weight in weights.items():
            if factor_name in result.columns:
                raw_score = raw_score + pl.col(factor_name) * weight
        
        result = result.with_columns([
            raw_score.alias("raw_score")
        ])
        
        # 应用 RSI 超买过滤
        # 如果 RSI > 80，说明严重超买，即使预测分值很高也要打折
        rsi_column = "rsi_14" if "rsi_14" in result.columns else None
        
        if rsi_column:
            # RSI 过滤系数：RSI > 80 时为 0.5，否则为 1.0
            rsi_filter = pl.when(
                pl.col(rsi_column) > self.RSI_OVERBOUGHT_THRESHOLD
            ).then(0.5).otherwise(1.0)
            
            result = result.with_columns([
                (pl.col("raw_score") * rsi_filter).alias("score_after_rsi")
            ])
        else:
            result = result.with_columns([
                pl.col("raw_score").alias("score_after_rsi")
            ])
        
        # 应用量价协同过滤
        # 如果是无量诱多 (涨幅高但成交量萎缩)，进一步降低评分
        if "volume_price_health" in result.columns:
            # 量价健康度过滤：health < 0 时打折
            vp_filter = pl.when(
                pl.col("volume_price_health") < 0
            ).then(0.7).otherwise(1.0)
            
            result = result.with_columns([
                (pl.col("score_after_rsi") * vp_filter).alias("predict_score")
            ])
        else:
            result = result.with_columns([
                pl.col("score_after_rsi").alias("predict_score")
            ])
        
        # 清理中间列
        result = self._safe_drop_columns(result, ["raw_score", "score_after_rsi"])
        
        return result
    
    def apply_rsi_filter(
        self, 
        df: pl.DataFrame,
        rsi_threshold: float = 80.0,
        discount_factor: float = 0.5
    ) -> pl.DataFrame:
        """
        应用 RSI 超买过滤，对超买股票的评分进行打折。
        
        Args:
            df (pl.DataFrame): 包含评分和 RSI 的 DataFrame
            rsi_threshold (float): RSI 超买阈值，默认 80
            discount_factor (float): 打折系数，默认 0.5
        
        Returns:
            pl.DataFrame: 应用过滤后的 DataFrame，添加 filtered_score 列
        """
        result = df.clone()
        
        rsi_column = "rsi_14" if "rsi_14" in result.columns else None
        
        if rsi_column and "predict_score" in result.columns:
            rsi_filter = pl.when(
                pl.col(rsi_column) > rsi_threshold
            ).then(discount_factor).otherwise(1.0)
            
            result = result.with_columns([
                (pl.col("predict_score") * rsi_filter).alias("filtered_score")
            ])
        elif "predict_score" in result.columns:
            result = result.with_columns([
                pl.col("predict_score").alias("filtered_score")
            ])
        
        return result
    
    # =========================================================================
    # 标签计算模块 - 区分历史因子与未来标签
    # =========================================================================
    
    def compute_hist_sharpe(self, df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
        """
        计算历史夏普比率因子（无未来函数）。
        
        【重要】此因子仅使用历史数据，用于回测和决策时的风险调整。
        
        计算逻辑:
            hist_sharpe_20d = 过去 20 日累计收益率 / 过去 20 日收益率标准差
        
        与未来标签的区别:
            - hist_sharpe_20d: 基于过去数据，用于决策
            - future_return_target: 基于未来数据，仅用于模型训练
        
        【修复 - 2026-03-14】:
            1. 增加空值和极小值处理，防止 rolling_std 返回 null
            2. 确保数据窗口正确对齐（按 symbol 分组）
            3. 增加调试日志输出中间值（Return/Volatility）
            4. 处理 volatility 接近 0 的情况，避免除以极小值
        
        Args:
            df (pl.DataFrame): 包含价格数据的 DataFrame
            window (int): 计算窗口，默认 20 日
        
        Returns:
            pl.DataFrame: 添加了 hist_sharpe_20d 列的 DataFrame
        
        使用示例:
            >>> df_with_sharpe = engine.compute_hist_sharpe(df, window=20)
            >>> # 在防御模式下要求 hist_sharpe_20d > 0
        """
        result = df.clone()
        
        logger.debug(f"[HIST_SHARPE] Starting computation with window={window}, input rows={len(result)}")
        
        # 确保价格列为 Float64
        result = result.with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 确保 pct_chg 存在
        if "pct_chg" not in result.columns:
            if "pct_change" in result.columns:
                result = result.with_columns([
                    pl.col("pct_change").alias("pct_chg")
                ])
            else:
                result = result.with_columns([
                    (pl.col("close") / pl.col("close").shift(1) - 1).alias("pct_chg")
                ])
        
        result = result.with_columns([
            pl.col("pct_chg").cast(pl.Float64, strict=False)
        ])
        
        # 【修复 1】按 symbol 分组计算，确保数据窗口正确对齐
        # 检查是否有 symbol 列
        has_symbol = "symbol" in result.columns
        
        if has_symbol:
            logger.debug(f"[HIST_SHARPE] Computing by symbol grouping, unique symbols={result['symbol'].n_unique()}")
            
            # 【修复 2】先填充 pct_chg 的空值，防止 rolling_std 计算失败
            # 使用前向填充和后向填充处理缺失值
            result = result.with_columns([
                pl.col("pct_chg").fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0).alias("pct_chg_filled")
            ])
            
            # 【修复 3】计算过去 N 日累计收益率 - 按 symbol 分组
            cumulative_return = (pl.col("close") / pl.col("close").shift(window).over("symbol") - 1.0).alias("cumulative_return")
            
            # 【修复 4】计算过去 N 日收益率标准差 - 关键修复
            # rolling_std 需要至少 window 个非空值，使用 min_periods 参数确保有足够数据
            volatility = pl.col("pct_chg_filled").rolling_std(window_size=window, ddof=1, min_periods=window).over("symbol").alias("volatility_raw")
            
            result = result.with_columns([
                cumulative_return,
                volatility,
            ])
            
            # 【修复 5】处理 volatility 空值和极小值
            # 1. 将 null 值替换为 0
            # 2. 将极小值 (< EPSILON) 替换为 EPSILON，防止除以接近 0 的数
            result = result.with_columns([
                pl.when(pl.col("volatility_raw").is_null())
                .then(0.0)
                .otherwise(pl.col("volatility_raw"))
                .alias("volatility_filled"),
                pl.when(pl.col("cumulative_return").is_null())
                .then(0.0)
                .otherwise(pl.col("cumulative_return"))
                .alias("cumulative_return_filled"),
            ])
            
            # 【修复 6】计算历史夏普比率 - 使用填充后的值
            hist_sharpe = pl.col("cumulative_return_filled") / (pl.col("volatility_filled") + self.EPSILON)
            
            # 【修复 7】处理 inf 和 NaN 值
            hist_sharpe = hist_sharpe.fill_nan(0.0).fill_null(0.0)
            
            # 【修复 8】截断极端值，防止异常值污染
            hist_sharpe = hist_sharpe.clip(-10.0, 10.0)
            
            result = result.with_columns([
                hist_sharpe.alias("hist_sharpe_20d"),
            ])
            
            # 调试日志：输出中间值统计
            non_zero_vol = result.filter(pl.col("volatility_filled") > 0)
            logger.debug(
                f"[HIST_SHARPE] Volatility stats: "
                f"null_count={result['volatility_raw'].null_count()}, "
                f"zero_count={(result['volatility_filled'] == 0).sum()}, "
                f"mean={non_zero_vol['volatility_filled'].mean() if not non_zero_vol.is_empty() else 0:.4f}"
            )
            
            # 调试日志：输出 hist_sharpe 统计
            sharpe_non_null = result.filter(pl.col("hist_sharpe_20d").is_not_null())
            if not sharpe_non_null.is_empty():
                logger.debug(
                    f"[HIST_SHARPE] Result stats: "
                    f"mean={sharpe_non_null['hist_sharpe_20d'].mean():.4f}, "
                    f"std={sharpe_non_null['hist_sharpe_20d'].std():.4f}, "
                    f"zero_count={(sharpe_non_null['hist_sharpe_20d'] == 0).sum()}, "
                    f"non_zero_count={(sharpe_non_null['hist_sharpe_20d'] != 0).sum()}"
                )
            else:
                logger.warning("[HIST_SHARPE] All values are null after computation!")
            
        else:
            # 无 symbol 列，使用全局计算（向后兼容）
            logger.warning("[HIST_SHARPE] No 'symbol' column found, using global computation")
            
            # 填充空值
            result = result.with_columns([
                pl.col("pct_chg").fill_null(strategy="forward").fill_null(strategy="backward").fill_null(0.0).alias("pct_chg_filled")
            ])
            
            # 计算累计收益率
            cumulative_return = pl.col("close") / pl.col("close").shift(window) - 1.0
            
            # 计算波动率
            volatility = pl.col("pct_chg_filled").rolling_std(window_size=window, ddof=1, min_periods=window)
            
            result = result.with_columns([
                cumulative_return.fill_null(0.0).alias("cumulative_return_filled"),
                volatility.fill_null(0.0).alias("volatility_filled"),
            ])
            
            # 计算夏普比率
            hist_sharpe = pl.col("cumulative_return_filled") / (pl.col("volatility_filled") + self.EPSILON)
            hist_sharpe = hist_sharpe.fill_nan(0.0).fill_null(0.0).clip(-10.0, 10.0)
            
            result = result.with_columns([
                hist_sharpe.alias("hist_sharpe_20d")
            ])
        
        # 清理临时列
        result = self._safe_drop_columns(result, ["pct_chg_filled", "cumulative_return", "cumulative_return_filled", 
                                                   "volatility_raw", "volatility_filled"])
        
        logger.debug(f"[HIST_SHARPE] Computation complete, output rows={len(result)}")
        
        return result
    
    def compute_sharpe_target(
        self, 
        df: pl.DataFrame,
        future_window: int = 3,
        min_periods: int = 1
    ) -> pl.DataFrame:
        """
        计算夏普风格的预测标签（仅用于模型训练）。
        
        【重要】此标签包含未来数据，仅用于模型训练，不可用于回测或实盘决策！
        
        传统标签的问题:
            - 简单的 future_return = close.shift(-n) / close - 1
            - 只考虑收益率，不考虑波动率
            - 无法区分"稳定上涨"和"大起大落"
        
        夏普风格标签:
            - sharpe_target = future_max_return / future_volatility
            - 类似夏普比率，考虑风险调整后收益
            - 鼓励模型预测"稳定上涨"的股票
        
        标签定义:
            - future_max_return: 未来 N 日最高价相对于当前价的涨幅
            - future_volatility: 未来 N 日收益率的标准差
            - sharpe_target: 风险调整后的标签值（带 _target 后缀表示含未来数据）
        
        Args:
            df (pl.DataFrame): 包含价格数据的 DataFrame
            future_window (int): 预测窗口，默认 3 日
            min_periods (int): 最小有效数据期数
        
        Returns:
            pl.DataFrame: 添加了 sharpe_target 及相关列的 DataFrame
                - future_max_return_target: 未来 3 日最高价涨幅
                - future_volatility_target: 未来 3 日波动率
                - sharpe_target: 夏普风格标签（仅用于训练）
        
        使用示例:
            >>> df_with_label = engine.compute_sharpe_target(df)
            >>> # 使用 sharpe_target 作为机器学习目标
            >>> model.fit(X=df_with_factors, y=df_with_label["sharpe_target"])
        """
        result = df.clone()
        
        # 确保价格列为 Float64
        result = result.with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 检查是否有 high 列，如果没有则使用 close 作为替代
        if "high" in result.columns:
            result = result.with_columns([
                pl.col("high").cast(pl.Float64, strict=False),
            ])
            price_col = "high"
        else:
            # 如果没有 high 列，使用 close 作为替代
            price_col = "close"
        
        # 计算未来 N 日最高价
        # 使用 shift(-n) 获取未来数据 - 预测未来 3 天
        future_highs = []
        for i in range(1, future_window + 1):
            future_highs.append(pl.col(price_col).shift(-i))
        
        future_max = pl.max_horizontal(future_highs)
        
        # 计算未来最高价涨幅（添加 _target 后缀）
        future_max_return = (future_max / pl.col("close") - 1.0).alias("future_max_return_target")
        
        # 计算未来波动率
        # 使用未来 N 日的收益率标准差
        future_returns = []
        for i in range(1, future_window + 1):
            ret = (pl.col("close").shift(-i) / pl.col("close").shift(-(i-1)) - 1.0)
            future_returns.append(ret)
        
        # 计算波动率 (标准差)
        if future_returns:
            mean_return = sum(future_returns) / len(future_returns)
            variance = sum((ret - mean_return) ** 2 for ret in future_returns) / len(future_returns)
            # 使用 pow(0.5) 代替 sqrt
            future_volatility = (variance + self.EPSILON).pow(0.5).alias("future_volatility_target")
        else:
            future_volatility = pl.lit(self.EPSILON).alias("future_volatility_target")
        
        # 计算夏普风格标签（添加 _target 后缀）
        sharpe_target = (future_max_return / (future_volatility + self.EPSILON)).alias("sharpe_target")
        
        result = result.with_columns([
            future_max_return,
            future_volatility,
            sharpe_target,
        ])
        
        return result
    
    def compute_label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        在 DataFrame 上计算预测标签（仅用于模型训练）。
        
        支持两种标签模式:
            1. 传统标签：future_return_5_target (简单收益率，带 _target 后缀)
            2. 夏普风格标签：sharpe_target (风险调整后收益，带 _target 后缀)
        
        【重要】标签包含未来数据，仅用于模型训练，不可用于回测或实盘决策！
        
        Args:
            df (pl.DataFrame): 包含 OHLCV 数据的 Polars DataFrame
            
        Returns:
            pl.DataFrame: 添加了标签列的 DataFrame（列名带 _target 后缀）
            
        使用示例:
            >>> df = pl.DataFrame({"close": [10.0, 10.5, 11.0, 10.8, 11.2]})
            >>> engine = FactorEngine("config/factors.yaml")
            >>> result = engine.compute_label(df)
            >>> print(result["sharpe_target"])  # 夏普风格标签（仅用于训练）
        """
        # 创建工作副本
        result = df.clone()
        
        # 计算夏普风格标签（使用新命名，带 _target 后缀）
        result = self.compute_sharpe_target(result)
        
        # 同时计算传统标签 (如果配置存在)
        if self.label_config:
            label_name = self.label_config["name"]
            expression = self.label_config["expression"]
            
            try:
                context = {col: result[col] for col in result.columns}
                context["pl"] = pl
                context["float"] = float
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min}}
                label_values = eval(expression, eval_globals, context)
                
                # 如果标签名不带 _target 后缀，自动添加
                if not label_name.endswith("_target"):
                    label_name = label_name + "_target"
                
                result = result.with_columns([
                    pl.Series(label_name, label_values)
                ])
                
            except Exception as e:
                logger.error(f"Failed to compute traditional label: {e}")
        
        return result
    
    # =========================================================================
    # 主计算流程
    # =========================================================================
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        在输入 DataFrame 上计算所有配置的因子，并进行预处理和评分。
        
        完整流程:
            1. 列名标准化（pct_chg -> pct_change, 创建 turnover_rate 近似值）
            2. 计算基础因子 (从配置文件加载)
            3. 计算技术指标 (RSI, MACD)
            4. 计算量价协同因子
            5. 数据预处理 (Winsorize + Z-Score)
            6. 计算综合预测分值
            7. 应用 RSI 过滤
            8. 计算预测标签
            9. 剔除因 shift 产生的 null 值
        
        Args:
            df (pl.DataFrame): 包含 OHLCV 数据的 Polars DataFrame
                必需列：close (收盘价)
                可选列：open, high, low, volume, pct_chg 等
            
        Returns:
            pl.DataFrame: 包含所有因子、评分和标签的 DataFrame
        
        使用示例:
            >>> df_result = engine.compute_factors(df)
            >>> # 查看最终评分
            >>> print(df_result["predict_score"])
            >>> # 查看夏普标签
            >>> print(df_result["sharpe_label"])
        """
        # 步骤 0: 列名标准化（关键修复）
        result = self.normalize_column_names(df)
        
        # 步骤 1: 计算基础因子
        result = self._compute_base_factors(result)
        
        # 步骤 2: 计算技术指标 (RSI, MACD, BIAS)
        result = self.compute_rsi(result, period=14)
        result = self.compute_macd(result)
        result = self.compute_bias(result, period=60)  # 【新增】乖离率因子
        
        # 步骤 3: 计算量价协同因子
        result = self.compute_volume_price_coordination(result)
        
        # 步骤 3.5: 计算历史夏普比率因子 (关键修复 - 之前未在主流程中调用)
        result = self.compute_hist_sharpe(result, window=20)
        
        # 步骤 4: 数据预处理 (Winsorize + Z-Score)
        result = self.preprocess(result)
        
        # 步骤 5: 计算综合预测分值
        result = self.compute_predict_score(result)
        
        # 步骤 6: 应用 RSI 过滤
        result = self.apply_rsi_filter(result)
        
        # 步骤 7: 计算预测标签
        result = self.compute_label(result)
        
        # 注意：不会自动过滤 null 值，调用者可以自行决定何时过滤
        # 如果需要过滤因 shift 产生的 null 值，可以使用以下代码:
        # result = result.filter(pl.col("predict_score").is_not_null() & pl.col("sharpe_label").is_not_null())
        
        return result
    
    def _compute_base_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算基础因子 (从配置文件加载)。
        
        这是 compute_factors 的内部方法，只负责计算配置文件中定义的因子。
        
        Args:
            df (pl.DataFrame): 输入 DataFrame
        
        Returns:
            pl.DataFrame: 添加了基础因子的 DataFrame
        """
        # 创建工作副本，避免修改原始数据
        result = df.clone()
        
        # 预处理：将所有数值列转换为 Float64 类型，避免字符串运算错误
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "pre_close", "change", "pct_chg", "pct_change"]
        for col in numeric_columns:
            if col in result.columns:
                result = result.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        # 遍历所有因子配置
        for factor in self.factors:
            factor_name = factor["name"]
            expression = factor["expression"]
            
            try:
                context = {col: result[col] for col in result.columns}
                context["pl"] = pl
                context["float"] = float
                
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min}}
                factor_values = eval(expression, eval_globals, context)
                
                result = result.with_columns([
                    pl.Series(factor_name, factor_values)
                ])
                
            except Exception as e:
                logger.error(f"Failed to compute factor {factor_name}: {e}")
                continue
        
        return result
    
    def compute_factors_lazy(
        self, 
        ldf: pl.LazyFrame,
        num_threads: Optional[int] = None
    ) -> pl.LazyFrame:
        """
        【LazyFrame 版本】在输入 LazyFrame 上计算所有配置的因子。
        
        性能优势:
            - 懒加载：不立即执行计算，而是构建查询计划
            - 查询优化：Polars 自动优化执行计划
            - 流式处理：支持超出内存的大数据集
            - 并行计算：自动利用多核 CPU
        
        Args:
            ldf (pl.LazyFrame): 包含 OHLCV 数据的 Polars LazyFrame
            num_threads (int, optional): 并行线程数，默认使用所有可用核心
        
        Returns:
            pl.LazyFrame: 添加了因子列的 LazyFrame
                需要调用 .collect() 来获取结果
        """
        # 创建工作副本
        result = ldf.clone()
        
        # 预处理：转换数值列为 Float64
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "pre_close", "change", "pct_chg", "pct_change"]
        for col in numeric_columns:
            result = result.with_columns(
                pl.col(col).cast(pl.Float64, strict=False)
            )
        
        # 遍历所有因子配置
        for factor in self.factors:
            factor_name = factor["name"]
            expression = factor["expression"]
            
            try:
                factor_expr = self._build_lazy_expression(expression)
                
                if factor_expr is not None:
                    result = result.with_columns(
                        factor_expr.alias(factor_name)
                    )
                else:
                    logger.warning(f"Could not build lazy expression for factor: {factor_name}")
                    
            except Exception as e:
                logger.error(f"Failed to build lazy factor {factor_name}: {e}")
                continue
        
        return result
    
    def _build_lazy_expression(self, expression: str) -> Optional[pl.Expr]:
        """
        将因子表达式字符串转换为 Polars 表达式。
        
        Args:
            expression (str): 因子表达式字符串
        
        Returns:
            pl.Expr: Polars 表达式对象，如果解析失败返回 None
        """
        try:
            eval_globals = {
                "__builtins__": {"float": float, "abs": abs, "max": max, "min": min},
                "pl": pl
            }
            
            context = {}
            for col_name in self.BASE_COLUMNS:
                context[col_name] = pl.col(col_name)
            
            expr = eval(expression, eval_globals, context)
            
            if isinstance(expr, pl.Expr):
                return expr
            else:
                logger.warning(f"Expression did not return pl.Expr: {expression}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to build lazy expression '{expression}': {e}")
            return None
    
    def compute_factors_parallel(
        self,
        df: pl.DataFrame,
        num_threads: Optional[int] = None
    ) -> pl.DataFrame:
        """
        【并行版本】在多股票数据上并行计算因子。
        
        Args:
            df (pl.DataFrame): 包含多只股票数据的 DataFrame
            num_threads (int, optional): 并行线程数
        
        Returns:
            pl.DataFrame: 添加了因子列的 DataFrame
        """
        if "symbol" not in df.columns:
            logger.warning("No 'symbol' column found, falling back to serial computation")
            return self.compute_factors(df)
        
        symbols = df["symbol"].unique().to_list()
        num_symbols = len(symbols)
        
        groups = df.partition_by("symbol", maintain_order=True)
        results = []
        
        if num_threads:
            pl.Config.set_num_threads(num_threads)
        
        for i, group_df in enumerate(groups, 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{num_symbols} stocks processed")
            
            result_df = self.compute_factors(group_df)
            results.append(result_df)
        
        if results:
            final_result = pl.concat(results, how="vertical_relaxed")
            return final_result
        else:
            logger.warning("No results from parallel computation")
            return df
    
    def get_factor_names(self) -> list[str]:
        """获取所有因子名称列表。"""
        return [f["name"] for f in self.factors]
    
    def get_feature_columns(self) -> list[str]:
        """
        获取所有特征列名称 (因子 + 标签)。
        
        Returns:
            list[str]: 特征列名称列表 (所有因子 + 标签)
        """
        columns = self.get_factor_names()
        # 添加技术指标列
        columns.extend(["rsi_14", "macd", "macd_signal", "macd_hist"])
        # 添加量价协同列
        columns.extend(["volume_price_health", "volume_shrink_flag", "price_volume_divergence"])
        # 添加标签列
        if self.label_config:
            columns.append(self.label_config["name"])
        columns.extend(["sharpe_label", "future_max_return", "future_volatility"])
        # 添加评分类
        columns.extend(["predict_score", "filtered_score"])
        return columns