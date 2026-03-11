"""
Factor Engine Module - Compute factors from price/volume data using Polars.

This module reads factor configurations from YAML files and computes
factors using vectorized operations with Polars DataFrame.

核心功能:
    - 从 YAML 配置文件加载因子定义
    - 使用 Polars 向量化操作计算因子
    - 支持动量、波动率、成交量等多种因子类型
    - 计算预测标签 (label)
    - 结果存储支持 Parquet 格式
"""

import yaml
import polars as pl
from pathlib import Path
from typing import Any
from loguru import logger


class FactorEngine:
    """
    因子计算引擎，使用 Polars 进行向量化计算。
    
    功能特性:
        - 从 YAML 配置文件加载因子定义
        - 使用 eval() 执行向量化因子表达式
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
        - 派生字段的：pct_change (涨跌幅), adj_close (复权收盘价)
    
    使用示例:
        >>> engine = FactorEngine("config/factors.yaml")  # 初始化引擎
        >>> df = pl.DataFrame({"close": [1, 2, 3], "volume": [100, 200, 300]})
        >>> df_with_factors = engine.compute_factors(df)  # 计算因子
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
    }
    
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
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        在输入 DataFrame 上计算所有配置的因子。
        
        此方法遍历所有因子配置，对每个因子:
        1. 提取因子名称和表达式
        2. 使用 eval() 在 DataFrame 列上执行表达式
        3. 将计算结果作为新列添加到 DataFrame
        
        Args:
            df (pl.DataFrame): 包含 OHLCV 数据的 Polars DataFrame
                必需列：close (收盘价)
                可选列：open, high, low, volume, pct_change 等
            
        Returns:
            pl.DataFrame: 添加了因子列的 DataFrame
                原始列 + 所有计算的因子列
        
        使用示例:
            >>> df = pl.DataFrame({
            ...     "close": [10.0, 10.5, 11.0, 10.8, 11.2],
            ...     "volume": [1000, 1200, 1100, 1300, 1400]
            ... })
            >>> engine = FactorEngine("config/factors.yaml")
            >>> result = engine.compute_factors(df)
            >>> print(result.columns)  # 查看包含的列
        
        注意:
            - 因子计算需要足够的历史数据，窗口期外的值为 null
            - 表达式使用 Polars Series 方法，确保语法兼容
        """
        logger.info(f"Computing {len(self.factors)} factors on {len(df)} rows")
        
        # 创建工作副本，避免修改原始数据
        result = df.clone()
        
        # 遍历所有因子配置
        for factor in self.factors:
            factor_name = factor["name"]  # 因子名称
            expression = factor["expression"]  # 因子表达式
            
            try:
                # 为 eval() 创建上下文，将 DataFrame 列作为变量
                # 这样表达式中可以直接使用 close, volume 等列名
                context = {col: result[col] for col in result.columns}
                # 注入 pl (polars) 到上下文中，支持 return_dtype=pl.Float64 等用法
                context["pl"] = pl
                # 注入 float 函数，支持 lambda x: float(x) 用法
                context["float"] = float
                
                # 使用 eval() 执行因子表达式
                # 注入 float 到 globals 以支持 lambda x: float(x) 用法
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min}}
                factor_values = eval(expression, eval_globals, context)
                
                # 将因子值作为新列添加到 DataFrame
                result = result.with_columns(
                    pl.Series(factor_name, factor_values)
                )
                
                logger.debug(f"Computed factor: {factor_name}")
                
            except Exception as e:
                logger.error(f"Failed to compute factor {factor_name}: {e}")
                raise
        
        return result
    
    def compute_label(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        在 DataFrame 上计算预测标签。
        
        标签用于机器学习模型的预测目标，通常是未来收益率。
        
        Args:
            df (pl.DataFrame): 包含 OHLCV 数据的 Polars DataFrame
            
        Returns:
            pl.DataFrame: 添加了标签列的 DataFrame
            
        Raises:
            ValueError: 如果标签配置不存在
            
        使用示例:
            >>> df = pl.DataFrame({"close": [10.0, 10.5, 11.0, 10.8, 11.2]})
            >>> engine = FactorEngine("config/factors.yaml")
            >>> result = engine.compute_label(df)
            >>> print(result["future_return_5"])  # 5 日未来收益率
        
        注意:
            - 标签通常是前视的 (使用 shift(-n))，会产生 null 值
            - 在训练模型时需要去除这些 null 值
        """
        if self.label_config is None:
            raise ValueError("No label configuration found")
        
        label_name = self.label_config["name"]  # 标签名称
        expression = self.label_config["expression"]  # 标签表达式
        
        logger.info(f"Computing label: {label_name}")
        
        # 创建工作副本
        result = df.clone()
        
        try:
            # 创建 eval() 上下文
            context = {col: result[col] for col in result.columns}
            label_values = eval(expression, {"__builtins__": {}}, context)
            
            # 添加标签列
            result = result.with_columns(
                pl.Series(label_name, label_values)
            )
            
        except Exception as e:
            logger.error(f"Failed to compute label: {e}")
            raise
        
        return result
    
    def get_factor_names(self) -> list[str]:
        """
        获取所有因子名称列表。
        
        Returns:
            list[str]: 因子名称列表
                示例：["momentum_5", "momentum_10", "volatility_20", ...]
        
        使用示例:
            >>> engine = FactorEngine("config/factors.yaml")
            >>> names = engine.get_factor_names()
            >>> print(f"共有 {len(names)} 个因子")
        """
        return [f["name"] for f in self.factors]
    
    def get_feature_columns(self) -> list[str]:
        """
        获取所有特征列名称 (因子 + 标签)。
        
        此方法返回的列名可用于:
        - 机器学习模型的特征选择
        - DataFrame 列筛选
        - 模型训练数据准备
        
        Returns:
            list[str]: 特征列名称列表 (所有因子 + 标签)
        
        使用示例:
            >>> engine = FactorEngine("config/factors.yaml")
            >>> columns = engine.get_feature_columns()
            >>> df_features = df.select(columns)
        """
        columns = self.get_factor_names()
        if self.label_config:
            columns.append(self.label_config["name"])
        return columns