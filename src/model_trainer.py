"""
Model Trainer Module - LightGBM model training for stock selection.

This module handles:
- Feature/label preparation from Parquet files or Database
- LightGBM model training with cross-validation
- Model persistence and evaluation
- Factor IC (Information Coefficient) analysis

核心功能:
    - 使用 LightGBM 训练股票选择模型
    - 时间序列交叉验证
    - 特征重要性分析
    - 因子 IC 分析（信息系数）
    - 模型保存和加载
    
【新增 - 2026-03-10】:
    - 因子 IC 分析：计算各因子与预测目标的秩相关性
    - 样本加权：对收益率分布两端样本赋予更高权重
    - 增强正则化：lambda_l1, lambda_l2 防止过拟合
"""

import lightgbm as lgb
import polars as pl
from pathlib import Path
from typing import Any
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from scipy.stats import spearmanr
import os

try:
    from .db_manager import DatabaseManager
    from .factor_engine import FactorEngine
    from .feature_pipeline import FeaturePipeline
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine
    from feature_pipeline import FeaturePipeline


class ModelTrainer:
    """
    LightGBM 模型训练器，用于股票选择。
    
    功能特性:
        - 时间序列交叉验证 (TimeSeriesSplit)
        - 特征重要性分析
        - 模型持久化 (保存/加载)
        - 早停机制防止过拟合
        - 支持从 Parquet 文件加载数据
        - 【新增】因子 IC 分析
        - 【新增】样本加权训练
    
    模型参数说明:
        - objective: "regression" (回归任务)
        - metric: "mse" (均方误差)
        - boosting_type: "gbdt" (梯度提升树)
        - num_leaves: 叶子节点数
        - max_depth: 树的最大深度
        - learning_rate: 学习率
        - min_child_samples: 每个叶子节点的最小样本数
        - subsample: 行采样比例
        - colsample_bytree: 列采样比例
        - lambda_l1: L1 正则化
        - lambda_l2: L2 正则化
    
    使用示例:
        >>> trainer = ModelTrainer()  # 使用默认参数
        >>> model = trainer.train(X_train, y_train, X_val, y_val)  # 训练模型
        >>> predictions = trainer.predict(X_test)  # 预测
        >>> trainer.save_model("models/my_model.txt")  # 保存模型
    """
    
    @staticmethod
    def load_parquet(path: str) -> pl.DataFrame:
        """
        从 Parquet 文件加载数据。
        
        Args:
            path (str): Parquet 文件路径
            
        Returns:
            pl.DataFrame: 加载的数据
            
        使用示例:
            >>> df = ModelTrainer.load_parquet("data/parquet/features_latest.parquet")
            >>> print(f"Loaded {len(df)} rows")
        """
        logger.info(f"Loading Parquet file: {path}")
        df = pl.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    
    @staticmethod
    def load_data_from_db(
        db: DatabaseManager,
        table_name: str = "stock_daily",
        start_date: str = None,
        end_date: str = None,
        symbols: list[str] = None,
    ) -> pl.DataFrame:
        """
        从数据库加载数据并计算因子。
        
        Args:
            db (DatabaseManager): 数据库管理器
            table_name (str): 数据表名
            start_date (str): 开始日期 (YYYY-MM-DD)
            end_date (str): 结束日期 (YYYY-MM-DD)
            symbols (list[str]): 股票代码列表
            
        Returns:
            pl.DataFrame: 包含因子的数据
        """
        logger.info("Loading data from database...")
        
        # 构建查询
        conditions = []
        if start_date:
            conditions.append(f"trade_date >= '{start_date}'")
        if end_date:
            conditions.append(f"trade_date <= '{end_date}'")
        if symbols:
            symbols_str = "', '".join(symbols)
            conditions.append(f"symbol IN ('{symbols_str}')")
        
        query = f"SELECT * FROM {table_name}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY symbol, trade_date"
        
        df = db.read_sql(query)
        logger.info(f"Loaded {len(df)} rows from database")
        
        return df
    
    @staticmethod
    def filter_constant_features(
        df: pl.DataFrame,
        feature_columns: list[str],
        std_threshold: float = 1e-6,
    ) -> list[str]:
        """
        过滤掉标准差接近 0 的常量特征。
        
        Args:
            df (pl.DataFrame): 输入 DataFrame
            feature_columns (list[str]): 特征列名列表
            std_threshold (float): 标准差阈值，低于此值视为常量特征
        
        Returns:
            list[str]: 过滤后的特征列名列表
        """
        logger.info(f"Filtering constant features (threshold={std_threshold})...")
        
        valid_features = []
        removed_features = []
        
        for col in feature_columns:
            if col in df.columns:
                std_val = df[col].std()
                if std_val is not None and std_val > std_threshold:
                    valid_features.append(col)
                else:
                    removed_features.append((col, std_val))
        
        if removed_features:
            logger.info(f"Removed {len(removed_features)} constant features:")
            for name, std in removed_features[:10]:
                logger.info(f"  - {name}: std={std:.6e}")
        
        logger.info(f"Remaining features: {len(valid_features)} (removed {len(removed_features)})")
        return valid_features
    
    @staticmethod
    def prepare_data(
        df: pl.DataFrame,
        feature_columns: list[str],
        label_column: str = "future_return_5",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        filter_constant_features: bool = True,
    ) -> dict[str, Any]:
        """
        准备训练/验证/测试数据，执行时间序列切分。
        
        时间序列切分原则:
            - 训练集：前 70% 的数据 (按时间排序)
            - 验证集：中间 15% 的数据
            - 测试集：最后 15% 的数据
        
        这种切分方式确保:
            - 不使用未来数据预测过去
            - 符合实际交易场景
        
        Args:
            df (pl.DataFrame): 包含特征和标签的 DataFrame
            feature_columns (list[str]): 特征列名列表
                示例：["momentum_5", "volatility_20", ...]
            label_column (str): 标签列名，默认 "future_return_5"
            train_ratio (float): 训练集比例，默认 0.7
            val_ratio (float): 验证集比例，默认 0.15
            test_ratio (float): 测试集比例，默认 0.15
            filter_constant_features (bool): 是否过滤常量特征
        
        Returns:
            dict[str, Any]: 包含以下键的字典:
                - "X_train", "y_train": 训练集
                - "X_val", "y_val": 验证集
                - "X_test", "y_test": 测试集
                - "feature_columns": 特征列名列表（过滤后）
            
        使用示例:
            >>> df = pl.read_parquet("data/parquet/features_latest.parquet")
            >>> features = ["momentum_5", "volatility_20"]
            >>> data = ModelTrainer.prepare_data(df, features)
            >>> print(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}")
        
        注意:
            - 数据必须按时间排序 (trade_date 升序)
            - 会自动删除标签列的空值
        """
        logger.info(f"Preparing data with {len(df)} rows")
        
        # 删除标签列的空值
        df = df.drop_nulls(subset=[label_column])
        logger.info(f"After dropping null labels: {len(df)} rows")
        
        # 【新增】过滤常量特征
        if filter_constant_features:
            feature_columns = ModelTrainer.filter_constant_features(df, feature_columns)
        
        # 计算切分点
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # 时间序列切分
        train_df = df.slice(0, train_end)
        val_df = df.slice(train_end, val_end - train_end)
        test_df = df.slice(val_end, n - val_end)
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # 提取特征和标签
        X_train = train_df.select(feature_columns)
        y_train = train_df[label_column]
        
        X_val = val_df.select(feature_columns)
        y_val = val_df[label_column]
        
        X_test = test_df.select(feature_columns)
        y_test = test_df[label_column]
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_columns,
        }
    
    @staticmethod
    def convert_to_multiclass_labels(
        y: np.ndarray,
        use_quantiles: bool = True,   # 【Iteration 7】启用分位数方法
        quantile_threshold: float = 0.15,  # 【Iteration 7】上下各 15% 分位数
    ) -> np.ndarray:
        """
        将连续收益率转换为三分类标签。
        
        【重构 - 三分类模型 2026-03-14】
        【Iteration 7 调优 - 2026-03-14】
        标签定义 (分位数方法):
            - 类别 2 (Long): 未来 5 日超额收益 > 上分位数 (约前 15%)
            - 类别 1 (Hold): 未来 5 日超额收益在中间 (约 70%)
            - 类别 0 (Short): 未来 5 日超额收益 < 下分位数 (约后 15%)
        
        Args:
            y (np.ndarray): 连续收益率值
            use_quantiles (bool): 是否使用分位数方法
            quantile_threshold (float): 分位数阈值 (0-0.5)
        
        Returns:
            np.ndarray: 三分类标签数组 (0, 1, 2)
        
        使用示例:
            >>> y = np.array([0.05, 0.01, -0.05, 0.0, -0.01])
            >>> labels = ModelTrainer.convert_to_multiclass_labels(y, use_quantiles=True)
        """
        labels = np.ones_like(y, dtype=np.int32)  # 默认为类别 1 (Hold)
        
        if use_quantiles:
            # 使用分位数方法，确保类别平衡
            upper_quantile = np.quantile(y, 1 - quantile_threshold)
            lower_quantile = np.quantile(y, quantile_threshold)
            
            logger.info(f"Using quantile method: upper={upper_quantile:.4f}, lower={lower_quantile:.4f}")
            
            labels[y > upper_quantile] = 2  # 类别 2 (Long) - 前 15%
            labels[y < lower_quantile] = 0  # 类别 0 (Short) - 后 15%
        else:
            # 使用固定阈值方法
            upper_threshold = 0.05
            lower_threshold = -0.05
            labels[y > upper_threshold] = 2
            labels[y < lower_threshold] = 0
        
        logger.info(f"Converted to multiclass labels: Class 0={np.sum(labels==0)}, Class 1={np.sum(labels==1)}, Class 2={np.sum(labels==2)}")
        return labels
    
    @staticmethod
    def calculate_sample_weights(
        y: np.ndarray,
        weight_method: str = "tail_focus",
        tail_threshold: float = 0.3,
    ) -> np.ndarray:
        """
        计算样本权重，让模型更关注"捕捉大机会"和"躲避大风险"。
        
        Args:
            y (np.ndarray): 标签值（未来收益率）
            weight_method (str): 权重计算方法
                - "tail_focus": 对收益率分布两端（大涨大跌）赋予更高权重
                - "uniform": 均匀权重（默认 baseline）
            tail_threshold (float): 尾部阈值，默认 0.3（30% 分位数）
        
        Returns:
            np.ndarray: 样本权重数组
        
        【新增 - 2026-03-10】:
            样本加权策略让模型更关注极端收益率样本，
            提高对"大机会"和"大风险"的预测能力。
        """
        if weight_method == "uniform":
            return np.ones_like(y)
        
        # tail_focus: 对收益率分布两端赋予更高权重
        # 使用分位数识别极端样本
        lower_quantile = np.percentile(np.abs(y), (1 - tail_threshold) * 100)
        
        # 权重与绝对收益率成正比
        weights = 1.0 + np.abs(y) / (lower_quantile + 1e-10)
        
        # 归一化权重，使平均权重为 1
        weights = weights / np.mean(weights)
        
        logger.info(f"Sample weights calculated: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
        return weights
    
    @staticmethod
    def calculate_factor_ic(
        features: pl.DataFrame,
        labels: pl.Series,
        feature_columns: list[str],
        save_polarity: bool = True,
        polarity_output_path: str = "data/factor_polarity.json",
    ) -> dict[str, float]:
        """
        计算各因子与标签的 IC（信息系数）- 秩相关性。
        
        IC（Information Coefficient）是衡量因子预测能力的重要指标，
        通过计算因子值与未来收益率的 Spearman 秩相关系数得到。
        
        【新增 - Iteration 2: 滚动 IC 极性管理】
        - 若某个因子的 IC 为负且统计显著，自动对其取反（X = -X）
        - 将此操作记录在 data/factor_polarity.json 中
        - 目的：将"因子失效"转化为"反转信号"，实现顺势交易
        
        Args:
            features (pl.DataFrame): 特征 DataFrame
            labels (pl.Series): 标签 Series（未来收益率）
            feature_columns (list[str]): 特征列名列表
            save_polarity (bool): 是否保存因子极性信息
            polarity_output_path (str): 极性输出文件路径
        
        Returns:
            dict[str, float]: 各因子的 IC 值字典
            
        【新增 - 2026-03-10】:
            IC 分析帮助识别真正有预测力的 Alpha 因子，
            IC 绝对值越大，说明因子预测能力越强。
            通常 IC>0.05 被认为是有意义的因子。
        """
        logger.info(f"Calculating Factor IC for {len(feature_columns)} factors...")
        
        ic_dict = {}
        polarity_dict = {}  # 记录因子极性：{factor_name: {"ic": float, "polarity": str, "reversed": bool}}
        labels_np = labels.to_numpy()
        
        for col in feature_columns:
            try:
                factor_values = features[col].to_numpy()
                
                # 处理空值
                valid_mask = ~(np.isnan(factor_values) | np.isnan(labels_np))
                if valid_mask.sum() < 10:
                    ic_dict[col] = 0.0
                    polarity_dict[col] = {"ic": 0.0, "polarity": "neutral", "reversed": False, "reason": "insufficient_data"}
                    continue
                
                # 计算 Spearman 秩相关系数
                ic, p_value = spearmanr(factor_values[valid_mask], labels_np[valid_mask])
                ic = ic if not np.isnan(ic) else 0.0
                ic_dict[col] = ic
                
                # 【Iteration 2: 滚动 IC 极性管理】
                # 判断 IC 是否统计显著 (p < 0.05)
                is_significant = p_value < 0.05 if not np.isnan(p_value) else False
                
                # 确定因子极性
                if ic > 0.02 and is_significant:
                    polarity = "positive"
                    reversed_flag = False
                    reason = "significant_positive_ic"
                elif ic < -0.02 and is_significant:
                    polarity = "negative"
                    reversed_flag = True  # 需要取反
                    reason = "significant_negative_ic_reversal_signal"
                else:
                    polarity = "neutral"
                    reversed_flag = False
                    reason = "ic_not_significant_or_weak"
                
                polarity_dict[col] = {
                    "ic": float(ic),
                    "p_value": float(p_value) if not np.isnan(p_value) else None,
                    "polarity": polarity,
                    "reversed": reversed_flag,
                    "reason": reason,
                    "is_significant": is_significant,
                }
                
            except Exception as e:
                logger.warning(f"Failed to calculate IC for {col}: {e}")
                ic_dict[col] = 0.0
                polarity_dict[col] = {"ic": 0.0, "polarity": "neutral", "reversed": False, "reason": f"error: {str(e)}"}
        
        # 按 IC 绝对值排序
        ic_sorted = sorted(ic_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        
        logger.info("=" * 50)
        logger.info("Factor IC Analysis (Top 10):")
        logger.info("=" * 50)
        
        # 统计极性分布
        positive_count = sum(1 for p in polarity_dict.values() if p["polarity"] == "positive")
        negative_count = sum(1 for p in polarity_dict.values() if p["polarity"] == "negative")
        neutral_count = sum(1 for p in polarity_dict.values() if p["polarity"] == "neutral")
        reversed_count = sum(1 for p in polarity_dict.values() if p["reversed"])
        total_factors = len(polarity_dict)
        
        for i, (name, ic) in enumerate(ic_sorted[:10], 1):
            polarity_info = polarity_dict.get(name, {})
            polarity = polarity_info.get("polarity", "unknown")
            reversed_flag = polarity_info.get("reversed", False)
            
            if polarity == "negative":
                logger.warning(f"  {i}. {name}: IC = {ic:.4f} ⚠️ [{polarity}] -> 需要取反 (反转信号)")
            elif polarity == "positive":
                logger.info(f"  {i}. {name}: IC = {ic:.4f} ✓ [{polarity}]")
            else:
                logger.info(f"  {i}. {name}: IC = {ic:.4f} [{polarity}]")
        
        # 输出极性统计
        logger.info(f"\n{'='*50}")
        logger.info(f"[IC POLARITY SUMMARY]")
        logger.info(f"  Total factors: {total_factors}")
        logger.info(f"  Positive IC (significant): {positive_count}")
        logger.info(f"  Negative IC (significant): {negative_count} -> 需要取反")
        logger.info(f"  Neutral/Weak IC: {neutral_count}")
        logger.info(f"  Factors to be reversed: {reversed_count}")
        logger.info(f"{'='*50}")
        
        # 【Iteration 2: 保存因子极性信息】
        if save_polarity:
            import json
            from pathlib import Path
            
            # 确保输出目录存在
            output_path = Path(polarity_output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建输出数据
            polarity_data = {
                "timestamp": str(datetime.now()) if 'datetime' in globals() else str(__import__('datetime').datetime.now()),
                "summary": {
                    "total_factors": total_factors,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "neutral_count": neutral_count,
                    "reversed_count": reversed_count,
                },
                "factors": polarity_dict,
            }
            
            # 保存为 JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(polarity_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Factor polarity saved to: {output_path}")
        
        return ic_dict
    
    @staticmethod
    def calculate_shuffle_importance(
        model: lgb.Booster,
        X: pl.DataFrame,
        y: pl.Series,
        feature_columns: list[str],
        n_repeats: int = 5,
        random_state: int = 42,
    ) -> dict[str, float]:
        """
        计算排列重要性（Shuffle Importance）- 通过随机打乱特征值来评估特征重要性。
        
        排列重要性原理:
            1. 计算原始模型的基准得分（如 MSE）
            2. 对每个特征，随机打乱其值
            3. 用打乱后的数据计算新得分
            4. 重要性 = 新得分 - 基准得分
            5. 重复多次取平均
        
        与 Gain 重要性的区别:
            - Gain 重要性：基于信息增益，可能高估某些特征
            - Shuffle Importance：基于预测性能下降，更可靠
            
        Args:
            model (lgb.Booster): 训练好的 LightGBM 模型
            X (pl.DataFrame): 特征 DataFrame
            y (pl.Series): 标签 Series
            feature_columns (list[str]): 特征列名列表
            n_repeats (int): 重复次数，默认 5
            random_state (int): 随机种子，默认 42
        
        Returns:
            dict[str, float]: 各特征的排列重要性字典
                - 正值：特征对预测有贡献
                - 负值：特征可能是噪声
                - 接近 0：特征对预测无影响
            
        【新增 - 2026-03-11】:
            防御性重训方案：剔除随机扰动后对预测无贡献的"干扰因子"
        """
        logger.info(f"Calculating Shuffle Importance for {len(feature_columns)} features...")
        
        np.random.seed(random_state)
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        
        # 计算基准得分
        baseline_pred = model.predict(X_np)
        baseline_mse = np.mean((baseline_pred - y_np) ** 2)
        logger.info(f"Baseline MSE: {baseline_mse:.6f}")
        
        importance_dict = {}
        
        for col_idx, col in enumerate(feature_columns):
            logger.info(f"  Evaluating {col} ({col_idx + 1}/{len(feature_columns)})")
            
            importance_scores = []
            
            for repeat in range(n_repeats):
                # 复制数据
                X_shuffled = X_np.copy()
                
                # 随机打乱当前特征
                shuffled_indices = np.random.permutation(len(y_np))
                X_shuffled[:, col_idx] = X_np[:, col_idx][shuffled_indices]
                
                # 计算打乱后的得分
                shuffled_pred = model.predict(X_shuffled)
                shuffled_mse = np.mean((shuffled_pred - y_np) ** 2)
                
                # 重要性 = 得分下降量
                importance_scores.append(shuffled_mse - baseline_mse)
            
            # 取平均
            importance_dict[col] = np.mean(importance_scores)
        
        # 按重要性排序
        importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("=" * 50)
        logger.info("Shuffle Importance Analysis (Top 10):")
        logger.info("=" * 50)
        for i, (name, imp) in enumerate(importance_sorted[:10], 1):
            logger.info(f"  {i}. {name}: {imp:.6f}")
        
        # 识别干扰因子（重要性接近 0 或为负）
        noise_features = [name for name, imp in importance_dict.items() if imp < 0.0001]
        if noise_features:
            logger.info(f"\nNoise features (shuffle importance ≈ 0): {len(noise_features)}")
            for name in noise_features[:10]:
                logger.info(f"  - {name}")
        
        return importance_dict
    
    def __init__(
        self,
        n_estimators: int = 1500,  # 【重构 - 三分类】增加至 1500 轮，让模型充分学习
        learning_rate: float = 0.005,  # 【重构 - 三分类】学习率
        max_depth: int = 6,  # 【重构 - 三分类】增加深度
        num_leaves: int = 31,  # 【重构 - 三分类】增加叶子节点
        min_child_samples: int = 30,  # 【重构 - 三分类】增加样本数防止过拟合
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        # 【重构 - 三分类】多分类参数
        num_class: int = 3,  # 三分类：0=Short, 1=Hold, 2=Long
    ) -> None:
        """
        使用超参数初始化模型训练器。
        
        【重构 - 三分类模型 2026-03-14】
        将回归模型改为三分类模型，预测股票未来 5 日趋势：
        - 类别 0 (Short): 未来 5 日超额收益 < -3%
        - 类别 1 (Hold): 未来 5 日超额收益在 [-3%, +3%] 之间
        - 类别 2 (Long): 未来 5 日超额收益 > +3%
        
        Args:
            n_estimators (int): boosting 轮数 (树的数量)，默认 1500
            learning_rate (float): 每棵树的学习率，默认 0.005
            max_depth (int): 树的最大深度，默认 6
            num_leaves (int): 每棵树的叶子节点数，默认 31
            min_child_samples (int): 每个叶子节点的最小样本数，默认 30
            subsample (float): 行采样比例，默认 0.8
            colsample_bytree (float): 列采样比例，默认 0.8
            random_state (int): 随机种子，默认 42
            lambda_l1 (float): L1 正则化参数，默认 0.0
            lambda_l2 (float): L2 正则化参数，默认 0.0
            num_class (int): 分类数量，默认 3（三分类）
        
        初始化后创建的属性:
            - self.params: LightGBM 参数字典
            - self.n_estimators: boosting 轮数
            - self.model: 训练后的模型 (初始为 None)
            - self.feature_importance_: 特征重要性字典
            - self.factor_ic_: 因子 IC 值
            - self.num_class: 分类数量
        """
        # LightGBM 模型参数配置 - 【重构 - 三分类】使用多分类目标
        self.params = {
            "objective": "multiclass",  # 【重构 - 三分类】多分类任务
            "num_class": num_class,  # 【重构 - 三分类】三分类
            "metric": "multi_logloss",  # 【重构 - 三分类】多分类对数损失
            "boosting_type": "gbdt",  # 梯度提升树
            "num_leaves": num_leaves,  # 叶子节点数
            "max_depth": max_depth,  # 最大深度
            "learning_rate": learning_rate,  # 学习率
            "min_child_samples": min_child_samples,  # 最小样本数
            "subsample": subsample,  # 行采样比例
            "colsample_bytree": colsample_bytree,  # 列采样比例
            "feature_fraction": 0.8,  # 列采样
            "bagging_fraction": 0.8,  # 行采样
            "bagging_freq": 5,  # 每 5 轮进行一次 bagging
            "random_state": random_state,  # 随机种子
            "n_jobs": -1,  # 使用所有 CPU 核心
            "verbose": 1,  # 输出训练日志
            # 正则化
            "lambda_l1": lambda_l1,  # L1 正则化
            "lambda_l2": lambda_l2,  # L2 正则化
            "min_data_in_leaf": min_child_samples,  # 叶子节点最小样本数
        }
        self.n_estimators = n_estimators  # boosting 轮数
        self.model: lgb.Booster | None = None  # 训练后的模型
        self.feature_importance_: dict[str, float] = {}  # 特征重要性
        self.factor_ic_: dict[str, float] = {}  # 因子 IC 值
        self.num_class = num_class  # 【重构 - 三分类】分类数量
    
    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.Series | None = None,
        sample_weights: np.ndarray | None = None,
    ) -> lgb.Booster:
        """
        训练 LightGBM 模型。
        
        训练流程:
        1. 将 Polars DataFrame/Series 转换为 numpy 数组
        2. 创建 LightGBM 数据集
        3. 配置早停和日志回调
        4. 训练模型 (有/无验证集)
        5. 提取特征重要性
        
        Args:
            X_train (pl.DataFrame): 训练特征 DataFrame
                列应为因子名称，如 ["momentum_5", "volatility_20", ...]
            
            y_train (pl.Series): 训练标签 Series
                通常是未来收益率，如 future_return_5
            
            X_val (pl.DataFrame, optional): 验证特征 DataFrame
                用于早停和验证集评估，如果提供则启用早停
            
            y_val (pl.Series, optional): 验证标签 Series
            
            sample_weights (np.ndarray, optional): 样本权重
                如果为 None，则使用均匀权重
                可使用 calculate_sample_weights() 生成
            
        Returns:
            lgb.Booster: 训练好的 LightGBM 模型
            
        早停机制:
            - 如果提供验证集，当验证集误差连续 50 轮不下降时停止
            - 每 100 轮记录一次训练日志
            
        使用示例:
            >>> trainer = ModelTrainer(n_estimators=500)
            >>> model = trainer.train(X_train, y_train, X_val, y_val)
            >>> print(f"最佳迭代轮数：{model.best_iteration}")
        """
        logger.info(f"Training on {len(X_train)} samples")
        
        # 将 Polars 数据转换为 numpy 数组 (LightGBM 需要)
        X_np = X_train.to_numpy()
        y_np = y_train.to_numpy()
        
        # 计算样本权重（如果未提供）
        if sample_weights is None:
            sample_weights = self.calculate_sample_weights(y_np)
        
        # 创建 LightGBM 训练数据集（带样本权重）
        train_data = lgb.Dataset(X_np, label=y_np, weight=sample_weights)
        
        # 配置回调函数 - 【优化】调整早停和日志频率
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),  # 【优化】早停：100 轮不改善则停止
            lgb.log_evaluation(period=50),  # 【优化】每 50 轮记录日志，更细致观察训练过程
        ]
        
        # 根据是否有验证集选择训练方式
        if X_val is not None and y_val is not None:
            # 创建验证数据集
            val_data = lgb.Dataset(
                X_val.to_numpy(),
                label=y_val.to_numpy(),
                reference=train_data,  # 参考训练集进行特征对齐
            )
            
            # 带验证集的训练
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                valid_sets=[train_data, val_data],
                valid_names=["train", "valid"],
                callbacks=callbacks,
            )
        else:
            # 无验证集的训练 (不使用早停)
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                callbacks=callbacks,
            )
        
        # 提取特征重要性 (基于 gain)
        if X_train.columns:
            importance = self.model.feature_importance(importance_type="gain")
            self.feature_importance_ = dict(
                zip(X_train.columns, importance.tolist())
            )
        
        logger.info(f"Training complete, best iteration: {self.model.best_iteration}")
        return self.model
    
    def cross_validate(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        n_splits: int = 5,
    ) -> dict[str, list[float]]:
        """
        执行时间序列交叉验证。
        
        使用时间序列交叉验证 (TimeSeriesSplit) 而不是普通 K 折交叉验证，
        因为金融数据具有时间序列特性，不能使用未来数据预测过去。
        
        TimeSeriesSplit 工作原理:
            - 第 1 折：训练 [0], 验证 [1]
            - 第 2 折：训练 [0, 1], 验证 [2]
            - 第 3 折：训练 [0, 1, 2], 验证 [3]
            - ...
        
        Args:
            X (pl.DataFrame): 特征 DataFrame
            y (pl.Series): 标签 Series
            n_splits (int): 交叉验证折数，默认 5
            
        Returns:
            dict[str, list[float]]: 每折的训练和验证得分
                - "train": 训练集 MSE 列表
                - "valid": 验证集 MSE 列表
            
        使用示例:
            >>> trainer = ModelTrainer()
            >>> scores = trainer.cross_validate(X, y, n_splits=5)
            >>> print(f"平均验证 MSE: {np.mean(scores['valid']):.6f}")
        
        注意:
            - 时间序列 CV 确保验证集总是在训练集之后
            - 防止数据泄露，更符合实际交易场景
        """
        logger.info(f"Running {n_splits}-fold time-series CV")
        
        # 创建时间序列交叉验证器
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores: dict[str, list[float]] = {"train": [], "valid": []}
        
        # 转换为 numpy 数组
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        
        # 遍历每一折
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_np)):
            logger.info(f"Fold {fold + 1}/{n_splits}")
            
            # 按索引分割数据
            X_train = pl.DataFrame(X_np[train_idx])
            y_train = pl.Series(y_np[train_idx])
            X_val = pl.DataFrame(X_np[val_idx])
            y_val = pl.Series(y_np[val_idx])
            
            # 训练模型
            self.train(X_train, y_train, X_val, y_val)
            
            # 预测并计算 MSE
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            train_mse = np.mean((train_pred - y_np[train_idx]) ** 2)
            val_mse = np.mean((val_pred - y_np[val_idx]) ** 2)
            
            scores["train"].append(train_mse)
            scores["valid"].append(val_mse)
            
            logger.info(f"  Train MSE: {train_mse:.6f}, Valid MSE: {val_mse:.6f}")
        
        return scores
    
    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """
        在新数据上进行预测。
        
        Args:
            X (pl.DataFrame): 用于预测的特征 DataFrame
                列必须与训练时相同
            
        Returns:
            np.ndarray: 预测值数组
                对于多分类模型，返回形状为 (n_samples, n_classes) 的概率矩阵
            
        Raises:
            ValueError: 如果模型尚未训练
            
        使用示例:
            >>> trainer = ModelTrainer()
            >>> trainer.train(X_train, y_train)
            >>> predictions = trainer.predict(X_test)
            >>> print(f"预测概率：{predictions[:5]}")
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X.to_numpy())
    
    def predict_class2_prob(self, X: pl.DataFrame) -> np.ndarray:
        """
        获取类别 2 (Long) 的概率值。
        
        【重构 - 三分类模型 2026-03-14】
        对于多分类模型，predict() 返回每个类别的概率分布。
        此方法提取类别 2 (大涨) 的概率作为选股依据。
        
        Args:
            X (pl.DataFrame): 用于预测的特征 DataFrame
                列必须与训练时相同
            
        Returns:
            np.ndarray: 类别 2 的概率数组，范围 [0, 1]
                值越大表示股票大涨的可能性越高
            
        Raises:
            ValueError: 如果模型尚未训练
            AssertionError: 如果模型不是三分类模型
            
        使用示例:
            >>> trainer = ModelTrainer()
            >>> trainer.train(X_train, y_train_multiclass)
            >>> long_probs = trainer.predict_class2_prob(X_test)
            >>> print(f"Long 概率：{long_probs[:5]}")
            >>> # 选取概率>0.4 的股票
            >>> top_stocks = long_probs > 0.4
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.num_class != 3:
            raise ValueError(f"predict_class2_prob requires num_class=3, got num_class={self.num_class}")
        
        # 获取所有类别的概率
        all_probs = self.model.predict(X.to_numpy())
        
        # 提取类别 2 (Long) 的概率 - 索引 2 对应类别 2
        # LightGBM 多分类输出形状：(n_samples, n_classes)
        class2_prob = all_probs[:, 2]
        
        return class2_prob
    
    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """
        获取前 N 个最重要的特征。
        
        特征重要性基于 "gain"(信息增益)，表示该特征对模型预测的贡献程度。
        gain 越大，说明该特征越重要。
        
        Args:
            n (int): 返回的特征数量，默认 10
            
        Returns:
            list[tuple[str, float]]: (特征名称，重要性) 元组列表
                按重要性降序排列
            
        使用示例:
            >>> trainer = ModelTrainer()
            >>> trainer.train(X_train, y_train)
            >>> top_features = trainer.get_top_features(n=5)
            >>> for name, importance in top_features:
            ...     print(f"{name}: {importance:.2f}")
        """
        if not self.feature_importance_:
            return []
        
        # 按重要性降序排序
        sorted_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]
    
    def get_factor_ic(self) -> dict[str, float]:
        """
        获取因子 IC 值字典。
        
        Returns:
            dict[str, float]: 因子 IC 值字典
        """
        return self.factor_ic_
    
    def save_model(self, path: str) -> None:
        """
        保存训练好的模型到文件。
        
        模型保存为 LightGBM 的文本格式，可以被加载用于预测。
        
        【修复 - 2026-03-14】
        1. 使用绝对路径硬化：Path(__file__).resolve().parent.parent / "data" / "models"
        2. 保存校验：save_model 后立即检查文件是否存在
        3. 权限检查：确保对 data/ 目录有写入权限
        
        Args:
            path (str): 输出文件路径
                示例："data/models/stock_model.txt"
            
        Raises:
            ValueError: 如果没有训练好的模型
            RuntimeError: 如果模型保存失败
            
        使用示例:
            >>> trainer = ModelTrainer()
            >>> trainer.train(X_train, y_train)
            >>> trainer.save_model("data/models/my_model.txt")
        
        注意:
            - 保存的模型包含树结构和参数
            - 不包含特征名称，需要单独保存特征列表
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # 【修复 - 2026-03-14】使用绝对路径硬化
        base_dir = Path(__file__).resolve().parent.parent
        model_dir = base_dir / "data" / "models"
        
        # 确保目录存在
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查写入权限
        if not os.access(model_dir, os.W_OK):
            raise PermissionError(f"No write permission for directory: {model_dir}")
        
        # 使用绝对路径
        abs_path = model_dir / Path(path).name
        abs_path = abs_path.resolve()
        
        logger.info(f"Saving model to: {abs_path}")
        self.model.save_model(str(abs_path))
        
        # 【修复 - 2026-03-14】保存校验：立即检查文件是否存在
        if not os.path.exists(abs_path):
            raise RuntimeError(f"Model Save Failed! File does not exist: {abs_path}")
        
        file_size = os.path.getsize(abs_path)
        logger.info(f"Model saved to {abs_path} ({file_size:,} bytes) ✓")
    
    def load_model(self, path: str) -> lgb.Booster:
        """
        从文件加载训练好的模型。
        
        Args:
            path (str): 模型文件路径
            
        Returns:
            lgb.Booster: 加载的 LightGBM 模型
            
        Raises:
            FileNotFoundError: 如果文件不存在
            
        使用示例:
            >>> trainer = ModelTrainer()
            >>> model = trainer.load_model("data/models/my_model.txt")
            >>> predictions = trainer.predict(X_test)
        
        注意:
            - 加载模型后，特征重要性不会自动恢复
            - 如需特征重要性，需重新训练或单独保存
        """
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Model loaded from {path}")
        return self.model


def run_training_from_db(
    config_path: str = "config/factors.yaml",
    model_output_path: str = "data/models/stock_model.txt",
    start_date: str = None,
    end_date: str = None,
    n_estimators: int = 1000,  # 【优化】增加至 1000 轮
    learning_rate: float = 0.01,  # 【优化】降低至 0.01
    max_depth: int = 4,
    lambda_l1: float = 0.0,  # 【优化】移除 L1 正则化
    lambda_l2: float = 0.0,  # 【优化】移除 L2 正则化
    use_sample_weights: bool = True,
) -> dict[str, Any]:
    """
    从数据库运行完整训练流程。
    
    执行流程:
    1. 从数据库读取历史数据
    2. 使用 FactorEngine 计算因子
    3. 准备训练/验证/测试集
    4. 计算因子 IC 分析
    5. 训练 LightGBM 模型（带样本加权）
    6. 保存模型到文件
    7. 输出训练报告
    
    Args:
        config_path (str): 因子配置文件路径
        model_output_path (str): 模型输出文件路径
        start_date (str, optional): 开始日期 (YYYY-MM-DD)
        end_date (str, optional): 结束日期 (YYYY-MM-DD)
        n_estimators (int): boosting 轮数
        learning_rate (float): 学习率
        max_depth (int): 树的最大深度
        lambda_l1 (float): L1 正则化参数
        lambda_l2 (float): L2 正则化参数
        use_sample_weights (bool): 是否使用样本加权
    
    Returns:
        dict[str, Any]: 训练结果
    
    使用示例:
        >>> results = run_training_from_db()
        >>> print(f"Model saved to: {model_output_path}")
    """
    logger.info("=" * 60)
    logger.info("MODEL TRAINING - Starting Full Pipeline")
    logger.info("=" * 60)
    
    # 初始化数据库和因子引擎
    db = DatabaseManager.get_instance()
    factor_engine = FactorEngine(config_path)
    
    # 获取因子列名
    feature_columns = factor_engine.get_factor_names()
    label_column = factor_engine.label_config["name"] if factor_engine.label_config else "future_return_5"
    
    logger.info(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    logger.info(f"Label column: {label_column}")
    
    # Step 1: 从数据库加载数据
    logger.info("=" * 50)
    logger.info("Step 1: Loading data from database")
    logger.info("=" * 50)
    
    query = "SELECT * FROM stock_daily"
    conditions = []
    if start_date:
        conditions.append(f"trade_date >= '{start_date}'")
    if end_date:
        conditions.append(f"trade_date <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY symbol, trade_date"
    
    df = db.read_sql(query)
    
    if df.is_empty():
        raise ValueError("No data found in database. Please run data sync first.")
    
    logger.info(f"Loaded {len(df)} rows")
    
    # Step 2: 准备数据 (计算 pct_change 等)
    logger.info("=" * 50)
    logger.info("Step 2: Preparing data")
    logger.info("=" * 50)
    
    # 转换数值列为 Float64
    numeric_columns = ["open", "high", "low", "close", "volume", "amount", "adj_factor", "turnover_rate", "pre_close", "change", "pct_chg"]
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
    
    # Step 3: 计算因子
    logger.info("=" * 50)
    logger.info("Step 3: Computing factors")
    logger.info("=" * 50)
    
    df_with_factors = factor_engine.compute_factors(df)
    
    # 计算标签
    if factor_engine.label_config:
        df_with_factors = factor_engine.compute_label(df_with_factors)
    
    logger.info(f"Computed factors. Columns: {df_with_factors.columns}")
    
    # Step 4: 准备训练数据
    logger.info("=" * 50)
    logger.info("Step 4: Preparing train/val/test split")
    logger.info("=" * 50)
    
    data = ModelTrainer.prepare_data(
        df=df_with_factors,
        feature_columns=feature_columns,
        label_column=label_column,
    )
    
    # Step 5: 因子 IC 分析
    logger.info("=" * 50)
    logger.info("Step 5: Factor IC Analysis")
    logger.info("=" * 50)
    
    factor_ic = ModelTrainer.calculate_factor_ic(
        features=data["X_train"],
        labels=data["y_train"],
        feature_columns=feature_columns,
    )
    
    # Step 6: 初始化训练器
    trainer = ModelTrainer(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
    )
    
    trainer.factor_ic_ = factor_ic
    
    # Step 7: 计算样本权重
    sample_weights = None
    if use_sample_weights:
        logger.info("=" * 50)
        logger.info("Step 7: Calculating sample weights")
        logger.info("=" * 50)
        sample_weights = ModelTrainer.calculate_sample_weights(
            data["y_train"].to_numpy(),
            weight_method="tail_focus",
        )
    
    # Step 8: 训练模型
    logger.info("=" * 50)
    logger.info("Step 8: Training LightGBM model")
    logger.info("=" * 50)
    
    trainer.train(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        sample_weights=sample_weights,
    )
    
    # Step 9: 输出特征重要性
    top_features = trainer.get_top_features(n=10)
    logger.info("=" * 50)
    logger.info("Top 10 Most Important Features:")
    logger.info("=" * 50)
    for i, (name, importance) in enumerate(top_features, 1):
        logger.info(f"  {i}. {name}: {importance:.2f}")
    
    # Step 10: 测试集评估
    test_pred = trainer.predict(data["X_test"])
    test_mse = np.mean((test_pred - data["y_test"].to_numpy()) ** 2)
    
    train_mse = np.mean((trainer.predict(data["X_train"]) - data["y_train"].to_numpy()) ** 2)
    val_mse = np.mean((trainer.predict(data["X_val"]) - data["y_val"].to_numpy()) ** 2)
    
    logger.info("=" * 50)
    logger.info("Model Evaluation:")
    logger.info("=" * 50)
    logger.info(f"  Train MSE: {train_mse:.6f}")
    logger.info(f"  Valid  MSE: {val_mse:.6f}")
    logger.info(f"  Test   MSE: {test_mse:.6f}")
    
    # Step 11: 保存模型
    logger.info("=" * 50)
    logger.info("Step 11: Saving model")
    logger.info("=" * 50)
    
    trainer.save_model(model_output_path)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Model saved to: {model_output_path}")
    logger.info("=" * 60)
    
    return {
        "model": trainer.model,
        "top_features": top_features,
        "factor_ic": factor_ic,
        "test_mse": test_mse,
        "train_mse": train_mse,
        "val_mse": val_mse,
        "train_samples": len(data["X_train"]),
        "val_samples": len(data["X_val"]),
        "test_samples": len(data["X_test"]),
    }


def run_training(
    parquet_path: str = "data/parquet/features_latest.parquet",
    feature_columns: list[str] = None,
    label_column: str = "label_5d",  # 【重构 - 三分类】使用 5 日趋势标签
    n_estimators: int = 1500,  # 【重构 - 三分类】增加至 1500 轮
    learning_rate: float = 0.005,  # 【重构 - 三分类】学习率
    max_depth: int = 6,  # 【重构 - 三分类】增加深度
    min_child_samples: int = 30,  # 【重构 - 三分类】最小样本数
    feature_fraction: float = 0.8,  # 【重构】列采样
    bagging_fraction: float = 0.8,  # 【重构】行采样
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    use_sample_weights: bool = True,
    # 【重构 - 三分类】标签转换参数 - 【Iteration 7】使用分位数方法平衡类别
    use_quantiles: bool = True,       # 【Iter7】启用分位数方法
    quantile_threshold: float = 0.15,  # 【Iter7】上下各 15% 分位数
    upper_threshold: float = 0.05,    # 【Iter6】固定阈值方法用
    lower_threshold: float = -0.05,   # 【Iter6】固定阈值方法用
) -> dict[str, Any]:
    """
    运行完整的训练流程（从 Parquet 文件）。
    
    此函数执行:
    1. 从 Parquet 文件加载数据
    2. 准备训练/验证/测试集 (时间序列切分)
    3. 【新增】计算因子 IC 分析
    4. 训练 LightGBM 模型（带样本加权）
    5. 输出前 10 个最重要的特征
    6. 在测试集上评估
    
    Args:
        parquet_path (str): Parquet 文件路径
        feature_columns (list[str], optional): 特征列名列表
            如果为 None，则自动检测因子列
        label_column (str): 标签列名
        n_estimators (int): boosting 轮数
        learning_rate (float): 学习率
        max_depth (int): 树的最大深度
        lambda_l1 (float): L1 正则化参数，默认 0.1
        lambda_l2 (float): L2 正则化参数，默认 0.1
        use_sample_weights (bool): 是否使用样本加权，默认 True
    
    Returns:
        dict[str, Any]: 训练结果，包含:
            - "model": 训练好的模型
            - "top_features": 前 10 个重要特征
            - "factor_ic": 因子 IC 值字典
            - "test_mse": 测试集 MSE
            - "train_samples": 训练样本数
            - "val_samples": 验证样本数
            - "test_samples": 测试样本数
    
    使用示例:
        >>> results = run_training()
        >>> print(f"Test MSE: {results['test_mse']:.6f}")
    """
    # 默认特征列 (包含新增因子)
    if feature_columns is None:
        feature_columns = [
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_5", "volatility_20",
            "volume_ma_ratio_5", "volume_ma_ratio_20",
            "price_position_20", "price_position_60",
            "ma_deviation_5", "ma_deviation_20",
            "rsi_14", "mfi_14",  # 新增技术指标因子
            "turnover_bias_20", "turnover_ma_ratio",  # 新增换手率因子
            "volume_price_divergence_5", "volume_price_divergence_20",
            "volume_price_correlation", "smart_money_flow",
            "volatility_contraction_10", "volume_shrink_ratio",
            "volume_price_stable", "accumulation_distribution_20",
        ]
    
    logger.info("=" * 50)
    logger.info("Starting Model Training")
    logger.info("=" * 50)
    
    # Step 1: 加载数据
    df = ModelTrainer.load_parquet(parquet_path)
    
    # Step 2: 准备数据
    data = ModelTrainer.prepare_data(
        df=df,
        feature_columns=feature_columns,
        label_column=label_column,
    )
    
    # 【重构 - 三分类模型 2026-03-14】将连续标签转换为三分类标签
    # 获取未来收益率用于转换
    y_train_continuous = data["y_train"].to_numpy()
    y_val_continuous = data["y_val"].to_numpy()
    y_test_continuous = data["y_test"].to_numpy()
    
    # 转换为三分类标签 (0, 1, 2) - 【Iteration 7】使用分位数方法
    y_train_multiclass = ModelTrainer.convert_to_multiclass_labels(
        y_train_continuous,
        use_quantiles=use_quantiles,
        quantile_threshold=quantile_threshold,
    )
    y_val_multiclass = ModelTrainer.convert_to_multiclass_labels(
        y_val_continuous,
        use_quantiles=use_quantiles,
        quantile_threshold=quantile_threshold,
    )
    y_test_multiclass = ModelTrainer.convert_to_multiclass_labels(
        y_test_continuous,
        use_quantiles=use_quantiles,
        quantile_threshold=quantile_threshold,
    )
    
    # 更新 data 中的标签为多分类标签
    data["y_train"] = pl.Series("y_train", y_train_multiclass)
    data["y_val"] = pl.Series("y_val", y_val_multiclass)
    data["y_test"] = pl.Series("y_test", y_test_multiclass)
    
    # Step 2.5: 【新增】因子 IC 分析
    logger.info("=" * 50)
    logger.info("Factor IC Analysis")
    logger.info("=" * 50)
    factor_ic = ModelTrainer.calculate_factor_ic(
        features=data["X_train"],
        labels=data["y_train"],
        feature_columns=feature_columns,
    )
    
    # Step 3: 初始化训练器 - 【重构】增强鲁棒性和随机性
    trainer = ModelTrainer(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
    )
    
    # 更新参数
    trainer.params["feature_fraction"] = feature_fraction
    trainer.params["bagging_fraction"] = bagging_fraction
    trainer.params["bagging_freq"] = 5
    
    # 保存因子 IC 值
    trainer.factor_ic_ = factor_ic
    
    # Step 4: 计算样本权重（如果启用）
    sample_weights = None
    if use_sample_weights:
        logger.info("=" * 50)
        logger.info("Calculating Sample Weights")
        logger.info("=" * 50)
        sample_weights = ModelTrainer.calculate_sample_weights(
            data["y_train"].to_numpy(),
            weight_method="tail_focus",
        )
    
    # Step 5: 训练模型（带样本权重）
    logger.info("=" * 50)
    logger.info("Training LightGBM model...")
    logger.info(f"Parameters: lambda_l1={lambda_l1}, lambda_l2={lambda_l2}, learning_rate={learning_rate}")
    logger.info("=" * 50)
    trainer.train(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        sample_weights=sample_weights,
    )
    
    # Step 6: 获取特征重要性
    top_features = trainer.get_top_features(n=10)
    logger.info("=" * 50)
    logger.info("Top 10 Most Important Features:")
    logger.info("=" * 50)
    for i, (name, importance) in enumerate(top_features, 1):
        logger.info(f"  {i}. {name}: {importance:.2f}")
    
    # Step 7: 测试集评估
    # 【重构 - 三分类】多分类模型输出形状为 (n_samples, n_classes)
    # 需要提取预测类别（概率最大的类别）
    test_pred_probs = trainer.predict(data["X_test"])
    test_pred_classes = np.argmax(test_pred_probs, axis=1)
    y_test_np = data["y_test"].to_numpy()
    
    # 计算分类准确率
    test_accuracy = np.mean(test_pred_classes == y_test_np)
    
    # 计算类别 2 (Long) 的预测准确率
    class2_mask = y_test_np == 2
    if np.sum(class2_mask) > 0:
        class2_accuracy = np.mean(test_pred_classes[class2_mask] == 2)
    else:
        class2_accuracy = 0.0
    
    train_pred_probs = trainer.predict(data["X_train"])
    train_pred_classes = np.argmax(train_pred_probs, axis=1)
    train_accuracy = np.mean(train_pred_classes == data["y_train"].to_numpy())
    
    val_pred_probs = trainer.predict(data["X_val"])
    val_pred_classes = np.argmax(val_pred_probs, axis=1)
    val_accuracy = np.mean(val_pred_classes == data["y_val"].to_numpy())
    
    logger.info("=" * 50)
    logger.info("Model Evaluation (Classification):")
    logger.info("=" * 50)
    logger.info(f"  Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"  Valid Accuracy: {val_accuracy:.4f}")
    logger.info(f"  Test  Accuracy: {test_accuracy:.4f}")
    logger.info(f"  Class 2 (Long) Accuracy: {class2_accuracy:.4f}")
    
    # 输出混淆矩阵统计
    logger.info("=" * 50)
    logger.info("Prediction Distribution:")
    logger.info("=" * 50)
    for true_class in range(3):
        mask = y_test_np == true_class
        if np.sum(mask) > 0:
            pred_dist = test_pred_classes[mask]
            logger.info(f"  True Class {true_class}: Pred as 0={np.sum(pred_dist==0)}, 1={np.sum(pred_dist==1)}, 2={np.sum(pred_dist==2)}")
    
    # 判断模型是否收敛
    logger.info("=" * 50)
    logger.info("Convergence Analysis:")
    logger.info("=" * 50)
    
    # 计算训练集和验证集的差距
    gap = val_accuracy - train_accuracy
    if abs(gap) < 0.001:
        logger.info(f"  Model is well-fitted (train={train_accuracy:.4f}, val={val_accuracy:.4f})")
    elif abs(gap) < 0.01:
        logger.info(f"  Model shows slight overfitting (gap={gap:.4f}, acceptable)")
    else:
        logger.info(f"  Model shows overfitting (gap={gap:.4f})")
    
    # 检查测试集表现
    if test_accuracy > 0.95:
        logger.info("  Test performance is EXCELLENT")
    elif test_accuracy > 0.90:
        logger.info("  Test performance is GOOD")
    else:
        logger.info("  Test performance needs improvement")
    
    logger.info("=" * 50)
    logger.info("Training Complete!")
    logger.info("=" * 50)
    
    return {
        "model": trainer.model,
        "top_features": top_features,
        "factor_ic": factor_ic,
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "class2_accuracy": class2_accuracy,
        "train_samples": len(data["X_train"]),
        "val_samples": len(data["X_val"]),
        "test_samples": len(data["X_test"]),
    }


if __name__ == "__main__":
    # 优先从数据库训练，如果 Parquet 文件不存在则回退到 Parquet
    import os
    
    model_output = "data/models/stock_model.txt"
    
    # 检查 Parquet 文件是否存在
    parquet_path = "data/parquet/features_latest.parquet"
    
    if os.path.exists(parquet_path):
        print(f"Using Parquet file: {parquet_path}")
        results = run_training(parquet_path=parquet_path)
        
        # 【修复 - 2026-03-14】保存模型
        print(f"\nSaving model to: {model_output}")
        trainer = ModelTrainer()
        trainer.model = results["model"]
        trainer.feature_importance_ = dict(results["top_features"])
        trainer.factor_ic_ = results["factor_ic"]
        trainer.save_model(model_output)
    else:
        print("Parquet file not found, training from database...")
        results = run_training_from_db(model_output_path=model_output)
    
    print(f"\nTraining completed!")
    print(f"Top features: {results['top_features']}")
    if 'test_accuracy' in results:
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Class 2 (Long) Accuracy: {results.get('class2_accuracy', 0):.4f}")
    else:
        print(f"Test MSE: {results.get('test_mse', 0):.6f}")
    print(f"Model saved to: {model_output}")
