"""
Model Trainer Module - LightGBM model training for stock selection.

This module handles:
- Feature/label preparation from Parquet files
- LightGBM model training with cross-validation
- Model persistence and evaluation

核心功能:
    - 使用 LightGBM 训练股票选择模型
    - 时间序列交叉验证
    - 特征重要性分析
    - 模型保存和加载
"""

import lightgbm as lgb
import polars as pl
from pathlib import Path
from typing import Any
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
import numpy as np


class ModelTrainer:
    """
    LightGBM 模型训练器，用于股票选择。
    
    功能特性:
        - 时间序列交叉验证 (TimeSeriesSplit)
        - 特征重要性分析
        - 模型持久化 (保存/加载)
        - 早停机制防止过拟合
    
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
    
    使用示例:
        >>> trainer = ModelTrainer()  # 使用默认参数
        >>> model = trainer.train(X_train, y_train, X_val, y_val)  # 训练模型
        >>> predictions = trainer.predict(X_test)  # 预测
        >>> trainer.save_model("models/my_model.txt")  # 保存模型
    """
    
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        """
        使用超参数初始化模型训练器。
        
        Args:
            n_estimators (int):  boosting 轮数 (树的数量)，默认 1000
                增加轮数可以提高性能，但会增加训练时间和过拟合风险
            
            learning_rate (float): 每棵树的学习率，默认 0.05
                较小的学习率需要更多的树，但通常能获得更好的泛化能力
                常用范围：0.01 - 0.1
            
            max_depth (int): 树的最大深度，默认 6
                控制模型复杂度，深度越大越容易过拟合
                常用范围：3 - 8
            
            num_leaves (int): 每棵树的叶子节点数，默认 31
                LightGBM 的主要复杂度参数，应设为 2^max_depth 以下
                常用范围：15 - 63
            
            min_child_samples (int): 每个叶子节点的最小样本数，默认 100
                控制叶子节点的最小样本数，防止过拟合
                常用范围：20 - 200
            
            subsample (float): 行采样比例，默认 0.8
                每棵树使用的数据比例，<1 可以减少过拟合
                常用范围：0.6 - 1.0
            
            colsample_bytree (float): 列采样比例，默认 0.8
                每棵树使用的特征比例，<1 可以减少过拟合
                常用范围：0.6 - 1.0
            
            random_state (int): 随机种子，默认 42
                确保结果可复现
        
        初始化后创建的属性:
            - self.params: LightGBM 参数字典
            - self.n_estimators: boosting 轮数
            - self.model: 训练后的模型 (初始为 None)
            - self.feature_importance_: 特征重要性字典
        """
        # LightGBM 模型参数配置
        self.params = {
            "objective": "regression",  # 回归任务
            "metric": "mse",  # 评估指标：均方误差
            "boosting_type": "gbdt",  # 梯度提升树
            "num_leaves": num_leaves,  # 叶子节点数
            "max_depth": max_depth,  # 最大深度
            "learning_rate": learning_rate,  # 学习率
            "min_child_samples": min_child_samples,  # 最小叶子样本数
            "subsample": subsample,  # 行采样比例
            "colsample_bytree": colsample_bytree,  # 列采样比例
            "random_state": random_state,  # 随机种子
            "n_jobs": -1,  # 使用所有 CPU 核心
            "verbose": -1,  # 关闭训练日志输出
        }
        self.n_estimators = n_estimators  # boosting 轮数
        self.model: lgb.Booster | None = None  # 训练后的模型
        self.feature_importance_: dict[str, float] = {}  # 特征重要性
    
    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame | None = None,
        y_val: pl.Series | None = None,
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
        
        # 创建 LightGBM 训练数据集
        train_data = lgb.Dataset(X_np, label=y_np)
        
        # 配置回调函数
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),  # 早停：50 轮不改善则停止
            lgb.log_evaluation(period=100),  # 每 100 轮记录日志
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
            
        Raises:
            ValueError: 如果模型尚未训练
            
        使用示例:
            >>> trainer = ModelTrainer()
            >>> trainer.train(X_train, y_train)
            >>> predictions = trainer.predict(X_test)
            >>> print(f"预测收益率：{predictions[:5]}")
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X.to_numpy())
    
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
    
    def save_model(self, path: str) -> None:
        """
        保存训练好的模型到文件。
        
        模型保存为 LightGBM 的文本格式，可以被加载用于预测。
        
        Args:
            path (str): 输出文件路径
                示例："data/models/stock_model.txt"
            
        Raises:
            ValueError: 如果没有训练好的模型
            
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
        
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
    
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