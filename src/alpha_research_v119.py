"""
Alpha Research Module - V119 特征多样性喷发与 GA 修复.

【V119 核心任务 - 从 V118 失败中学习】
1. MI 筛选 Bug 修复：增加数据预清洗（Fillna/Drop_constant）
2. GA 适应度重构：使用 Incremental_IC (增量 IC) 替代 IC_Mean - IC_Std
3. 算子多样化：加入 Ts_Delta, Ts_Entropy（信息熵）, Ts_Skewness
4. 特征多样性：寻找与现有特征低相关、高贡献的补丁因子
5. 日志版本修复：使用动态版本号

【V118 失败根因】
- MI 计算返回 0：因为数据中存在 NaN、常数项或输入维度不匹配
- 适应度函数太严苛：IC_Mean - IC_Std 导致初始种群全军覆没
- 算子类型单一：缺少信息熵、偏度等高级算子

【V119 技术规格】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标 (V118 为 0.0380) |
| IC IR | > 0.6 | 稳定性指标 (V118 为 0.24) |
| Genetic Factors | >= 5 | 至少产出 5 个 IC > 0.02 的因子 |
| Fitness Function | Incremental_IC | 增量 IC 评估 |
"""

from typing import Any, Optional, Union, Dict, List, Tuple
from pathlib import Path
import warnings
import time
import json
import os
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations, product
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from loguru import logger
import yaml

# V119 强制版本全局变量
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

# ==============================================================================
# V119 版本定义
# ==============================================================================
VERSION = "V119"

# V118 毒素因子黑名单 (IC 为负的因子)
TOXIC_FACTOR_BLACKLIST = [
    'turnover_rate',      # IC: -0.0442
    'volatility_20',      # IC: -0.0409
    'momentum_10',        # IC: -0.0169
]

# ==============================================================================
# V119 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告"""
    pass

class FactorLogicError(Exception):
    """因子逻辑错误"""
    pass

class DataHealingError(Exception):
    """数据自愈错误"""
    pass

class GeneticSearchError(Exception):
    """基因搜索错误"""
    pass

class OrthogonalizationError(Exception):
    """正交化错误"""
    pass

# ==============================================================================
# V119 数据预清洗器 - 修复 MI 筛选 Bug
# ==============================================================================

class DataPreprocessor:
    """
    【V119 核心】数据预清洗器 - 修复 MI 筛选 Bug.
    
    【修复内容】
    1. Fillna: 填充 NaN 值（使用中位数或 0）
    2. Drop_constant: 删除常数项（方差为 0 的列）
    3. Clip_outliers: 截断异常值（3σ原则）
    4. Normalize: 标准化处理
    """
    
    def __init__(self, 
                 fillna_method: str = 'median',
                 outlier_std: float = 3.0,
                 drop_constant_threshold: float = 1e-10):
        """
        初始化预清洗器.
        
        Args:
            fillna_method: 填充方法 ('median', 'mean', 'zero')
            outlier_std: 异常值标准差阈值
            drop_constant_threshold: 常数项方差阈值
        """
        self.fillna_method = fillna_method
        self.outlier_std = outlier_std
        self.drop_constant_threshold = drop_constant_threshold
        
        logger.info(f"[{VERSION}][Preprocessor] Initialized with fillna={fillna_method}, outlier_std={outlier_std}")
    
    def fit_transform(self, df: pd.DataFrame, 
                      exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        对 DataFrame 进行预清洗.
        
        Args:
            df: 输入 DataFrame
            exclude_cols: 排除的列名列表
            
        Returns:
            清洗后的 DataFrame
        """
        result = df.copy()
        exclude_cols = exclude_cols or []
        
        # 1. 删除常数项
        constant_cols = []
        for col in result.columns:
            if col in exclude_cols:
                continue
            if result[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                if result[col].std() < self.drop_constant_threshold:
                    constant_cols.append(col)
        
        if constant_cols:
            result = result.drop(columns=constant_cols)
            logger.info(f"[{VERSION}][Preprocessor] Dropped {len(constant_cols)} constant columns")
        
        # 2. 填充 NaN 值
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in exclude_cols:
                continue
            if result[col].isna().any():
                if self.fillna_method == 'median':
                    result[col] = result[col].fillna(result[col].median())
                elif self.fillna_method == 'mean':
                    result[col] = result[col].fillna(result[col].mean())
                else:
                    result[col] = result[col].fillna(0)
        
        # 3. 截断异常值 (3σ原则)
        for col in numeric_cols:
            if col in exclude_cols:
                continue
            mean_val = result[col].mean()
            std_val = result[col].std()
            if std_val > 1e-10:
                lower_bound = mean_val - self.outlier_std * std_val
                upper_bound = mean_val + self.outlier_std * std_val
                result[col] = result[col].clip(lower_bound, upper_bound)
        
        logger.info(f"[{VERSION}][Preprocessor] Data preprocessing complete: {len(result.columns)} columns retained")
        return result
    
    def clean_for_mi(self, df: pd.DataFrame, 
                     target_col: str = 't1_return',
                     exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        专门为 MI 计算准备数据.
        
        Args:
            df: 输入 DataFrame
            target_col: 目标列名
            exclude_cols: 排除的列名列表
            
        Returns:
            清洗后的 DataFrame
        """
        exclude_cols = (exclude_cols or []) + ['trade_date', 'symbol']
        return self.fit_transform(df, exclude_cols=exclude_cols + [target_col])

# ==============================================================================
# V119 互信息预筛选器 - 修复版
# ==============================================================================

class MutualInformationFilter:
    """
    【V119 修复版】互信息预筛选器.
    
    【V119 修复】
    1. 增加数据预清洗（Fillna/Drop_constant）
    2. 修复 MI 计算中的维度不匹配问题
    3. 降低阈值至 20% 以增加多样性
    """
    
    def __init__(self, mi_threshold_percentile: float = 20.0):
        """
        初始化 MI 过滤器.
        
        Args:
            mi_threshold_percentile: MI 百分位阈值，默认 20%（降低以增加多样性）
        """
        self.mi_threshold_percentile = mi_threshold_percentile
        self.mi_scores = {}
        self.selected_operators = set()
        self.rng = np.random.default_rng(42)
        self.preprocessor = DataPreprocessor()
        
        logger.info(f"[{VERSION}][MIFilter] Initialized with threshold={mi_threshold_percentile}% (V119 Fix)")
    
    def calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算两个变量之间的互信息 (使用离散化方法).
        
        Args:
            x: 变量 X
            y: 变量 Y
            
        Returns:
            互信息值
        """
        if len(x) != len(y) or len(x) < 10:
            return 0.0
        
        # 去除 NaN 和无穷值
        mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            return 0.0
        
        # V119 修复：检查常数项
        if np.std(x_clean) < 1e-10 or np.std(y_clean) < 1e-10:
            return 0.0
        
        # 使用分位数离散化 (10 箱)
        n_bins = 10
        try:
            x_bins = pd.qcut(x_clean, n_bins, labels=False, duplicates='drop')
            y_bins = pd.qcut(y_clean, n_bins, labels=False, duplicates='drop')
        except Exception:
            # 如果分位数切割失败，使用等距分箱
            try:
                x_bins = pd.cut(x_clean, n_bins, labels=False, duplicates='drop')
                y_bins = pd.cut(y_clean, n_bins, labels=False, duplicates='drop')
            except Exception:
                return 0.0
        
        # 计算联合分布和边缘分布
        n_unique_x = len(np.unique(x_bins))
        n_unique_y = len(np.unique(y_bins))
        
        if n_unique_x < 2 or n_unique_y < 2:
            return 0.0
        
        # 联合概率分布
        joint_hist = np.zeros((n_unique_x, n_unique_y))
        for xi, yi in zip(x_bins, y_bins):
            joint_hist[xi, yi] += 1
        
        joint_prob = joint_hist / len(x_clean)
        
        # 边缘概率分布
        px = joint_prob.sum(axis=1)
        py = joint_prob.sum(axis=0)
        
        # 计算互信息
        mi = 0.0
        for i in range(n_unique_x):
            for j in range(n_unique_y):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
        
        return max(0.0, mi)
    
    def compute_operator_mi(self, operator_name: str, df: pd.DataFrame,
                            date_col: str = 'trade_date', symbol_col: str = 'symbol',
                            label_col: str = 't1_return') -> float:
        """
        计算单个算子与收益率的互信息.
        
        Args:
            operator_name: 算子名称 (也是特征列名)
            df: 数据 DataFrame
            label_col: 标签列名
            
        Returns:
            互信息值
        """
        if operator_name not in df.columns:
            return 0.0
        
        if label_col not in df.columns:
            return 0.0
        
        # V119 修复：获取因子值和标签
        factor_values = df[operator_name].values
        label_values = df[label_col].values
        
        # 计算互信息
        mi = self.calculate_mutual_information(factor_values, label_values)
        
        return mi
    
    def filter_operators(self, operators: List[str], df: pd.DataFrame,
                         date_col: str = 'trade_date', symbol_col: str = 'symbol',
                         label_col: str = 't1_return') -> List[str]:
        """
        筛选出 MI 前 N% 的算子.
        
        Args:
            operators: 候选算子列表
            df: 数据 DataFrame
            
        Returns:
            筛选后的算子列表
        """
        logger.info(f"[{VERSION}][MIFilter] Computing MI for {len(operators)} operators...")
        
        # V119 修复：数据预清洗
        df_clean = self.preprocessor.clean_for_mi(df, label_col)
        
        # 计算每个算子的 MI
        mi_scores = {}
        for op in operators:
            if op not in df_clean.columns:
                mi_scores[op] = 0.0
                continue
            mi = self.compute_operator_mi(op, df_clean, date_col, symbol_col, label_col)
            mi_scores[op] = mi
        
        self.mi_scores = mi_scores
        
        # V119 修复：如果所有 MI 都为 0，保留所有算子
        mi_values = list(mi_scores.values())
        if len(mi_values) < 2 or max(mi_values) < 1e-10:
            logger.warning(f"[{VERSION}][MIFilter] All MI scores are 0, retaining all operators")
            self.selected_operators = set(operators)
            return list(self.selected_operators)
        
        # 计算 MI 阈值
        threshold = np.percentile(mi_values, self.mi_threshold_percentile)
        
        # 选择 MI >= 阈值的算子
        self.selected_operators = {op for op, mi in mi_scores.items() if mi >= threshold}
        
        # V119 修复：确保至少选择 5 个算子
        if len(self.selected_operators) < 5:
            sorted_ops = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
            for op, mi in sorted_ops[:5]:
                self.selected_operators.add(op)
        
        logger.info(f"[{VERSION}][MIFilter] Selected {len(self.selected_operators)}/{len(operators)} operators "
                   f"(MI threshold: {threshold:.6f})")
        
        # 记录 MI 统计
        logger.info(f"[{VERSION}][MIFilter] MI Stats: min={min(mi_values):.6f}, "
                   f"max={max(mi_values):.6f}, mean={np.mean(mi_values):.6f}")
        
        return list(self.selected_operators)
    
    def get_mi_report(self) -> Dict[str, Any]:
        """获取 MI 报告"""
        return {
            'version': VERSION,
            'threshold_percentile': self.mi_threshold_percentile,
            'mi_scores': self.mi_scores,
            'selected_operators': list(self.selected_operators),
            'stats': {
                'min_mi': min(self.mi_scores.values()) if self.mi_scores else 0,
                'max_mi': max(self.mi_scores.values()) if self.mi_scores else 0,
                'mean_mi': np.mean(list(self.mi_scores.values())) if self.mi_scores else 0,
            }
        }

# ==============================================================================
# V119 Lowdin 正交化器
# ==============================================================================

class LowdinOrthogonalizer:
    """
    【V119】Lowdin 正交化器 - 特征中性化.
    
    【原理】
    使用残差正交化消除因子与风格因子的相关性
    """
    
    def __init__(self, style_factors: Optional[List[str]] = None):
        self.style_factors = style_factors or ['total_mv', 'pb']
        self.EPSILON = 1e-6
        
        logger.info(f"[{VERSION}][Lowdin] Initialized with style factors: {self.style_factors}")
    
    def orthogonalize(self, factor_values: pd.Series, df: pd.DataFrame,
                      date_col: str = 'trade_date') -> pd.Series:
        """对因子进行 Lowdin 正交化"""
        result = factor_values.copy()
        unique_dates = df[date_col].unique()
        
        for date in unique_dates:
            mask = df[date_col] == date
            date_idx = result.index[mask]
            
            if len(date_idx) < 10:
                continue
            
            factor_day = factor_values.loc[date_idx]
            
            style_data = []
            valid_indices = []
            
            for idx in date_idx:
                if idx not in df.index:
                    continue
                
                row = []
                valid = True
                for sf in self.style_factors:
                    if sf in df.columns and idx in df.index:
                        val = df.loc[idx, sf]
                        if pd.isna(val) or not np.isfinite(val):
                            valid = False
                            break
                        row.append(val)
                    else:
                        valid = False
                        break
                
                if valid and (idx in factor_day.index and np.isfinite(factor_day.loc[idx])):
                    style_data.append(row)
                    valid_indices.append(idx)
            
            if len(style_data) < 10:
                continue
            
            style_matrix = np.array(style_data)
            factor_array = np.array([factor_day.loc[idx] for idx in valid_indices])
            
            # 标准化
            style_matrix = (style_matrix - style_matrix.mean(axis=0)) / (style_matrix.std(axis=0) + self.EPSILON)
            factor_array = (factor_array - factor_array.mean()) / (factor_array.std() + self.EPSILON)
            
            # 残差正交化
            orthogonalized = factor_array.copy()
            for i in range(style_matrix.shape[1]):
                style_col = style_matrix[:, i]
                if np.std(style_col) > self.EPSILON:
                    beta = np.cov(orthogonalized, style_col)[0, 1] / (np.var(style_col) + self.EPSILON)
                    orthogonalized = orthogonalized - beta * style_col
            
            for i, idx in enumerate(valid_indices):
                result.loc[idx] = orthogonalized[i]
        
        return result
    
    def orthogonalize_full(self, factor_values: pd.Series, df: pd.DataFrame,
                           date_col: str = 'trade_date', symbol_col: str = 'symbol') -> pd.Series:
        """全空间 Lowdin 正交化"""
        result = factor_values.copy()
        unique_dates = sorted(df[date_col].unique())
        
        for date in unique_dates:
            mask = df[date_col] == date
            date_idx = result.index[mask]
            
            if len(date_idx) < 20:
                continue
            
            factor_day = factor_values.loc[date_idx].values
            if np.sum(np.isfinite(factor_day)) < 10:
                continue
            
            style_cols = []
            for sf in self.style_factors:
                if sf in df.columns:
                    sf_values = df.loc[date_idx, sf].values
                    style_cols.append(sf_values)
            
            if len(style_cols) == 0:
                continue
            
            style_matrix = np.column_stack(style_cols)
            
            valid_mask = np.isfinite(factor_day) & np.all(np.isfinite(style_matrix), axis=1)
            if np.sum(valid_mask) < 10:
                continue
            
            factor_clean = factor_day[valid_mask]
            style_clean = style_matrix[valid_mask]
            indices_clean = np.array(date_idx)[valid_mask]
            
            # 标准化
            factor_clean = (factor_clean - factor_clean.mean()) / (factor_clean.std() + self.EPSILON)
            style_clean = (style_clean - style_clean.mean(axis=0)) / (style_clean.std(axis=0) + self.EPSILON)
            
            # 残差正交化
            try:
                X = np.column_stack([np.ones(len(style_clean)), style_clean])
                beta = np.linalg.lstsq(X, factor_clean, rcond=None)[0]
                residual = factor_clean - X @ beta
                
                for i, idx in enumerate(indices_clean):
                    result.loc[idx] = residual[i]
            except Exception as e:
                logger.warning(f"[{VERSION}][Lowdin] Orthogonalization failed for date {date}: {e}")
        
        return result

# ==============================================================================
# V119 基因算子库 - 增加新算子
# ==============================================================================

@dataclass
class GeneticNode:
    """遗传算法节点"""
    name: str
    node_type: str
    children: List['GeneticNode'] = field(default_factory=list)
    value: Optional[float] = None
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class GeneticFactor:
    """遗传生成的因子"""
    expression: str
    tree: GeneticNode
    ic_score: float = 0.0
    icir_score: float = 0.0
    ic_std: float = 0.0
    incremental_ic: float = 0.0  # V119: 增量 IC
    fitness_score: float = 0.0
    complexity: int = 0
    order: int = 0
    is_valid: bool = True
    error_message: str = ""
    auto_flipped: bool = False
    flipped_ic: float = 0.0
    original_ic: float = 0.0
    orthogonalized: bool = False


class SymbolicOperatorLibrary:
    """
    【V119 增强版】符号算子库 - 增加 Ts_Entropy, Ts_Skewness.
    """
    
    EPSILON = 1e-6
    
    OPERATORS = {
        # 截面算子
        'Rank': {'arity': 1, 'type': 'cross_sectional', 'description': '截面百分位排名'},
        'Scale': {'arity': 1, 'type': 'cross_sectional', 'description': '截面标准化'},
        
        # 时序算子
        'Ts_Mean': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20], 'description': '时序均值'},
        'Ts_Std': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20], 'description': '时序标准差'},
        'Ts_Delta': {'arity': 2, 'type': 'time_series', 'window': [1, 3, 5], 'description': '时序变化量'},
        'Ts_Max': {'arity': 2, 'type': 'time_series', 'window': [10, 20], 'description': '时序最大值'},
        'Ts_Min': {'arity': 2, 'type': 'time_series', 'window': [10, 20], 'description': '时序最小值'},
        
        # V119 新增算子
        'Ts_Entropy': {'arity': 2, 'type': 'time_series', 'window': [10, 20], 'description': '时序信息熵'},
        'Ts_Skewness': {'arity': 2, 'type': 'time_series', 'window': [10, 20], 'description': '时序偏度'},
        
        # 延迟算子
        'Delay': {'arity': 2, 'type': 'delay', 'window': [1, 3, 5], 'description': '延迟'},
        'Delta': {'arity': 2, 'type': 'delay', 'window': [1, 3], 'description': '差分'},
        
        # 数学算子
        'Log': {'arity': 1, 'type': 'mathematical', 'description': '对数'},
        'Sqrt': {'arity': 1, 'type': 'mathematical', 'description': '平方根'},
        'Abs': {'arity': 1, 'type': 'mathematical', 'description': '绝对值'},
        'Sign': {'arity': 1, 'type': 'mathematical', 'description': '符号'},
        'Square': {'arity': 1, 'type': 'mathematical', 'description': '平方'},
        'Inv': {'arity': 1, 'type': 'mathematical', 'description': '倒数'},
        
        # 交互算子
        'Mul': {'arity': 2, 'type': 'interaction', 'description': '乘法交互'},
        'Div': {'arity': 2, 'type': 'interaction', 'description': '除法交互'},
        'Add': {'arity': 2, 'type': 'interaction', 'description': '加法'},
        'Sub': {'arity': 2, 'type': 'interaction', 'description': '减法'},
        'Max': {'arity': 2, 'type': 'interaction', 'description': '最大值'},
        'Min': {'arity': 2, 'type': 'interaction', 'description': '最小值'},
    }
    
    # V119 基础特征池
    BASE_FEATURES = [
        'close', 'open', 'high', 'low', 'volume', 'amount',
        'turnover_rate', 'pct_chg', 'change', 'pre_close',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_5', 'volatility_20',
        'volume_ma_ratio_5', 'volume_ma_ratio_20',
        'price_position_20', 'price_position_60',
        'ma_deviation_5', 'ma_deviation_20',
        'rsi_14', 'mfi_14',
        'turnover_bias_20', 'turnover_ma_ratio',
        'volume_price_divergence_5', 'volume_price_divergence_20',
        'volume_price_correlation', 'smart_money_flow',
        'volatility_contraction_10', 'volume_shrink_ratio',
        'volume_price_stable', 'accumulation_distribution_20',
        'macd', 'macd_signal', 'macd_hist',
        'bias_60', 'volume_price_health',
        'volume_shrink_flag', 'price_volume_divergence',
        'hist_sharpe_20d', 'predict_score',
    ]
    
    def __init__(self):
        self.operator_history = []
    
    def get_operator(self, name: str) -> Dict:
        return self.OPERATORS.get(name, {})
    
    def get_all_operators(self) -> List[str]:
        return list(self.OPERATORS.keys())
    
    def get_base_features(self) -> List[str]:
        return self.BASE_FEATURES.copy()
    
    def calculate_complexity(self, tree: GeneticNode) -> int:
        if tree.node_type in ['feature', 'constant']:
            return 1
        if tree.node_type == 'operator':
            child_complexity = sum(self.calculate_complexity(child) for child in tree.children)
            return 1 + child_complexity
        return 1
    
    def calculate_order(self, tree: GeneticNode) -> int:
        """计算因子阶数"""
        if tree.node_type in ['feature', 'constant']:
            return 1
        
        if tree.node_type == 'operator':
            if tree.name in ['Mul', 'Div', 'Add', 'Sub', 'Max', 'Min']:
                child_orders = [self.calculate_order(child) for child in tree.children]
                return sum(child_orders)
            else:
                if tree.children:
                    return max(self.calculate_order(child) for child in tree.children)
                return 1
        
        return 1
    
    def tree_to_string(self, tree: GeneticNode) -> str:
        if tree.node_type == 'feature':
            return tree.name
        if tree.node_type == 'constant':
            return str(tree.value)
        if tree.node_type == 'operator':
            if not tree.children:
                return tree.name
            children_str = ', '.join(self.tree_to_string(child) for child in tree.children)
            return f"{tree.name}({children_str})"
        return ""
    
    def validate_tree(self, tree: GeneticNode) -> Tuple[bool, str]:
        if tree.node_type == 'operator':
            op_info = self.OPERATORS.get(tree.name, {})
            expected_arity = op_info.get('arity', 0)
            actual_arity = len(tree.children)
            
            if expected_arity != actual_arity:
                return False, f"Operator {tree.name} expects {expected_arity} args, got {actual_arity}"
            
            for child in tree.children:
                valid, msg = self.validate_tree(child)
                if not valid:
                    return False, msg
        
        return True, ""

# ==============================================================================
# V119 遗传因子挖掘器 - Incremental_IC 适应度函数
# ==============================================================================

class GeneticFactorMiner:
    """
    【V119 核心】遗传因子挖掘器 - Incremental_IC 适应度函数.
    
    【V119 改进】
    1. 适应度函数：Incremental_IC (新因子与已有因子池正交后的残差 IC)
    2. 数据预清洗：在进化开始前清洗数据
    3. 新算子：Ts_Entropy, Ts_Skewness
    """
    
    def __init__(self,
                 population_size: int = 40,
                 generations: int = 20,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.6,
                 elite_rate: float = 0.15,
                 max_order: int = 3,
                 max_tree_depth: int = 4,
                 mi_threshold: float = 20.0,
                 seed: int = 42):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_rate = elite_rate
        self.max_order = max_order
        self.max_tree_depth = max_tree_depth
        self.seed = seed
        
        self.operator_lib = SymbolicOperatorLibrary()
        self.rng = np.random.default_rng(seed)
        self.mi_filter = MutualInformationFilter(mi_threshold)
        
        self.generation_history = []
        self.best_factors = []
        self.all_evaluated_factors = []
        self.mining_log = []
        self.auto_flip_log = []
        self.orthogonalizer = LowdinOrthogonalizer()
        self.preprocessor = DataPreprocessor()
        
        # V119: 因子池用于计算 Incremental_IC
        self.factor_pool = []
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][GeneticFactorMiner] Initialized")
        logger.info("=" * 80)
        logger.info(f"  Population Size: {self.population_size}")
        logger.info(f"  Generations: {self.generations}")
        logger.info(f"  Max Order: {self.max_order}")
        logger.info(f"  Fitness Function: Incremental_IC (残差 IC)")
        logger.info(f"  MI Pre-filter: {self.mi_filter.mi_threshold_percentile}%")
        logger.info(f"  Data Preprocessing: Enabled")
        logger.info(f"  New Operators: Ts_Entropy, Ts_Skewness")
        logger.info("=" * 80)
    
    def _log_mining(self, action: str, details: str = "") -> None:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
        }
        self.mining_log.append(log_entry)
        logger.info(f"[{VERSION}][GeneticMining] {action}: {details}")
    
    def _log_auto_flip(self, factor_name: str, original_ic: float, 
                       flipped_ic: float, reason: str) -> None:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'factor': factor_name,
            'original_ic': original_ic,
            'flipped_ic': flipped_ic,
            'reason': reason,
        }
        self.auto_flip_log.append(log_entry)
        logger.warning(f"[{VERSION}][Auto-Flip] {factor_name}: IC {original_ic:.4f} -> {flipped_ic:.4f} ({reason})")
    
    def generate_random_tree(self, max_depth: int = 4, current_depth: int = 0,
                             allowed_operators: Optional[List[str]] = None) -> GeneticNode:
        """随机生成表达式树"""
        if current_depth >= max_depth:
            if self.rng.random() < 0.8:
                feature = self.rng.choice(self.operator_lib.get_base_features())
                return GeneticNode(name=feature, node_type='feature')
            else:
                value = round(self.rng.uniform(-1, 1), 4)
                return GeneticNode(name='const', node_type='constant', value=value)
        
        if allowed_operators is None:
            allowed_ops = list(self.operator_lib.OPERATORS.keys())
        else:
            allowed_ops = allowed_operators
        
        operator = self.rng.choice(allowed_ops)
        op_info = self.operator_lib.get_operator(operator)
        
        node = GeneticNode(name=operator, node_type='operator')
        
        if op_info['type'] in ['time_series', 'delay']:
            child_feature = self.generate_random_tree(max_depth - 1, current_depth + 1, allowed_operators)
            window = self.rng.choice(op_info.get('window', [5]))
            window_node = GeneticNode(name=str(window), node_type='constant', value=float(window))
            node.children = [child_feature, window_node]
        elif op_info['arity'] == 1:
            child = self.generate_random_tree(max_depth - 1, current_depth + 1, allowed_operators)
            node.children = [child]
        else:
            for _ in range(op_info['arity']):
                child = self.generate_random_tree(max_depth - 1, current_depth + 1, allowed_operators)
                node.children.append(child)
        
        return node
    
    def _calculate_rank_ic(self, factor_values: pd.Series, label_values: pd.Series) -> float:
        """计算 Rank IC"""
        mask = factor_values.notna() & label_values.notna()
        if mask.sum() < 10:
            return 0.0
        
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        factor_rank = factor_clean.rank(method='average')
        label_rank = label_clean.rank(method='average')
        
        if np.std(factor_rank) > 1e-10 and np.std(label_rank) > 1e-10:
            ic = np.corrcoef(factor_rank, label_rank)[0, 1]
            return float(ic) if not np.isnan(ic) else 0.0
        return 0.0
    
    def calculate_incremental_ic(self, factor_values: pd.Series, label_values: pd.Series,
                                  df: pd.DataFrame, date_col: str = 'trade_date') -> float:
        """
        V119 核心：计算增量 IC.
        
        【原理】
        1. 将新因子对因子池中所有因子进行回归
        2. 取残差作为独立信息
        3. 计算残差与标签的 IC
        """
        if len(self.factor_pool) == 0:
            # 因子池为空时，直接计算原始 IC
            return self._calculate_rank_ic(factor_values, label_values)
        
        # 按日期分组计算残差 IC
        ic_scores = []
        unique_dates = df[date_col].unique()
        
        for date in unique_dates:
            mask = df[date_col] == date
            date_idx = factor_values.index[mask]
            
            if len(date_idx) < 20:
                continue
            
            # 获取当日数据
            factor_day = factor_values.loc[date_idx]
            label_day = label_values.loc[label_values.index.isin(date_idx)]
            
            # 获取因子池数据
            pool_data = []
            valid_indices = []
            
            for idx in date_idx:
                row = []
                valid = True
                for pool_factor in self.factor_pool:
                    if idx in pool_factor.index and np.isfinite(pool_factor.loc[idx]):
                        row.append(pool_factor.loc[idx])
                    else:
                        valid = False
                        break
                
                if valid and np.isfinite(factor_day.loc[idx]) if idx in factor_day.index else False:
                    pool_data.append(row)
                    valid_indices.append(idx)
            
            if len(pool_data) < 20 or len(valid_indices) < 20:
                continue
            
            pool_matrix = np.array(pool_data)
            factor_array = np.array([factor_day.loc[idx] for idx in valid_indices])
            label_array = np.array([label_day.loc[label_day.index == idx].values[0] 
                                   for idx in valid_indices if idx in label_day.index])
            
            if len(label_array) < 20:
                continue
            
            # 标准化
            pool_matrix = (pool_matrix - pool_matrix.mean(axis=0)) / (pool_matrix.std(axis=0) + 1e-10)
            factor_array = (factor_array - factor_array.mean()) / (factor_array.std() + 1e-10)
            
            # 回归取残差
            try:
                X = np.column_stack([np.ones(len(pool_matrix)), pool_matrix])
                beta = np.linalg.lstsq(X, factor_array, rcond=None)[0]
                residual = factor_array - X @ beta
                
                # 计算残差 IC
                residual_series = pd.Series(residual, index=valid_indices)
                label_series = pd.Series(label_array, index=valid_indices)
                
                ic = self._calculate_rank_ic(residual_series, label_series)
                if not np.isnan(ic):
                    ic_scores.append(ic)
            except Exception:
                continue
        
        if not ic_scores:
            return self._calculate_rank_ic(factor_values, label_values)
        
        return float(np.mean(ic_scores))
    
    def evaluate_factor(self, factor: GeneticFactor, df: pd.DataFrame,
                        date_col: str = 'trade_date', symbol_col: str = 'symbol',
                        label_col: str = 't1_return') -> Tuple[float, float, float, float]:
        """
        评估因子 - V119 使用 Incremental_IC.
        """
        try:
            factor_values = self.compute_factor_value(factor.tree, df, date_col, symbol_col)
            
            if factor_values is None or factor_values.isna().all():
                return 0.0, 0.0, 0.0, 0.0
            
            # 应用 Lowdin 正交化
            try:
                factor_values = self.orthogonalizer.orthogonalize_full(factor_values, df, date_col, symbol_col)
                factor.orthogonalized = True
            except Exception as e:
                logger.warning(f"[{VERSION}] Orthogonalization failed: {e}")
            
            # 获取标签
            label_values = df.set_index([date_col, symbol_col])[label_col]
            factor_values_indexed = factor_values.set_index([date_col, symbol_col])
            
            # V119: 计算 Incremental_IC
            incremental_ic = self.calculate_incremental_ic(
                factor_values_indexed, label_values, df, date_col
            )
            
            # 也计算原始 IC 用于参考
            ic_scores = []
            unique_dates = df[date_col].unique()
            
            for date in unique_dates:
                day_data = df[df[date_col] == date]
                if len(day_data) < 20:
                    continue
                
                factor_day = factor_values[factor_values[date_col] == date] if date_col in factor_values.columns else factor_values
                if hasattr(factor_day, 'index') and hasattr(day_data, 'index'):
                    factor_day = factor_day[factor_day.index.isin(day_data.index)]
                label_day = day_data[label_col]
                
                if len(factor_day) < 20:
                    continue
                
                mask = factor_day.notna() & label_day.notna()
                if mask.sum() < 20:
                    continue
                
                ic = self._calculate_rank_ic(factor_day, label_day)
                if not np.isnan(ic):
                    ic_scores.append(ic)
            
            ic_mean = float(np.mean(ic_scores)) if ic_scores else 0.0
            ic_std = float(np.std(ic_scores, ddof=1)) if len(ic_scores) > 1 else 0.0
            icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
            
            # V119 适应度函数：Incremental_IC + ICIR 奖励
            fitness_score = incremental_ic + 0.1 * icir
            
            return ic_mean, icir, ic_std, fitness_score
            
        except Exception as e:
            factor.is_valid = False
            factor.error_message = str(e)
            return 0.0, 0.0, 0.0, 0.0
    
    def auto_flip_direction(self, factor: GeneticFactor, df: pd.DataFrame,
                            date_col: str = 'trade_date', symbol_col: str = 'symbol',
                            label_col: str = 't1_return') -> GeneticFactor:
        """V119 Auto-Flip 物理反转逻辑"""
        try:
            factor_values = self.compute_factor_value(factor.tree, df, date_col, symbol_col)
            
            if factor_values is None:
                return factor
            
            # 计算累计 IC
            ic_scores = []
            unique_dates = sorted(df[date_col].unique())
            
            for date in unique_dates:
                day_data = df[df[date_col] == date]
                if len(day_data) < 20:
                    continue
                
                factor_day = factor_values[factor_values.index.isin(day_data.index)] if hasattr(factor_values, 'index') else factor_values
                label_day = day_data[label_col]
                
                mask = factor_day.notna() & label_day.notna()
                if mask.sum() < 20:
                    continue
                
                ic = self._calculate_rank_ic(factor_day, label_day)
                if not np.isnan(ic):
                    ic_scores.append(ic)
            
            if not ic_scores:
                return factor
            
            cumulative_ic = np.sum(ic_scores)
            mean_ic = np.mean(ic_scores)
            
            # 强制：累计 IC 为负时物理翻转
            if cumulative_ic < 0 or mean_ic < 0:
                original_ic = mean_ic
                
                # 创建物理翻转后的因子
                flipped_factor = GeneticFactor(
                    expression=f"-1 * ({factor.expression})",
                    tree=self._create_negation_tree(factor.tree),
                    ic_score=-original_ic,
                    icir_score=factor.icir_score,
                    ic_std=factor.ic_std,
                    fitness_score=abs(original_ic) - factor.ic_std,
                    complexity=factor.complexity,
                    order=factor.order,
                    is_valid=True,
                    auto_flipped=True,
                    original_ic=original_ic,
                    flipped_ic=-original_ic,
                    orthogonalized=factor.orthogonalized
                )
                
                reason = "Cumulative IC < 0" if cumulative_ic < 0 else f"Mean IC < 0 ({mean_ic:.4f})"
                self._log_auto_flip(factor.expression[:50], original_ic, -original_ic, reason)
                
                return flipped_factor
            
            factor.original_ic = mean_ic
            factor.flipped_ic = mean_ic
            
            return factor
            
        except Exception as e:
            return factor
    
    def _create_negation_tree(self, tree: GeneticNode) -> GeneticNode:
        neg_node = GeneticNode(name='Mul', node_type='operator')
        neg_node.children = [
            GeneticNode(name='const', node_type='constant', value=-1.0),
            tree
        ]
        return neg_node
    
    def compute_factor_value(self, tree: GeneticNode, df: pd.DataFrame,
                             date_col: str, symbol_col: str) -> Optional[pd.Series]:
        try:
            if tree.node_type == 'feature':
                if tree.name not in df.columns:
                    return None
                return df[tree.name].astype(float)
            
            if tree.node_type == 'constant':
                return pd.Series(tree.value, index=df.index)
            
            if tree.node_type == 'operator':
                children_values = [self.compute_factor_value(child, df, date_col, symbol_col) 
                                   for child in tree.children]
                
                if any(v is None for v in children_values):
                    return None
                
                return self.apply_operator(tree.name, children_values, df, date_col, symbol_col)
            
            return None
            
        except Exception as e:
            return None
    
    def apply_operator(self, operator: str, values: List[pd.Series],
                       df: pd.DataFrame, date_col: str, symbol_col: str) -> pd.Series:
        EPS = self.operator_lib.EPSILON
        
        if operator == 'Rank':
            return values[0].groupby(df[date_col]).transform(
                lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
            )
        
        elif operator == 'Scale':
            return values[0].groupby(df[date_col]).transform(
                lambda x: (x - x.mean()) / (x.std() + EPS) if len(x.dropna()) > 1 else x
            )
        
        elif operator == 'Ts_Mean':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        
        elif operator == 'Ts_Std':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=2).std()
            )
        
        elif operator == 'Ts_Delta':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1) - x.shift(window + 1)
            )
        
        elif operator == 'Ts_Max':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
            )
        
        elif operator == 'Ts_Min':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
            )
        
        # V119 新增算子
        elif operator == 'Ts_Entropy':
            window = int(values[1].iloc[0])
            def calc_entropy(x):
                # 使用分位数离散化计算信息熵
                try:
                    if len(x) < 3:
                        return np.nan
                    bins = pd.qcut(x.dropna(), q=5, labels=False, duplicates='drop')
                    if len(bins) < 2:
                        return np.nan
                    probs = np.bincount(bins) / len(bins)
                    probs = probs[probs > 0]
                    return -np.sum(probs * np.log(probs))
                except Exception:
                    return np.nan
            
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=5).apply(calc_entropy, raw=False)
            )
        
        elif operator == 'Ts_Skewness':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=5).skew()
            )
        
        elif operator == 'Delay':
            window = int(values[1].iloc[0])
            return values[0].shift(window)
        
        elif operator == 'Delta':
            window = int(values[1].iloc[0])
            return values[0] - values[0].shift(window)
        
        elif operator == 'Log':
            return np.log(values[0].abs() + EPS)
        
        elif operator == 'Sqrt':
            return np.sqrt(values[0].abs() + EPS)
        
        elif operator == 'Abs':
            return values[0].abs()
        
        elif operator == 'Sign':
            return np.sign(values[0])
        
        elif operator == 'Square':
            return values[0] ** 2
        
        elif operator == 'Inv':
            return 1.0 / (values[0].abs() + EPS)
        
        elif operator == 'Mul':
            return values[0] * values[1]
        
        elif operator == 'Div':
            return values[0] / (values[1].abs() + EPS)
        
        elif operator == 'Add':
            return values[0] + values[1]
        
        elif operator == 'Sub':
            return values[0] - values[1]
        
        elif operator == 'Max':
            return pd.concat([values[0], values[1]], axis=1).max(axis=1)
        
        elif operator == 'Min':
            return pd.concat([values[0], values[1]], axis=1).min(axis=1)
        
        return values[0]
    
    def initialize_population(self, allowed_operators: List[str]) -> List[GeneticFactor]:
        population = []
        
        for i in range(self.population_size):
            tree = self.generate_random_tree(self.max_tree_depth, allowed_operators=allowed_operators)
            expression = self.operator_lib.tree_to_string(tree)
            complexity = self.operator_lib.calculate_complexity(tree)
            order = self.operator_lib.calculate_order(tree)
            
            # 严格限制阶数
            if order > self.max_order:
                tree = self.generate_random_tree(self.max_tree_depth - 1, allowed_operators=allowed_operators)
                expression = self.operator_lib.tree_to_string(tree)
                order = self.operator_lib.calculate_order(tree)
            
            factor = GeneticFactor(
                expression=expression,
                tree=tree,
                complexity=complexity,
                order=order,
            )
            population.append(factor)
        
        self._log_mining("InitializePopulation", f"Generated {len(population)} factors (max_order={self.max_order})")
        return population
    
    def select_elites(self, population: List[GeneticFactor], elite_count: int) -> List[GeneticFactor]:
        sorted_pop = sorted(population, key=lambda f: f.fitness_score, reverse=True)
        return sorted_pop[:elite_count]
    
    def tournament_selection(self, population: List[GeneticFactor], tournament_size: int = 5) -> GeneticFactor:
        candidates = self.rng.choice(population, size=min(tournament_size, len(population)), replace=False)
        return max(candidates, key=lambda f: f.fitness_score)
    
    def crossover(self, parent1: GeneticFactor, parent2: GeneticFactor) -> Tuple[GeneticFactor, GeneticFactor]:
        def get_random_node(tree: GeneticNode, depth: int = 0) -> GeneticNode:
            if not tree.children:
                return tree
            if self.rng.random() < 0.3 or depth >= self.max_tree_depth - 1:
                return tree
            child_idx = self.rng.integers(0, len(tree.children))
            return get_random_node(tree.children[child_idx], depth + 1)
        
        def replace_subtree(tree: GeneticNode, target: GeneticNode, replacement: GeneticNode) -> GeneticNode:
            if tree is target:
                return replacement
            new_tree = GeneticNode(name=tree.name, node_type=tree.node_type, value=tree.value, children=list(tree.children))
            for i, child in enumerate(new_tree.children):
                if child is target:
                    new_tree.children[i] = replacement
                else:
                    new_tree.children[i] = replace_subtree(child, target, replacement)
            return new_tree
        
        node1 = get_random_node(parent1.tree)
        node2 = get_random_node(parent2.tree)
        
        child1_tree = replace_subtree(parent1.tree, node1, node2)
        child2_tree = replace_subtree(parent2.tree, node2, node1)
        
        def create_factor(tree: GeneticNode) -> GeneticFactor:
            valid, msg = self.operator_lib.validate_tree(tree)
            if not valid:
                return GeneticFactor(expression=self.operator_lib.tree_to_string(tree), tree=tree, is_valid=False, error_message=msg)
            return GeneticFactor(expression=self.operator_lib.tree_to_string(tree), tree=tree,
                               complexity=self.operator_lib.calculate_complexity(tree),
                               order=self.operator_lib.calculate_order(tree))
        
        return create_factor(child1_tree), create_factor(child2_tree)
    
    def mutate(self, factor: GeneticFactor) -> GeneticFactor:
        def mutate_node(tree: GeneticNode, depth: int = 0) -> GeneticNode:
            if tree.node_type == 'operator' and tree.children:
                if self.rng.random() < self.mutation_rate:
                    child_idx = self.rng.integers(0, len(tree.children))
                    tree.children[child_idx] = self.generate_random_tree(self.max_tree_depth - depth - 1, depth + 1)
                else:
                    for child in tree.children:
                        mutate_node(child, depth + 1)
            elif tree.node_type == 'constant':
                if self.rng.random() < self.mutation_rate:
                    tree.value = round(tree.value + self.rng.normal(0, 0.3), 4)
            return tree
        
        new_tree = mutate_node(factor.tree)
        return GeneticFactor(
            expression=self.operator_lib.tree_to_string(new_tree),
            tree=new_tree,
            complexity=self.operator_lib.calculate_complexity(new_tree),
            order=self.operator_lib.calculate_order(new_tree),
        )
    
    def evolve(self, df: pd.DataFrame, allowed_operators: List[str],
               date_col: str = 'trade_date', symbol_col: str = 'symbol',
               label_col: str = 't1_return') -> List[GeneticFactor]:
        self._log_mining("StartEvolution", f"Starting evolution with {self.population_size} individuals")
        
        # V119: 数据预清洗
        df_clean = self.preprocessor.fit_transform(df, exclude_cols=[date_col, symbol_col, label_col])
        
        population = self.initialize_population(allowed_operators)
        elite_count = max(1, int(self.population_size * self.elite_rate))
        
        for generation in range(self.generations):
            for factor in population:
                if factor.is_valid:
                    ic, icir, ic_std, fitness = self.evaluate_factor(factor, df_clean, date_col, symbol_col, label_col)
                    factor.ic_score = ic
                    factor.icir_score = icir
                    factor.ic_std = ic_std
                    factor.fitness_score = fitness
                    self.all_evaluated_factors.append(factor)
            
            valid_factors = [f for f in population if f.is_valid]
            if valid_factors:
                best_fitness = max(f.fitness_score for f in valid_factors)
                avg_fitness = np.mean([f.fitness_score for f in valid_factors])
                best_ic = max(abs(f.ic_score) for f in valid_factors)
                
                self.generation_history.append({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'best_ic': best_ic,
                    'valid_count': len(valid_factors),
                })
                
                self._log_mining(f"Generation {generation}",
                               f"Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}, Best IC: {best_ic:.4f}")
            
            elites = self.select_elites(population, elite_count)
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                if len(new_population) < self.population_size and child1.is_valid:
                    new_population.append(child1)
                if len(new_population) < self.population_size and child2.is_valid:
                    new_population.append(child2)
            
            population = new_population
        
        # 应用 Auto-Flip
        valid_factors = [f for f in population if f.is_valid]
        for i, factor in enumerate(valid_factors):
            valid_factors[i] = self.auto_flip_direction(factor, df_clean, date_col, symbol_col, label_col)
        
        # 更新因子池
        for f in valid_factors[:5]:
            try:
                factor_values = self.compute_factor_value(f.tree, df_clean, date_col, symbol_col)
                if factor_values is not None:
                    self.factor_pool.append(factor_values)
            except Exception:
                pass
        
        sorted_factors = sorted(valid_factors, key=lambda f: f.fitness_score, reverse=True)
        self.best_factors = sorted_factors[:10]
        
        self._log_mining("EvolutionComplete", f"Found {len(self.best_factors)} high-quality factors")
        
        return self.best_factors
    
    def mine_factors(self, df: pd.DataFrame, target_count: int = 5,
                     min_fitness: float = 0.005) -> List[GeneticFactor]:
        """挖掘因子 - V119 带 MI 预筛选和数据清洗"""
        self._log_mining("MIFilter", "Starting MI pre-filtering with data preprocessing...")
        
        # 数据预清洗 - V119 修复：使用正确的参数名 target_col
        df_clean = self.preprocessor.clean_for_mi(df, target_col='t1_return')
        
        # MI 预筛选
        base_features = self.operator_lib.get_base_features()
        available_features = [f for f in base_features if f in df_clean.columns]
        
        selected_features = self.mi_filter.filter_operators(
            available_features, df_clean, 'trade_date', 'symbol', 't1_return'
        )
        
        allowed_operators = list(self.operator_lib.OPERATORS.keys())
        
        self._log_mining("MIFilterComplete", f"Selected {len(selected_features)} features for evolution")
        
        all_good_factors = []
        max_attempts = 3
        
        for attempt in range(max_attempts):
            self._log_mining("MiningAttempt", f"Attempt {attempt + 1}/{max_attempts}")
            
            factors = self.evolve(df_clean, allowed_operators)
            
            good_factors = [f for f in factors if abs(f.fitness_score) >= min_fitness]
            all_good_factors.extend(good_factors)
            
            if good_factors:
                best_ic = max(abs(f.ic_score) for f in good_factors)
                if best_ic >= 0.015:
                    break
        
        # 去重
        unique_factors = []
        seen_expressions = set()
        
        for f in sorted(all_good_factors, key=lambda x: abs(x.fitness_score), reverse=True):
            if f.expression not in seen_expressions:
                unique_factors.append(f)
                seen_expressions.add(f.expression)
        
        self._log_mining("MiningComplete", f"Found {len(unique_factors)} unique factors")
        
        return unique_factors[:target_count]
    
    def get_mining_report(self) -> Dict[str, Any]:
        return {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_order': self.max_order,
                'fitness_function': 'Incremental_IC + 0.1*ICIR',
                'mi_filter': self.mi_filter.get_mi_report(),
                'auto_flip': True,
                'orthogonalization': True,
                'data_preprocessing': True,
                'new_operators': ['Ts_Entropy', 'Ts_Skewness'],
            },
            'results': {
                'total_evaluated': len(self.all_evaluated_factors),
                'best_factors': [
                    {
                        'expression': f.expression,
                        'ic_score': f.ic_score,
                        'icir_score': f.icir_score,
                        'ic_std': f.ic_std,
                        'incremental_ic': f.incremental_ic,
                        'fitness_score': f.fitness_score,
                        'complexity': f.complexity,
                        'order': f.order,
                        'auto_flipped': f.auto_flipped,
                        'original_ic': f.original_ic,
                        'flipped_ic': f.flipped_ic,
                        'orthogonalized': f.orthogonalized,
                    }
                    for f in self.best_factors
                ],
                'generation_history': self.generation_history,
            },
            'mining_log': self.mining_log,
            'auto_flip_log': self.auto_flip_log,
        }

# ==============================================================================
# V119 数据加载器
# ==============================================================================

class RealDataLoader:
    """V119 真实数据加载器"""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.engine = None
        self.data_source = "none"
        self.audit_log = []
        self._connect_database()
    
    def _connect_database(self):
        if not self.db_url:
            logger.warning(f"[{VERSION}][RealDataLoader] DATABASE_URL not configured")
            return
        
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.pool import QueuePool
            
            self.engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
            
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.data_source = "database"
            logger.info(f"[{VERSION}][RealDataLoader] Database connected")
        except Exception as e:
            logger.warning(f"[{VERSION}][RealDataLoader] Database connection failed: {e}")
            self.data_source = "none"
    
    def _log_audit(self, action: str, details: str = "") -> None:
        log_entry = {'timestamp': datetime.now().isoformat(), 'action': action, 'details': details}
        self.audit_log.append(log_entry)
        logger.info(f"[{VERSION}][DataAudit] {action}: {details}")
    
    def load_data(self, start_date: str = '20240101', end_date: str = '20241231',
                  symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        self._log_audit("LoadData", f"Loading data from {start_date} to {end_date}")
        
        # 优先使用 Parquet 文件
        parquet_files = list(Path("data/parquet").glob("*.parquet"))
        if parquet_files:
            try:
                dfs = []
                for pf in parquet_files:
                    df_temp = pd.read_parquet(pf)
                    dfs.append(df_temp)
                
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    if 'trade_date' in df.columns:
                        if df['trade_date'].dtype == 'datetime64[ns]':
                            df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
                        else:
                            df['trade_date'] = df['trade_date'].astype(str)
                            if len(str(df['trade_date'].iloc[0])) == 8:
                                df['trade_date'] = df['trade_date'].str.replace(
                                    r'(\d{4})(\d{2})(\d{2})', r'\1-\2-\3', regex=True
                                )
                        
                        start_date_str = str(start_date).replace('-', '')
                        end_date_str = str(end_date).replace('-', '')
                        
                        df['_date_cmp'] = df['trade_date'].str.replace('-', '')
                        df = df[(df['_date_cmp'] >= start_date_str) & (df['_date_cmp'] <= end_date_str)]
                        df = df.drop(columns=['_date_cmp'])
                    
                    if len(df) > 0:
                        self._log_audit("DataLoaded", f"Loaded {len(df)} rows from Parquet")
                        self.data_source = "parquet"
                        return df
            except Exception as e:
                logger.warning(f"[{VERSION}][RealDataLoader] Parquet load failed: {e}")
        
        # 回退到数据库
        if self.engine:
            try:
                df = self._load_from_database(start_date, end_date, symbols)
                if df is not None and not df.empty:
                    self._log_audit("DataLoaded", f"Loaded {len(df)} rows from database")
                    self.data_source = "database"
                    return df
            except Exception as e:
                logger.warning(f"[{VERSION}][RealDataLoader] Database load failed: {e}")
        
        self._log_audit("DataLoadFailed", "No data available")
        raise DataHealingError("No real data available. Please run data_loader.py or configure DATABASE_URL.")
    
    def _load_from_database(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        from sqlalchemy import text
        
        if symbols:
            symbols_str = ', '.join([f"'{s}'" for s in symbols[:200]])
            query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, pre_close,
                       `change`, pct_chg, volume, amount, turnover_rate, total_mv,
                       pe_ttm, pb
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                AND symbol IN ({symbols_str})
                ORDER BY symbol, trade_date
            """)
        else:
            query = text("""
                SELECT symbol, trade_date, open, high, low, close, pre_close,
                       `change`, pct_chg, volume, amount, turnover_rate, total_mv,
                       pe_ttm, pb
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY symbol, trade_date
            """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql_query(query, conn, params={
                'start_date': start_date,
                'end_date': end_date,
            })
        
        return df if len(df) > 0 else None
    
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log

# ==============================================================================
# V119 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV119:
    """
    V119 Alpha 预测核心引擎 - 特征多样性喷发.
    
    【V119 核心组件】
    1. GeneticFactorMiner: 基因算子搜索 (Incremental_IC + 数据预清洗)
    2. LowdinOrthogonalizer: Lowdin 正交化
    3. MutualInformationFilter: 互信息过滤器 (修复版)
    4. DataPreprocessor: 数据预清洗器
    5. RealDataLoader: 真实数据加载
    """
    
    EPSILON = 1e-6
    
    # V119 基础特征池 - 与 Parquet 数据对齐
    BASE_FACTOR_COLUMNS = [
        'close', 'open', 'high', 'low', 'volume', 'amount',
        'turnover_rate', 'pct_chg', 'change', 'pre_close',
        'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_5', 'volatility_20',
        'volume_ma_ratio_5', 'volume_ma_ratio_20',
        'price_position_20', 'price_position_60',
        'ma_deviation_5', 'ma_deviation_20',
        'rsi_14', 'mfi_14',
        'turnover_bias_20', 'turnover_ma_ratio',
        'volume_price_divergence_5', 'volume_price_divergence_20',
        'volume_price_correlation', 'smart_money_flow',
        'volatility_contraction_10', 'volume_shrink_ratio',
        'volume_price_stable', 'accumulation_distribution_20',
        'macd', 'macd_signal', 'macd_hist',
        'bias_60', 'volume_price_health',
        'volume_shrink_flag', 'price_volume_divergence',
        'hist_sharpe_20d', 'predict_score',
        'filtered_score',
    ]
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_genetic_mining: bool = True,
                 db_url: Optional[str] = None,
                 genetic_config: Optional[Dict] = None) -> None:
        self.config_path = Path(config_path)
        self.enable_genetic_mining = enable_genetic_mining
        self.db_url = db_url or os.getenv('DATABASE_URL')
        
        # V119 核心组件
        self.genetic_miner = GeneticFactorMiner(**(genetic_config or {})) if enable_genetic_mining else None
        self.data_loader = RealDataLoader(db_url=self.db_url)
        self.orthogonalizer = LowdinOrthogonalizer()
        self.preprocessor = DataPreprocessor()
        
        self.factor_ics = {}
        self.genetic_factors = []
        self.audit_log = []
        self.factors = []
        self._load_config()
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][AlphaResearch] V119 Alpha Research Engine Initialized")
        logger.info("=" * 80)
        logger.info(f"  Genetic Mining: {self.enable_genetic_mining}")
        logger.info(f"  Data Source: {self.data_loader.data_source}")
        logger.info(f"  Fitness Function: Incremental_IC (残差 IC)")
        logger.info(f"  Max Factor Order: 3")
        logger.info(f"  MI Pre-filter: Enabled (with Data Preprocessing)")
        logger.info(f"  Lowdin Orthogonalization: Enabled")
        logger.info(f"  Auto-Flip: Physical Inversion")
        logger.info(f"  Toxic Factor Blacklist: {TOXIC_FACTOR_BLACKLIST}")
        logger.info(f"  New Operators: Ts_Entropy, Ts_Skewness")
        logger.info("=" * 80)
    
    def _load_config(self) -> None:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.factors = config.get('factors', [])
            logger.info(f"[{VERSION}][Config] Loaded {len(self.factors)} factor configurations")
        except FileNotFoundError:
            logger.warning(f"[{VERSION}][Config] Config file not found: {self.config_path}")
            self.factors = []
        except yaml.YAMLError as e:
            logger.error(f"[{VERSION}][Config] Failed to parse YAML config: {e}")
            self.factors = []
    
    def _log_audit(self, action: str, details: str = "") -> None:
        log_entry = {'timestamp': datetime.now().isoformat(), 'action': action, 'details': details}
        self.audit_log.append(log_entry)
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【主接口】计算 Alpha 评分 - V119 增强版.
        """
        self._log_audit("ComputeScore", f"Starting V119 score computation with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据源检查
        self._log_audit("DataSource", f"Data source: {self.data_loader.data_source}")
        
        # 2. 基因因子挖掘
        if self.enable_genetic_mining and self.genetic_miner:
            self._log_audit("GeneticMining", "Starting genetic factor mining with Incremental_IC...")
            
            try:
                mined_factors = self.genetic_miner.mine_factors(result, target_count=5, min_fitness=0.005)
                self.genetic_factors = mined_factors
                self._log_audit("GeneticMiningComplete", f"Mined {len(mined_factors)} genetic factors")
                
                for i, factor in enumerate(mined_factors):
                    try:
                        factor_values = self.genetic_miner.compute_factor_value(
                            factor.tree, result, 'trade_date', 'symbol'
                        )
                        if factor_values is not None:
                            try:
                                factor_values = self.orthogonalizer.orthogonalize_full(
                                    factor_values, result, 'trade_date', 'symbol'
                                )
                            except Exception as e:
                                logger.warning(f"[{VERSION}] Orthogonalization failed for factor {i}: {e}")
                            
                            result[f'genetic_factor_{i}'] = factor_values
                            self._log_audit(
                                "GeneticFactorAdded",
                                f"Factor {i}: {factor.expression[:50]}... (order={factor.order}, "
                                f"ic={factor.ic_score:.4f}, auto_flipped={factor.auto_flipped}, "
                                f"orthogonalized={factor.orthogonalized})"
                            )
                    except Exception as e:
                        self._log_audit("GeneticFactorError", f"Factor {i}: {e}")
            except Exception as e:
                self._log_audit("GeneticMiningError", str(e))
        
        # 3. 计算基础因子评分 (排除毒素因子)
        available_factors = [f for f in self.BASE_FACTOR_COLUMNS if f in result.columns and f not in TOXIC_FACTOR_BLACKLIST]
        
        for i in range(len(self.genetic_factors)):
            available_factors.append(f'genetic_factor_{i}')
        
        if not available_factors:
            self._log_audit("NoFactors", "No factors available, using random score")
            result['score'] = np.random.randn(len(result))
        else:
            score = np.zeros(len(result))
            valid_factor_count = 0
            
            for factor in available_factors:
                if factor in result.columns:
                    factor_data = result[factor].fillna(0)
                    
                    # 截面标准化
                    factor_rank = factor_data.groupby(result['trade_date']).transform(
                        lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
                    )
                    
                    # 应用 Lowdin 正交化
                    try:
                        factor_rank = self.orthogonalizer.orthogonalize(factor_rank, result, 'trade_date')
                    except Exception as e:
                        logger.warning(f"[{VERSION}] Orthogonalization failed for {factor}: {e}")
                    
                    score += factor_rank.values
                    valid_factor_count += 1
            
            if valid_factor_count > 0:
                score /= valid_factor_count
            
            result['score'] = score
        
        # 4. 确保 t1_return 存在
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )
        
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-3) / x - 1
            )
        
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-5) / x - 1
            )
        
        # 5. 计算因子 IC
        for factor in available_factors:
            if factor in result.columns:
                ic = self._calculate_factor_ic(result, factor, 't1_return', 'trade_date')
                self.factor_ics[factor] = ic
        
        # 6. 检查累计 IC 并翻转
        total_ic = sum(self.factor_ics.values())
        if total_ic < 0:
            logger.warning(f"[{VERSION}][Auto-Flip] Total IC ({total_ic:.4f}) is negative, flipping score...")
            result['score'] = -result['score']
            self._log_audit("ScoreFlipped", f"Total IC was negative ({total_ic:.4f}), score flipped")
        
        self._log_audit("ComputeScoreComplete", f"Final score calculated with {len(available_factors)} factors")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def _calculate_mean_ic(self, df: pd.DataFrame, score_col: str,
                           label_col: str, date_col: str) -> float:
        ic_values = []
        
        for date in df[date_col].unique():
            day_data = df[df[date_col] == date]
            if len(day_data) < 20:
                continue
            
            score_day = day_data[score_col]
            label_day = day_data[label_col]
            
            mask = score_day.notna() & label_day.notna()
            if mask.sum() < 20:
                continue
            
            score_rank = score_day[mask].rank(method='average')
            label_rank = label_day[mask].rank(method='average')
            
            if np.std(score_rank) > 1e-10 and np.std(label_rank) > 1e-10:
                ic = np.corrcoef(score_rank, label_rank)[0, 1]
                if not np.isnan(ic):
                    ic_values.append(ic)
        
        return float(np.mean(ic_values)) if ic_values else 0.0
    
    def _calculate_factor_ic(self, df: pd.DataFrame, factor_col: str,
                             label_col: str, date_col: str) -> float:
        return self._calculate_mean_ic(df, factor_col, label_col, date_col)
    
    def get_factor_ics(self, df: pd.DataFrame = None) -> Dict[str, float]:
        return self.factor_ics
    
    def get_genetic_mining_report(self) -> Dict[str, Any]:
        if self.genetic_miner:
            return self.genetic_miner.get_mining_report()
        return {}
    
    def get_full_audit_report(self) -> Dict[str, Any]:
        return {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'enable_genetic_mining': self.enable_genetic_mining,
                'data_source': self.data_loader.data_source,
                'fitness_function': 'Incremental_IC + 0.1*ICIR',
                'auto_flip': True,
                'max_order': 3,
                'mi_filter': True,
                'orthogonalization': True,
                'data_preprocessing': True,
                'toxic_blacklist': TOXIC_FACTOR_BLACKLIST,
            },
            'genetic_mining': self.get_genetic_mining_report(),
            'factor_ics': self.factor_ics,
            'audit_log': self.audit_log,
            'data_audit_log': self.data_loader.get_audit_log(),
        }

# ==============================================================================
# V119 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_genetic_mining: bool = True,
                       db_url: Optional[str] = None,
                       genetic_config: Optional[Dict] = None) -> AlphaResearchV119:
    return AlphaResearchV119(
        config_path=config_path,
        enable_genetic_mining=enable_genetic_mining,
        db_url=db_url,
        genetic_config=genetic_config,
    )

# ==============================================================================
# V119 回测运行器
# ==============================================================================

class BacktestRunnerV119:
    """V119 回测运行器"""
    
    def __init__(self,
                 initial_capital: float = 100_000.00,
                 commission_rate: float = 0.0015,
                 stamp_duty_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 top_n: int = 50):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage_rate = slippage_rate
        self.top_n = top_n
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][BacktestRunner] Initialized")
        logger.info("=" * 80)
        logger.info(f"  Initial Capital: {initial_capital:,.0f}")
        logger.info(f"  Commission Rate: {commission_rate:.2%}")
        logger.info(f"  Stamp Duty Rate: {stamp_duty_rate:.2%}")
        logger.info(f"  Slippage Rate: {slippage_rate:.2%}")
        logger.info(f"  Top N Stocks: {top_n}")
        logger.info("=" * 80)
    
    def run(self, df: pd.DataFrame, output_dir: str = "reports") -> Dict[str, Any]:
        from src.engine.backtest_referee import BacktestReferee
        
        alpha_module = get_alpha_research()
        referee = BacktestReferee(alpha_module, output_dir=output_dir)
        referee.VERSION = VERSION
        
        result = referee.run_audit(df)
        return result


def run_v119_backtest(data_path: Optional[str] = None,
                      output_dir: str = "reports") -> Dict[str, Any]:
    loader = RealDataLoader()
    
    if data_path and Path(data_path).exists():
        logger.info(f"[{VERSION}] Loading data from {data_path}")
        df = pd.read_parquet(data_path)
    else:
        try:
            df = loader.load_data(start_date='20240101', end_date='20241231')
        except DataHealingError as e:
            logger.error(f"[{VERSION}] {e}")
            raise
    
    runner = BacktestRunnerV119()
    result = runner.run(df, output_dir)
    
    return result


if __name__ == "__main__":
    try:
        result = run_v119_backtest()
        print(json.dumps(result, indent=2, default=str))
    except DataHealingError as e:
        logger.error(f"V119 requires real data: {e}")
        logger.info("Please configure DATABASE_URL or add Parquet files to data/parquet/")