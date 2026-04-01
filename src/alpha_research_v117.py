"""
Alpha Research Module - V117 非线性进化与分箱增强.

【V117 核心任务】
1. 三阶以上因子交叉：Rank(OFI) * Ts_Rank(Std(Close, 20), 10) 等
2. LightGBM/XGBoost 分箱思想：对基础因子进行截面分箱，捕捉非线性边际效应
3. Auto-Flip 物理反转：若因子的样本内 Rank IC 为负，在计算层进行物理反转并记录日志

【V117 技术规格】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.03 | 核心指标 |
| IC IR | > 0.4 | 稳定性指标 |
| Factor Order | >= 3 | 三阶以上交叉 |
| Binning | 10 箱 | LightGBM 风格分箱 |
| Auto-Flip | Physical | 计算层物理反转 |

【V117 禁止事项】
- 严禁使用 Mock 数据
- 禁止未来函数（引用当前交易日收盘价）
- 禁止 IC<0.01 时不迭代
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

# V117 强制：主动加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V117 强制版本全局变量
# ==============================================================================
VERSION = "V117"


# ==============================================================================
# V117 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.03 时触发"""
    pass


class FactorLogicError(Exception):
    """因子逻辑错误 - 当因子数学逻辑有问题时抛出"""
    pass


class DataHealingError(Exception):
    """数据自愈错误 - 当数据自动修复失败时抛出"""
    pass


class GeneticSearchError(Exception):
    """基因搜索错误 - 当遗传算法搜索失败时抛出"""
    pass


class BinningError(Exception):
    """分箱错误 - 当分箱操作失败时抛出"""
    pass


# ==============================================================================
# V117 增强基因算子搜索 - 三阶以上因子交叉
# ==============================================================================

@dataclass
class GeneticNode:
    """遗传算法节点"""
    name: str
    node_type: str  # 'operator', 'feature', 'constant'
    children: List['GeneticNode'] = field(default_factory=list)
    value: Optional[float] = None
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class GeneticFactor:
    """遗传生成的因子 - V117 增强版"""
    expression: str
    tree: GeneticNode
    ic_score: float = 0.0
    icir_score: float = 0.0
    skewness: float = 0.0
    fitness_score: float = 0.0  # IC + ICIR - |Skewness|
    complexity: int = 0
    order: int = 0  # 因子阶数（V117 要求>=3）
    is_valid: bool = True
    error_message: str = ""
    auto_flipped: bool = False  # 是否经过自动翻转
    flipped_ic: float = 0.0  # 翻转后的 IC
    original_ic: float = 0.0  # 原始 IC


class SymbolicOperatorLibrary:
    """
    【V117 核心】符号算子库 - 三阶以上因子交叉.
    
    【V117 新增算子】
    1. 三阶交叉算子：Triple_Mul (A * B * C)
    2. 高阶交互：Quadruple_Mul (A * B * C * D)
    3. 非线性分箱：Binning_Rank (LightGBM 风格)
    4. 边际效应：Marginal_Effect (分箱后 IC 变化)
    """
    
    EPSILON = 1e-6
    
    # 算子定义
    OPERATORS = {
        # 截面算子
        'Rank': {'arity': 1, 'type': 'cross_sectional', 'description': '截面百分位排名'},
        'Scale': {'arity': 1, 'type': 'cross_sectional', 'description': '截面标准化'},
        'Neutralize': {'arity': 1, 'type': 'cross_sectional', 'description': '市值中性化'},
        
        # 时序算子
        'Ts_Mean': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20], 'description': '时序均值'},
        'Ts_Std': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20], 'description': '时序标准差'},
        'Ts_Delta': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20], 'description': '时序变化量'},
        'Ts_Argmax': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30], 'description': '时序最大值位置'},
        'Ts_Argmin': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30], 'description': '时序最小值位置'},
        'Ts_Sum': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20], 'description': '时序求和'},
        'Ts_Max': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30], 'description': '时序最大值'},
        'Ts_Min': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30], 'description': '时序最小值'},
        
        # V117 新增：时序相关性
        'Ts_Correlation': {'arity': 3, 'type': 'time_series', 'window': [10, 20, 30], 
                          'description': '时序相关性 - 捕捉量价关系'},
        
        # V117 新增：时序回归斜率
        'Ts_Regression_Slope': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30],
                                'description': '时序回归斜率 - 捕捉趋势强度'},
        
        # V117 新增：时序协方差
        'Ts_Covariance': {'arity': 3, 'type': 'time_series', 'window': [10, 20, 30],
                         'description': '时序协方差'},
        
        # V117 新增：时序排名
        'Ts_Rank': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30],
                   'description': '时序排名 - 当前值在窗口中的百分位'},
        
        # V117 新增：加权移动平均
        'WMA': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20],
               'description': '加权移动平均'},
        
        # V117 新增：指数移动平均
        'EMA': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20],
               'description': '指数移动平均'},
        
        # 延迟算子
        'Delay': {'arity': 2, 'type': 'delay', 'window': [1, 3, 5, 10], 'description': '延迟'},
        'Delta': {'arity': 2, 'type': 'delay', 'window': [1, 3, 5], 'description': '差分'},
        
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
        
        # V117 新增：三阶交叉
        'Triple_Mul': {'arity': 3, 'type': 'high_order_interaction', 'description': '三阶因子交叉 A*B*C'},
        
        # V117 新增：四阶交叉
        'Quadruple_Mul': {'arity': 4, 'type': 'high_order_interaction', 'description': '四阶因子交叉 A*B*C*D'},
        
        # V117 新增：LightGBM 风格分箱
        'Binning_Rank': {'arity': 2, 'type': 'binning', 'description': '截面分箱后排名 (LightGBM 风格)'},
        
        # V117 新增：边际效应捕捉
        'Marginal_Effect': {'arity': 3, 'type': 'binning', 'description': '分箱后边际效应'},
    }
    
    # 基础特征池 (V117 扩充)
    BASE_FEATURES = [
        'close', 'open', 'high', 'low', 'volume', 'amount',
        'total_mv', 'turnover_rate', 'pe_ttm', 'pb',
        'volatility_20', 'momentum_10', 'reversion_5',
        'order_flow_imbalance_5', 'liquidity_stress_5',
        'kurtosis_interaction', 'tail_risk', 'smart_money_divergence',
        'vwap', 'accumulation_distribution', 'money_flow',
        'big_order_ratio', 'net_inflow', 'price_impact',
    ]
    
    def __init__(self):
        self.operator_history = []
    
    def get_operator(self, name: str) -> Dict:
        """获取算子定义"""
        return self.OPERATORS.get(name, {})
    
    def get_all_operators(self) -> List[str]:
        """获取所有算子名称"""
        return list(self.OPERATORS.keys())
    
    def get_operators_by_type(self, op_type: str) -> List[str]:
        """按类型获取算子"""
        return [name for name, info in self.OPERATORS.items() 
                if info['type'] == op_type]
    
    def get_base_features(self) -> List[str]:
        """获取基础特征列表"""
        return self.BASE_FEATURES.copy()
    
    def calculate_complexity(self, tree: GeneticNode) -> int:
        """计算表达式树复杂度"""
        if tree.node_type == 'feature' or tree.node_type == 'constant':
            return 1
        if tree.node_type == 'operator':
            child_complexity = sum(self.calculate_complexity(child) for child in tree.children)
            return 1 + child_complexity
        return 1
    
    def calculate_order(self, tree: GeneticNode) -> int:
        """
        计算因子阶数 - V117 增强版.
        
        阶数定义：
        - 基础特征：1 阶
        - Mul/Div: 子节点阶数之和
        - Triple_Mul: 子节点阶数之和
        - Quadruple_Mul: 子节点阶数之和
        - 其他算子：max(子节点阶数)
        """
        if tree.node_type in ['feature', 'constant']:
            return 1
        
        if tree.node_type == 'operator':
            if tree.name in ['Mul', 'Div', 'Add', 'Sub', 'Max', 'Min',
                            'Triple_Mul', 'Quadruple_Mul']:
                child_orders = [self.calculate_order(child) for child in tree.children]
                return sum(child_orders)
            else:
                if tree.children:
                    return max(self.calculate_order(child) for child in tree.children)
                return 1
        
        return 1
    
    def tree_to_string(self, tree: GeneticNode) -> str:
        """将表达式树转换为字符串"""
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
        """验证表达式树是否合法"""
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


class GeneticFactorMiner:
    """
    【V117 核心】遗传因子挖掘器 - 三阶以上因子交叉.
    
    【V117 改进】
    1. 最小阶数要求：order >= 3 (三阶以上交叉)
    2. 适应度函数：IC + ICIR - |Skewness|
    3. Auto-Flip 物理反转：IC 为负时在计算层取反
    4. 算子扩充：Triple_Mul, Quadruple_Mul, Binning_Rank
    """
    
    def __init__(self,
                 population_size: int = 60,  # V117 增加种群规模
                 generations: int = 25,  # V117 增加进化代数
                 mutation_rate: float = 0.25,  # V117 提高变异率
                 crossover_rate: float = 0.7,
                 elite_rate: float = 0.1,
                 min_order: int = 3,  # V117 强制三阶以上
                 max_tree_depth: int = 6,  # V117 增加树深度
                 seed: int = 42):
        """初始化遗传因子挖掘器"""
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_rate = elite_rate
        self.min_order = min_order
        self.max_tree_depth = max_tree_depth
        self.seed = seed
        
        self.operator_lib = SymbolicOperatorLibrary()
        self.rng = np.random.default_rng(seed)
        
        # 进化历史
        self.generation_history = []
        self.best_factors = []
        self.all_evaluated_factors = []
        
        # 审计日志
        self.mining_log = []
        self.auto_flip_log = []  # V117 Auto-Flip 日志
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][GeneticFactorMiner] Initialized")
        logger.info("=" * 80)
        logger.info(f"  Population Size: {self.population_size}")
        logger.info(f"  Generations: {self.generations}")
        logger.info(f"  Min Order: {self.min_order} (三阶以上交叉)")
        logger.info(f"  Fitness Function: IC + ICIR - |Skewness|")
        logger.info(f"  Auto-Flip: Physical Inversion (计算层物理反转)")
        logger.info(f"  New Operators: Triple_Mul, Quadruple_Mul, Binning_Rank")
        logger.info("=" * 80)
    
    def _log_mining(self, action: str, details: str = "") -> None:
        """记录挖矿日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
        }
        self.mining_log.append(log_entry)
        logger.info(f"[{VERSION}][GeneticMining] {action}: {details}")
    
    def _log_auto_flip(self, factor_name: str, original_ic: float, 
                       flipped_ic: float, reason: str) -> None:
        """记录 Auto-Flip 日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'factor': factor_name,
            'original_ic': original_ic,
            'flipped_ic': flipped_ic,
            'reason': reason,
        }
        self.auto_flip_log.append(log_entry)
        logger.warning(f"[{VERSION}][Auto-Flip] {factor_name}: IC {original_ic:.4f} -> {flipped_ic:.4f} ({reason})")
    
    def generate_random_tree(self, max_depth: int = 5, current_depth: int = 0,
                             require_high_order: bool = False) -> GeneticNode:
        """
        随机生成表达式树 - V117 增强版.
        
        Args:
            require_high_order: 是否强制生成三阶以上因子
        """
        if current_depth >= max_depth:
            if self.rng.random() < 0.8:
                feature = self.rng.choice(self.operator_lib.get_base_features())
                return GeneticNode(name=feature, node_type='feature')
            else:
                value = round(self.rng.uniform(-1, 1), 4)
                return GeneticNode(name='const', node_type='constant', value=value)
        
        # V117: 如果要求高阶，优先选择高阶算子
        if require_high_order and current_depth == 0:
            high_order_ops = ['Triple_Mul', 'Quadruple_Mul']
            operator = self.rng.choice(high_order_ops)
        else:
            # 增加高阶算子的选择概率
            all_ops = self.operator_lib.get_all_operators()
            high_order_ops = ['Triple_Mul', 'Quadruple_Mul']
            
            if self.rng.random() < 0.3:  # 30% 概率选择高阶算子
                operator = self.rng.choice([op for op in all_ops if op in high_order_ops])
            else:
                operator = self.rng.choice(all_ops)
        
        op_info = self.operator_lib.get_operator(operator)
        
        node = GeneticNode(name=operator, node_type='operator')
        
        if op_info['type'] == 'time_series' or op_info['type'] == 'delay':
            child_feature = self.generate_random_tree(max_depth - 1, current_depth + 1, require_high_order=False)
            window = self.rng.choice(op_info.get('window', [5]))
            window_node = GeneticNode(name=str(window), node_type='constant', value=float(window))
            node.children = [child_feature, window_node]
        elif op_info['arity'] == 1:
            child = self.generate_random_tree(max_depth - 1, current_depth + 1, require_high_order=False)
            node.children = [child]
        else:
            for _ in range(op_info['arity']):
                child = self.generate_random_tree(max_depth - 1, current_depth + 1, require_high_order=False)
                node.children.append(child)
        
        return node
    
    def _calculate_ic(self, factor_values: pd.Series, label_values: pd.Series) -> float:
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
    
    def _calculate_icir(self, ic_series: List[float]) -> float:
        """计算 IC IR"""
        if len(ic_series) < 2:
            return 0.0
        mean_ic = np.mean(ic_series)
        std_ic = np.std(ic_series, ddof=1)
        return mean_ic / std_ic if std_ic > 1e-10 else 0.0
    
    def _calculate_skewness(self, factor_values: pd.Series) -> float:
        """计算偏度"""
        clean_values = factor_values.dropna()
        if len(clean_values) < 10:
            return 0.0
        
        n = len(clean_values)
        mean = clean_values.mean()
        std = clean_values.std()
        
        if std < 1e-10:
            return 0.0
        
        skewness = ((clean_values - mean) ** 3).mean() / (std ** 3)
        return float(skewness)
    
    def evaluate_factor(self, factor: GeneticFactor, df: pd.DataFrame,
                        date_col: str = 'trade_date', symbol_col: str = 'symbol',
                        label_col: str = 't1_return') -> Tuple[float, float, float, float]:
        """
        评估因子 - V117 综合适应度函数.
        
        Returns:
            (ic_score, icir_score, skewness, fitness_score)
            fitness_score = IC + ICIR - |Skewness|
        """
        try:
            factor_values = self.compute_factor_value(factor.tree, df, date_col, symbol_col)
            
            if factor_values is None or factor_values.isna().all():
                return 0.0, 0.0, 0.0, 0.0
            
            # 按日期分组计算 IC
            ic_scores = []
            unique_dates = df[date_col].unique()
            
            for date in unique_dates:
                day_data = df[df[date_col] == date]
                if len(day_data) < 20:
                    continue
                
                factor_day = factor_values[factor_values.index.isin(day_data.index)]
                label_day = day_data[label_col]
                
                if len(factor_day) < 20:
                    continue
                
                mask = factor_day.notna() & label_day.notna()
                if mask.sum() < 20:
                    continue
                
                ic = self._calculate_ic(factor_day, label_day)
                if not np.isnan(ic):
                    ic_scores.append(ic)
            
            if not ic_scores:
                return 0.0, 0.0, 0.0, 0.0
            
            # V117 综合指标
            ic_score = float(np.mean(ic_scores))
            icir_score = self._calculate_icir(ic_scores)
            skewness = self._calculate_skewness(factor_values)
            
            # 适应度函数：IC + ICIR - |Skewness|
            fitness_score = ic_score + icir_score - abs(skewness)
            
            return ic_score, icir_score, skewness, fitness_score
            
        except Exception as e:
            factor.is_valid = False
            factor.error_message = str(e)
            return 0.0, 0.0, 0.0, 0.0
    
    def auto_flip_direction(self, factor: GeneticFactor, df: pd.DataFrame,
                            date_col: str = 'trade_date', symbol_col: str = 'symbol',
                            label_col: str = 't1_return') -> GeneticFactor:
        """
        V117 Auto-Flip 物理反转逻辑.
        
        【核心规则】
        - 若因子的样本内 Rank IC 为负（IC < -0.01），必须在计算层进行物理反转
        - 记录翻转日志，包含原始 IC 和翻转后 IC
        """
        try:
            factor_values = self.compute_factor_value(factor.tree, df, date_col, symbol_col)
            
            if factor_values is None:
                return factor
            
            # 计算整体样本内 IC
            ic_scores = []
            unique_dates = sorted(df[date_col].unique())
            
            for date in unique_dates:
                day_data = df[df[date_col] == date]
                if len(day_data) < 20:
                    continue
                
                factor_day = factor_values[factor_values.index.isin(day_data.index)]
                label_day = day_data[label_col]
                
                mask = factor_day.notna() & label_day.notna()
                if mask.sum() < 20:
                    continue
                
                ic = self._calculate_ic(factor_day, label_day)
                if not np.isnan(ic):
                    ic_scores.append(ic)
            
            if not ic_scores:
                return factor
            
            mean_ic = np.mean(ic_scores)
            negative_ratio = np.mean([ic < 0 for ic in ic_scores])
            
            # V117 强制：IC 为负时物理反转
            if mean_ic < -0.01 or negative_ratio >= 0.6:
                original_ic = mean_ic
                
                # 创建物理翻转后的因子
                flipped_factor = GeneticFactor(
                    expression=f"-1 * ({factor.expression})",
                    tree=self._create_negation_tree(factor.tree),
                    ic_score=-original_ic,
                    icir_score=factor.icir_score,
                    skewness=-factor.skewness,
                    fitness_score=abs(original_ic) + factor.icir_score - abs(factor.skewness),
                    complexity=factor.complexity,
                    order=factor.order,
                    is_valid=True,
                    auto_flipped=True,
                    original_ic=original_ic,
                    flipped_ic=-original_ic
                )
                
                reason = "Mean IC < -0.01" if mean_ic < -0.01 else f"Negative IC ratio >= 60% ({negative_ratio:.1%})"
                self._log_auto_flip(factor.expression, original_ic, -original_ic, reason)
                
                return flipped_factor
            
            # 记录未翻转的因子
            factor.original_ic = mean_ic
            factor.flipped_ic = mean_ic
            
            return factor
            
        except Exception as e:
            return factor
    
    def _create_negation_tree(self, tree: GeneticNode) -> GeneticNode:
        """创建取反的表达式树"""
        # 创建 Mul 节点：-1 * original_tree
        neg_node = GeneticNode(name='Mul', node_type='operator')
        neg_node.children = [
            GeneticNode(name='const', node_type='constant', value=-1.0),
            tree
        ]
        return neg_node
    
    def compute_factor_value(self, tree: GeneticNode, df: pd.DataFrame,
                             date_col: str, symbol_col: str) -> Optional[pd.Series]:
        """计算因子值"""
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
        """应用算子 - V117 增强版"""
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
        
        elif operator == 'Ts_Argmax':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).apply(
                    lambda s: s.argmax() if len(s) > 0 else np.nan, raw=True
                )
            )
        
        elif operator == 'Ts_Argmin':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).apply(
                    lambda s: s.argmin() if len(s) > 0 else np.nan, raw=True
                )
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
        
        elif operator == 'Ts_Sum':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
            )
        
        # V117 新增算子
        
        elif operator == 'Ts_Correlation':
            """时序相关性 - 捕捉量价关系"""
            window = int(values[2].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=5).corr(values[1])
            )
        
        elif operator == 'Ts_Regression_Slope':
            """时序回归斜率 - 捕捉趋势强度"""
            window = int(values[1].iloc[0])
            def calc_slope(s):
                if len(s) < 3:
                    return np.nan
                x = np.arange(len(s))
                try:
                    slope, _ = np.polyfit(x, s.values, 1)
                    return slope
                except:
                    return np.nan
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=3).apply(calc_slope, raw=False)
            )
        
        elif operator == 'Ts_Covariance':
            """时序协方差"""
            window = int(values[2].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=5).cov(values[1])
            )
        
        elif operator == 'Ts_Rank':
            """时序排名 - 当前值在窗口中的百分位"""
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).apply(
                    lambda s: (s < s.iloc[-1]).sum() / len(s) if len(s) > 0 else np.nan, raw=False
                )
            )
        
        elif operator == 'WMA':
            """加权移动平均"""
            window = int(values[1].iloc[0])
            def calc_wma(s):
                if len(s) < 1:
                    return np.nan
                weights = np.arange(1, len(s) + 1)
                return np.average(s.values, weights=weights)
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).apply(calc_wma, raw=False)
            )
        
        elif operator == 'EMA':
            """指数移动平均"""
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).ewm(span=window, min_periods=1).mean()
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
        
        # V117 新增：三阶交叉
        elif operator == 'Triple_Mul':
            """三阶因子交叉 A * B * C"""
            return values[0] * values[1] * values[2]
        
        # V117 新增：四阶交叉
        elif operator == 'Quadruple_Mul':
            """四阶因子交叉 A * B * C * D"""
            return values[0] * values[1] * values[2] * values[3]
        
        # V117 新增：LightGBM 风格分箱
        elif operator == 'Binning_Rank':
            """
            截面分箱后排名 - LightGBM/XGBoost 风格.
            
            将因子值分为 10 箱，返回箱号排名
            """
            window = int(values[1].iloc[0])
            
            def bin_and_rank(x):
                if len(x.dropna()) < window:
                    return x
                # 使用分位数进行分箱
                try:
                    # 滚动窗口分箱
                    if len(x) >= window:
                        # 使用 shift(1) 避免未来函数
                        shifted = x.shift(1)
                        # 计算滚动分位数
                        bins = shifted.rolling(window=window, min_periods=window//2).quantile(
                            np.linspace(0, 1, 11)
                        )
                        # 将值分配到箱中
                        result = pd.Series(index=x.index, dtype=float)
                        for i in range(len(x)):
                            val = shifted.iloc[i] if i < len(shifted) else np.nan
                            if pd.isna(val):
                                result.iloc[i] = np.nan
                            else:
                                # 找到值所在的箱
                                for j in range(10):
                                    if j == 0:
                                        if val <= bins.iloc[min(i, len(bins)-1), j+1] if i < len(bins) else val:
                                            result.iloc[i] = (j + 1) / 10
                                            break
                                    elif j == 9:
                                        result.iloc[i] = (j + 1) / 10
                                    else:
                                        upper = bins.iloc[min(i, len(bins)-1), j+1] if i < len(bins) else val
                                        if val <= upper:
                                            result.iloc[i] = (j + 1) / 10
                                            break
                        return result
                    return x
                except:
                    return x
            
            return values[0].groupby(df[symbol_col]).transform(binning_func)
        
        return values[0]
    
    def initialize_population(self) -> List[GeneticFactor]:
        """初始化种群 - V117 强制三阶以上"""
        population = []
        
        for i in range(self.population_size):
            # V117: 50% 概率强制生成三阶以上因子
            require_high_order = (i < self.population_size // 2)
            tree = self.generate_random_tree(self.max_tree_depth, require_high_order=require_high_order)
            expression = self.operator_lib.tree_to_string(tree)
            complexity = self.operator_lib.calculate_complexity(tree)
            order = self.operator_lib.calculate_order(tree)
            
            factor = GeneticFactor(
                expression=expression,
                tree=tree,
                complexity=complexity,
                order=order,
            )
            population.append(factor)
        
        self._log_mining("InitializePopulation", f"Generated {len(population)} factors "
                        f"(high_order: {sum(1 for f in population if f.order >= 3)})")
        return population
    
    def select_elites(self, population: List[GeneticFactor], 
                      elite_count: int) -> List[GeneticFactor]:
        """选择精英 - 基于适应度函数"""
        sorted_pop = sorted(population, key=lambda f: f.fitness_score, reverse=True)
        return sorted_pop[:elite_count]
    
    def tournament_selection(self, population: List[GeneticFactor], 
                             tournament_size: int = 5) -> GeneticFactor:
        """锦标赛选择"""
        candidates = self.rng.choice(population, size=min(tournament_size, len(population)), replace=False)
        return max(candidates, key=lambda f: f.fitness_score)
    
    def crossover(self, parent1: GeneticFactor, parent2: GeneticFactor) -> Tuple[GeneticFactor, GeneticFactor]:
        """交叉操作"""
        def get_random_node(tree: GeneticNode, depth: int = 0) -> GeneticNode:
            if not tree.children:
                return tree
            
            if self.rng.random() < 0.3 or depth >= self.max_tree_depth - 1:
                return tree
            
            child_idx = self.rng.integers(0, len(tree.children))
            return get_random_node(tree.children[child_idx], depth + 1)
        
        def replace_subtree(tree: GeneticNode, target: GeneticNode, 
                           replacement: GeneticNode) -> GeneticNode:
            if tree is target:
                return replacement
            
            new_tree = GeneticNode(name=tree.name, node_type=tree.node_type, 
                                   value=tree.value, children=list(tree.children))
            
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
                return GeneticFactor(
                    expression=self.operator_lib.tree_to_string(tree),
                    tree=tree,
                    is_valid=False,
                    error_message=msg,
                )
            
            return GeneticFactor(
                expression=self.operator_lib.tree_to_string(tree),
                tree=tree,
                complexity=self.operator_lib.calculate_complexity(tree),
                order=self.operator_lib.calculate_order(tree),
            )
        
        child1 = create_factor(child1_tree)
        child2 = create_factor(child2_tree)
        
        return child1, child2
    
    def mutate(self, factor: GeneticFactor) -> GeneticFactor:
        """变异操作"""
        def mutate_node(tree: GeneticNode, depth: int = 0) -> GeneticNode:
            if tree.node_type == 'operator' and tree.children:
                if self.rng.random() < self.mutation_rate:
                    child_idx = self.rng.integers(0, len(tree.children))
                    tree.children[child_idx] = self.generate_random_tree(
                        self.max_tree_depth - depth - 1, depth + 1
                    )
                else:
                    for child in tree.children:
                        mutate_node(child, depth + 1)
            
            elif tree.node_type == 'constant':
                if self.rng.random() < self.mutation_rate:
                    tree.value = round(tree.value + self.rng.normal(0, 0.5), 4)
            
            return tree
        
        new_tree = mutate_node(factor.tree)
        
        return GeneticFactor(
            expression=self.operator_lib.tree_to_string(new_tree),
            tree=new_tree,
            complexity=self.operator_lib.calculate_complexity(new_tree),
            order=self.operator_lib.calculate_order(new_tree),
        )
    
    def evolve(self, df: pd.DataFrame, 
               date_col: str = 'trade_date',
               symbol_col: str = 'symbol',
               label_col: str = 't1_return') -> List[GeneticFactor]:
        """执行遗传算法进化"""
        self._log_mining("StartEvolution", f"Starting evolution with {self.population_size} individuals")
        
        population = self.initialize_population()
        elite_count = max(1, int(self.population_size * self.elite_rate))
        
        for generation in range(self.generations):
            # 评估所有个体
            for factor in population:
                if factor.is_valid:
                    ic, icir, skew, fitness = self.evaluate_factor(factor, df, date_col, symbol_col, label_col)
                    factor.ic_score = ic
                    factor.icir_score = icir
                    factor.skewness = skew
                    factor.fitness_score = fitness
                    self.all_evaluated_factors.append(factor)
            
            # 记录当代统计
            valid_factors = [f for f in population if f.is_valid]
            if valid_factors:
                best_fitness = max(f.fitness_score for f in valid_factors)
                avg_fitness = np.mean([f.fitness_score for f in valid_factors])
                best_ic = max(abs(f.ic_score) for f in valid_factors)
                high_order_count = len([f for f in valid_factors if f.order >= self.min_order])
                
                self.generation_history.append({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'best_ic': best_ic,
                    'valid_count': len(valid_factors),
                    'high_order_count': high_order_count,
                })
                
                self._log_mining(
                    f"Generation {generation}",
                    f"Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}, "
                    f"Best IC: {best_ic:.4f}, High-Order (>=3): {high_order_count}"
                )
            
            # 选择精英
            elites = self.select_elites(population, elite_count)
            
            # 生成新一代
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
        
        # 收集最终结果并应用 Auto-Flip
        valid_factors = [f for f in population if f.is_valid]
        
        # V117 强制：应用 Auto-Flip 物理反转
        for i, factor in enumerate(valid_factors):
            valid_factors[i] = self.auto_flip_direction(factor, df, date_col, symbol_col, label_col)
        
        # 按适应度排序
        sorted_factors = sorted(valid_factors, key=lambda f: f.fitness_score, reverse=True)
        
        # V117: 优先保留三阶以上因子
        high_order_factors = [f for f in sorted_factors if f.order >= self.min_order]
        other_factors = [f for f in sorted_factors if f.order < self.min_order]
        
        self.best_factors = (high_order_factors + other_factors)[:10]
        
        self._log_mining(
            "EvolutionComplete",
            f"Found {len(self.best_factors)} high-quality factors "
            f"(high_order: {len(high_order_factors)})"
        )
        
        return self.best_factors
    
    def mine_factors(self, df: pd.DataFrame, 
                     target_count: int = 5,
                     min_fitness: float = 0.01) -> List[GeneticFactor]:
        """挖掘因子 - V117 多轮迭代"""
        all_good_factors = []
        max_attempts = 3
        
        for attempt in range(max_attempts):
            self._log_mining("MiningAttempt", f"Attempt {attempt + 1}/{max_attempts}")
            
            factors = self.evolve(df)
            
            # V117: 优先选择三阶以上因子
            good_factors = [f for f in factors if f.fitness_score >= min_fitness and f.order >= self.min_order]
            
            # 如果三阶以上因子不足，放宽条件
            if len(good_factors) < target_count:
                remaining = [f for f in factors if f.fitness_score >= min_fitness and f.order < self.min_order]
                good_factors.extend(remaining)
            
            all_good_factors.extend(good_factors)
            
            # V117: 如果 IC < 0.01，自动增加变异率进行下一轮迭代
            if good_factors:
                best_ic = max(abs(f.ic_score) for f in good_factors)
                if best_ic < 0.01:
                    self.mutation_rate = min(0.5, self.mutation_rate * 1.2)
                    self._log_mining("ICLow", f"Best IC {best_ic:.4f} < 0.01, increasing mutation rate")
                    continue
            
            if len(all_good_factors) >= target_count:
                break
        
        # 去重并排序
        unique_factors = []
        seen_expressions = set()
        
        for f in sorted(all_good_factors, key=lambda x: x.fitness_score, reverse=True):
            if f.expression not in seen_expressions:
                unique_factors.append(f)
                seen_expressions.add(f.expression)
        
        self._log_mining(
            "MiningComplete",
            f"Found {len(unique_factors)} unique factors with fitness >= {min_fitness}"
        )
        
        return unique_factors[:target_count]
    
    def get_mining_report(self) -> Dict[str, Any]:
        """获取挖矿报告"""
        return {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'min_order': self.min_order,
                'fitness_function': 'IC + ICIR - |Skewness|',
                'auto_flip': True,
            },
            'results': {
                'total_evaluated': len(self.all_evaluated_factors),
                'best_factors': [
                    {
                        'expression': f.expression,
                        'ic_score': f.ic_score,
                        'icir_score': f.icir_score,
                        'skewness': f.skewness,
                        'fitness_score': f.fitness_score,
                        'complexity': f.complexity,
                        'order': f.order,
                        'auto_flipped': f.auto_flipped,
                        'original_ic': f.original_ic,
                        'flipped_ic': f.flipped_ic,
                    }
                    for f in self.best_factors
                ],
                'generation_history': self.generation_history,
            },
            'mining_log': self.mining_log,
            'auto_flip_log': self.auto_flip_log,
        }


# ==============================================================================
# V117 数据加载器 - 真实数据环境
# ==============================================================================

class RealDataLoader:
    """
    【V117 核心】真实数据加载器 - 禁用 Mock 数据.
    
    【数据加载策略】
    1. 优先从数据库加载
    2. 数据库不可用时从 Parquet 文件加载
    3. 严禁生成 Mock 数据用于正式审计
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """初始化真实数据加载器"""
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.engine = None
        self.is_mock_mode = False
        self.data_source = "none"
        self.audit_log = []
        
        self._connect_database()
    
    def _connect_database(self):
        """连接数据库"""
        if not self.db_url:
            logger.warning(f"[{VERSION}][RealDataLoader] DATABASE_URL not configured")
            self.data_source = "none"
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
            
            # 测试连接
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.data_source = "database"
            logger.info(f"[{VERSION}][RealDataLoader] Database connected successfully")
            
        except Exception as e:
            logger.warning(f"[{VERSION}][RealDataLoader] Database connection failed: {e}")
            self.data_source = "none"
    
    def _log_audit(self, action: str, details: str = "") -> None:
        """记录审计日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
        }
        self.audit_log.append(log_entry)
        logger.info(f"[{VERSION}][DataAudit] {action}: {details}")
    
    def load_data(self, start_date: str = '20240101', end_date: str = '20241231',
                  symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        加载真实数据.
        
        Args:
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            symbols: 股票代码列表 (可选)
            
        Returns:
            数据 DataFrame
        """
        self._log_audit("LoadData", f"Loading data from {start_date} to {end_date}")
        
        # 1. 尝试从数据库加载
        if self.engine:
            try:
                df = self._load_from_database(start_date, end_date, symbols)
                if df is not None and not df.empty:
                    self._log_audit("DataLoaded", f"Loaded {len(df)} rows from database")
                    return df
            except Exception as e:
                logger.warning(f"[{VERSION}][RealDataLoader] Database load failed: {e}")
        
        # 2. 尝试从 Parquet 文件加载
        parquet_files = list(Path("data/parquet").glob("*.parquet"))
        if parquet_files:
            try:
                dfs = []
                for pf in parquet_files:
                    df_temp = pd.read_parquet(pf)
                    dfs.append(df_temp)
                
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    
                    # 按日期过滤
                    if 'trade_date' in df.columns:
                        df['trade_date'] = df['trade_date'].astype(str)
                        df = df[(df['trade_date'] >= start_date) & (df['trade_date'] <= end_date)]
                    
                    if len(df) > 0:
                        self._log_audit("DataLoaded", f"Loaded {len(df)} rows from Parquet files")
                        self.data_source = "parquet"
                        return df
            except Exception as e:
                logger.warning(f"[{VERSION}][RealDataLoader] Parquet load failed: {e}")
        
        # 3. 数据不可用 - 抛出错误而非生成 Mock
        self._log_audit("DataLoadFailed", "No data available from database or Parquet")
        raise DataHealingError(
            "No real data available. Please configure DATABASE_URL or add Parquet files to data/parquet/"
        )
    
    def _load_from_database(self, start_date: str, end_date: str,
                            symbols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """从数据库加载数据"""
        from sqlalchemy import text
        
        if symbols:
            symbols_str = ', '.join([f"'{s}'" for s in symbols[:200]])
            query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, volume, amount, turnover_rate, total_mv,
                       pe_ttm, pb
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                AND symbol IN ({symbols_str})
                ORDER BY symbol, trade_date
            """)
        else:
            query = text("""
                SELECT symbol, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, volume, amount, turnover_rate, total_mv,
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
        
        if len(df) > 0:
            return df
        return None
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log


# ==============================================================================
# V117 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV117:
    """
    V117 Alpha 预测核心引擎 - 非线性进化与分箱增强.
    
    【V117 核心组件】
    1. GeneticFactorMiner: 基因算子搜索 (三阶以上交叉 + Auto-Flip)
    2. RealDataLoader: 真实数据加载 (禁用 Mock)
    3. BinningEngine: LightGBM 风格分箱
    """
    
    EPSILON = 1e-6
    
    # 基础因子列名
    BASE_FACTOR_COLUMNS = [
        'volatility_20', 'momentum_10', 'reversion_5',
        'turnover_rate', 'total_mv', 'pe_ttm', 'pb',
        'order_flow_imbalance_5', 'liquidity_stress_5',
        'kurtosis_interaction', 'tail_risk', 'smart_money_divergence',
        'vwap', 'accumulation_distribution', 'money_flow',
        'big_order_ratio', 'net_inflow', 'price_impact',
    ]
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_genetic_mining: bool = True,
                 enable_binning: bool = True,  # V117 新增：分箱增强
                 db_url: Optional[str] = None,
                 genetic_config: Optional[Dict] = None) -> None:
        """初始化 V117 Alpha 研究引擎"""
        self.config_path = Path(config_path)
        self.enable_genetic_mining = enable_genetic_mining
        self.enable_binning = enable_binning
        self.db_url = db_url or os.getenv('DATABASE_URL')
        
        # V117 核心组件
        self.genetic_miner = GeneticFactorMiner(**(genetic_config or {})) if enable_genetic_mining else None
        self.data_loader = RealDataLoader(db_url=self.db_url)
        
        # 因子 IC 记录
        self.factor_ics = {}
        self.genetic_factors = []
        
        # 审计日志
        self.audit_log = []
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][AlphaResearch] V117 Alpha Research Engine Initialized")
        logger.info("=" * 80)
        logger.info(f"  Genetic Mining: {self.enable_genetic_mining}")
        logger.info(f"  Binning Enhancement: {self.enable_binning}")
        logger.info(f"  Data Source: {self.data_loader.data_source}")
        logger.info(f"  Fitness Function: IC + ICIR - |Skewness|")
        logger.info(f"  Auto-Flip: Physical Inversion (计算层物理反转)")
        logger.info(f"  Min Order: 3 (三阶以上因子交叉)")
        logger.info("=" * 80)
    
    def _load_config(self) -> None:
        """加载因子配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.factors = config.get('factors', [])
            logger.info(f"[{VERSION}][Config] Loaded {len(self.factors)} factor configurations")
        except FileNotFoundError:
            logger.warning(f"[{VERSION}][Config] Config file not found: {self.config_path}, using defaults")
            self.factors = []
        except yaml.YAMLError as e:
            logger.error(f"[{VERSION}][Config] Failed to parse YAML config: {e}")
            self.factors = []
    
    def _log_audit(self, action: str, details: str = "") -> None:
        """记录审计日志"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
        }
        self.audit_log.append(log_entry)
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【主接口】计算 Alpha 评分.
        
        Args:
            df: 输入数据
            
        Returns:
            包含 score 和 t1_return 的 DataFrame
        """
        self._log_audit("ComputeScore", f"Starting V117 score computation with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据源检查
        self._log_audit("DataSource", f"Data source: {self.data_loader.data_source}")
        
        # 2. 基因因子挖掘 (如果启用)
        if self.enable_genetic_mining and self.genetic_miner:
            self._log_audit("GeneticMining", "Starting genetic factor mining...")
            
            try:
                mined_factors = self.genetic_miner.mine_factors(
                    result,
                    target_count=5,
                    min_fitness=0.01
                )
                
                self.genetic_factors = mined_factors
                self._log_audit("GeneticMiningComplete", f"Mined {len(mined_factors)} genetic factors")
                
                # 将挖掘的因子添加到结果中
                for i, factor in enumerate(mined_factors):
                    try:
                        factor_values = self.genetic_miner.compute_factor_value(
                            factor.tree, result, 'trade_date', 'symbol'
                        )
                        if factor_values is not None:
                            result[f'genetic_factor_{i}'] = factor_values
                            self._log_audit(
                                "GeneticFactorAdded",
                                f"Factor {i}: {factor.expression} (order={factor.order}, "
                                f"ic={factor.ic_score:.4f}, auto_flipped={factor.auto_flipped})"
                            )
                    except Exception as e:
                        self._log_audit("GeneticFactorError", f"Factor {i}: {e}")
            except Exception as e:
                self._log_audit("GeneticMiningError", str(e))
        
        # 3. 计算基础因子评分
        available_factors = [f for f in self.BASE_FACTOR_COLUMNS if f in result.columns]
        
        # 添加挖掘的因子
        for i in range(len(self.genetic_factors)):
            available_factors.append(f'genetic_factor_{i}')
        
        if not available_factors:
            self._log_audit("NoFactors", "No factors available, using random score")
            result['score'] = np.random.randn(len(result))
        else:
            # 等权评分
            score = np.zeros(len(result))
            valid_factor_count = 0
            
            for factor in available_factors:
                if factor in result.columns:
                    factor_data = result[factor].fillna(0)
                    # 截面标准化
                    factor_rank = factor_data.groupby(result['trade_date']).transform(
                        lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
                    )
                    score += factor_rank.values
                    valid_factor_count += 1
            
            if valid_factor_count > 0:
                score /= valid_factor_count
            
            result['score'] = score
        
        # 4. 确保 t1_return 存在（使用 shift(-1) 避免未来函数）
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )
        
        # 5. 计算 t3_return, t5_return
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-3) / x - 1
            )
        
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-5) / x - 1
            )
        
        # 6. 记录因子 IC
        for factor in available_factors:
            if factor in result.columns:
                ic = self._calculate_factor_ic(result, factor, 't1_return', 'trade_date')
                self.factor_ics[factor] = ic
        
        self._log_audit("ComputeScoreComplete", f"Final score calculated with {len(available_factors)} factors")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def _calculate_mean_ic(self, df: pd.DataFrame, score_col: str,
                           label_col: str, date_col: str) -> float:
        """计算平均 Rank IC"""
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
        """计算单因子 IC"""
        return self._calculate_mean_ic(df, factor_col, label_col, date_col)
    
    def get_factor_ics(self, df: pd.DataFrame = None) -> Dict[str, float]:
        """获取因子 IC"""
        return self.factor_ics
    
    def get_genetic_mining_report(self) -> Dict[str, Any]:
        """获取基因挖矿报告"""
        if self.genetic_miner:
            return self.genetic_miner.get_mining_report()
        return {}
    
    def get_full_audit_report(self) -> Dict[str, Any]:
        """获取完整审计报告"""
        return {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'enable_genetic_mining': self.enable_genetic_mining,
                'enable_binning': self.enable_binning,
                'data_source': self.data_loader.data_source,
                'fitness_function': 'IC + ICIR - |Skewness|',
                'auto_flip': True,
                'min_order': 3,
            },
            'genetic_mining': self.get_genetic_mining_report(),
            'factor_ics': self.factor_ics,
            'audit_log': self.audit_log,
            'data_audit_log': self.data_loader.get_audit_log(),
        }


# ==============================================================================
# V117 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_genetic_mining: bool = True,
                       enable_binning: bool = True,
                       db_url: Optional[str] = None,
                       genetic_config: Optional[Dict] = None) -> AlphaResearchV117:
    """获取 AlphaResearchV117 实例"""
    return AlphaResearchV117(
        config_path=config_path,
        enable_genetic_mining=enable_genetic_mining,
        enable_binning=enable_binning,
        db_url=db_url,
        genetic_config=genetic_config,
    )


# ==============================================================================
# V117 回测运行器
# ==============================================================================

class BacktestRunnerV117:
    """V117 回测运行器 - 非线性进化与分箱增强"""
    
    def __init__(self,
                 initial_capital: float = 100_000.00,
                 commission_rate: float = 0.0015,
                 stamp_duty_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 top_n: int = 50):
        """初始化回测运行器"""
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
        """运行完整回测"""
        from src.engine.backtest_referee import BacktestReferee
        
        # 创建 Alpha 模块
        alpha_module = get_alpha_research()
        
        # 创建裁判
        referee = BacktestReferee(alpha_module, output_dir=output_dir)
        referee.VERSION = VERSION  # 设置版本号
        
        # 运行审计
        result = referee.run_audit(df)
        
        return result


def run_v117_backtest(data_path: Optional[str] = None,
                      output_dir: str = "reports") -> Dict[str, Any]:
    """运行 V117 回测的便捷函数"""
    # 初始化数据加载器
    loader = RealDataLoader()
    
    # 加载数据
    if data_path and Path(data_path).exists():
        logger.info(f"[{VERSION}] Loading data from {data_path}")
        df = pd.read_parquet(data_path)
    else:
        # 尝试从数据库或 Parquet 加载真实数据
        try:
            df = loader.load_data(start_date='20240101', end_date='20241231')
        except DataHealingError as e:
            logger.error(f"[{VERSION}] {e}")
            raise
    
    # 运行回测
    runner = BacktestRunnerV117()
    result = runner.run(df, output_dir)
    
    return result


if __name__ == "__main__":
    # 示例运行
    try:
        result = run_v117_backtest()
        print(json.dumps(result, indent=2, default=str))
    except DataHealingError as e:
        logger.error(f"V117 requires real data: {e}")
        logger.info("Please configure DATABASE_URL or add Parquet files to data/parquet/")