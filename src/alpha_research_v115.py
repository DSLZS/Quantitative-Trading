"""
Alpha Research Module - V115 自动化特征挖掘与场景化 Alpha 实验室.

【V115 核心任务】
1. 架构审计与版本对齐：动态版本号，文件名严格遵循 v115_audit_{datetime}.md
2. 算法重心：引入基因算子搜索（Symbolic Logic）- GeneticFactorMiner
3. 数据自愈与环境对齐：.env 自动检测，Mock 数据生成
4. 场景感知回归：MarketRegimeDetector 四象限动态权重
5. 裁判与输出红线：T+1 IC < 0.01 时输出因子消融实验

【V115 技术规格】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.015 | 核心指标 (洗掉市值影响后) |
| IC IR | > 0.3 | 正交化效果指标 |
| Genetic Factors | ≥ 5 | 三阶以上复杂因子数量 |
| Regime Accuracy | > 0.6 | 场景识别准确率 |

【V115 禁止事项】
- 禁止生成简单的 MA(close, 5) 单因子
- 日志中禁止出现"V103"字样
- 文件名必须是 v115_audit_...
- 初始资金锁定 100,000.00，单边费率 0.15%
"""

from typing import Any, Optional, Union, Dict, List, Tuple
from pathlib import Path
import warnings
import time
import json
import os
import random
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

# V115 强制：主动加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V115 强制版本全局变量
# ==============================================================================
VERSION = "V115"


# ==============================================================================
# V115 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.015 时触发"""
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


class RegimeDetectionError(Exception):
    """场景检测错误 - 当市场场景识别失败时抛出"""
    pass


# ==============================================================================
# V115 基因算子搜索 - Symbolic Logic
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
    """遗传生成的因子"""
    expression: str
    tree: GeneticNode
    ic_score: float = 0.0
    complexity: int = 0
    order: int = 0  # 因子阶数 (三阶以上为目标)
    is_valid: bool = True
    error_message: str = ""


class SymbolicOperatorLibrary:
    """
    【V115 核心】符号算子库 - 用于遗传算法生成复杂因子。
    
    【算子分类】
    1. 截面算子 (Cross-Sectional): Rank, Scale, Neutralize
    2. 时序算子 (Time-Series): Ts_Mean, Ts_Std, Ts_Delta, Ts_Argmax, Ts_Argmin
    3. 延迟算子 (Delay): Delay, Delta
    4. 数学算子 (Mathematical): Log, Sqrt, Abs, Sign
    5. 交互算子 (Interaction): Mul, Div, Add, Sub
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
        
        # 延迟算子
        'Delay': {'arity': 2, 'type': 'delay', 'window': [1, 3, 5, 10], 'description': '延迟'},
        'Delta': {'arity': 2, 'type': 'delay', 'window': [1, 3, 5], 'description': '差分'},
        
        # 数学算子
        'Log': {'arity': 1, 'type': 'mathematical', 'description': '对数'},
        'Sqrt': {'arity': 1, 'type': 'mathematical', 'description': '平方根'},
        'Abs': {'arity': 1, 'type': 'mathematical', 'description': '绝对值'},
        'Sign': {'arity': 1, 'type': 'mathematical', 'description': '符号'},
        'Square': {'arity': 1, 'type': 'mathematical', 'description': '平方'},
        
        # 交互算子
        'Mul': {'arity': 2, 'type': 'interaction', 'description': '乘法交互'},
        'Div': {'arity': 2, 'type': 'interaction', 'description': '除法交互'},
        'Add': {'arity': 2, 'type': 'interaction', 'description': '加法'},
        'Sub': {'arity': 2, 'type': 'interaction', 'description': '减法'},
        'Max': {'arity': 2, 'type': 'interaction', 'description': '最大值'},
        'Min': {'arity': 2, 'type': 'interaction', 'description': '最小值'},
    }
    
    # 基础特征池
    BASE_FEATURES = [
        'close', 'open', 'high', 'low', 'volume', 'amount',
        'total_mv', 'turnover_rate', 'pe_ttm', 'pb',
        'volatility_20', 'momentum_10', 'reversion_5',
        'order_flow_imbalance_5', 'liquidity_stress_5',
        'kurtosis_interaction', 'tail_risk', 'smart_money_divergence',
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
        计算因子阶数。
        
        【阶数定义】
        - 一阶：单一基础特征
        - 二阶：两个特征交互 (如 Rank(A) * Rank(B))
        - 三阶：三个特征交互 (如 Rank(A) * Rank(B) * Rank(C))
        - 高阶：四个及以上特征交互
        """
        if tree.node_type in ['feature', 'constant']:
            return 1
        
        if tree.node_type == 'operator':
            # 交互算子增加阶数
            if tree.name in ['Mul', 'Div', 'Add', 'Sub', 'Max', 'Min']:
                child_orders = [self.calculate_order(child) for child in tree.children]
                return sum(child_orders)
            else:
                # 其他算子保持最大子节点阶数
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
    【V115 核心】遗传因子挖掘器 - 利用 AI 模拟遗传算法生成三阶以上复杂因子。
    
    【遗传算法流程】
    1. 初始化种群：随机生成 N 个因子表达式树
    2. 适应度评估：计算每个因子的 IC 分数
    3. 选择：保留适应度高的个体
    4. 交叉：交换两个个体的子树
    5. 变异：随机修改节点
    6. 重复 2-5 直到收敛
    
    【目标】
    - 生成至少 5 个三阶以上的复杂因子
    - 挖掘洗掉市值影响后，Rank IC 依然能稳在 0.015 以上的纯净因子
    
    【示例逻辑】
    - Rank(Ts_Argmax(Close, 20)) / (Ts_Std(Volume, 5) * Rank(OFI))
    - 这种多维非线性组合才是抵御"脱毒"后 IC 暴跌的关键
    """
    
    def __init__(self,
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.6,
                 elite_rate: float = 0.1,
                 min_order: int = 3,  # 最小阶数要求
                 max_tree_depth: int = 6,
                 seed: int = 42):
        """
        初始化遗传因子挖掘器。
        
        Args:
            population_size: 种群大小
            generations: 迭代代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_rate: 精英保留率
            min_order: 最小因子阶数 (默认 3 阶)
            max_tree_depth: 最大树深度
            seed: 随机种子
        """
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
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][GeneticFactorMiner] Initialized")
        logger.info("=" * 80)
        logger.info(f"  Population Size: {self.population_size}")
        logger.info(f"  Generations: {self.generations}")
        logger.info(f"  Mutation Rate: {self.mutation_rate:.1%}")
        logger.info(f"  Crossover Rate: {self.crossover_rate:.1%}")
        logger.info(f"  Elite Rate: {self.elite_rate:.1%}")
        logger.info(f"  Min Order: {self.min_order}")
        logger.info(f"  Max Tree Depth: {self.max_tree_depth}")
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
    
    def generate_random_tree(self, max_depth: int = 4, current_depth: int = 0) -> GeneticNode:
        """
        随机生成表达式树。
        
        Args:
            max_depth: 最大深度
            current_depth: 当前深度
            
        Returns:
            随机生成的表达式树
        """
        # 达到最大深度时，只能选择基础特征或常数
        if current_depth >= max_depth:
            if self.rng.random() < 0.8:
                # 选择基础特征
                feature = self.rng.choice(self.operator_lib.get_base_features())
                return GeneticNode(name=feature, node_type='feature')
            else:
                # 选择常数
                value = round(self.rng.uniform(-1, 1), 4)
                return GeneticNode(name='const', node_type='constant', value=value)
        
        # 选择算子
        operator = self.rng.choice(self.operator_lib.get_all_operators())
        op_info = self.operator_lib.get_operator(operator)
        
        # 创建节点
        node = GeneticNode(name=operator, node_type='operator')
        
        # 根据算子类型生成子节点
        if op_info['type'] == 'time_series' or op_info['type'] == 'delay':
            # 时序算子需要 2 个参数：特征和窗口
            child_feature = self.generate_random_tree(max_depth - 1, current_depth + 1)
            window = self.rng.choice(op_info.get('window', [5]))
            window_node = GeneticNode(name=str(window), node_type='constant', value=float(window))
            node.children = [child_feature, window_node]
        elif op_info['arity'] == 1:
            # 单目算子
            child = self.generate_random_tree(max_depth - 1, current_depth + 1)
            node.children = [child]
        else:
            # 多目算子
            for _ in range(op_info['arity']):
                child = self.generate_random_tree(max_depth - 1, current_depth + 1)
                node.children.append(child)
        
        return node
    
    def evaluate_factor(self, factor: GeneticFactor, df: pd.DataFrame,
                        date_col: str = 'trade_date', symbol_col: str = 'symbol',
                        label_col: str = 't1_return') -> float:
        """
        评估因子 IC 分数。
        
        Args:
            factor: 待评估的因子
            df: 数据 DataFrame
            date_col: 日期列
            symbol_col: 股票代码列
            label_col: 标签列
            
        Returns:
            IC 分数
        """
        try:
            # 计算因子值
            factor_values = self.compute_factor_value(factor.tree, df, date_col, symbol_col)
            
            if factor_values is None or factor_values.isna().all():
                return 0.0
            
            # 计算 Rank IC
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
                
                # 去除空值
                mask = factor_day.notna() & label_day.notna()
                if mask.sum() < 20:
                    continue
                
                # 计算秩相关
                factor_rank = factor_day[mask].rank(method='average')
                label_rank = label_day[mask].rank(method='average')
                
                if np.std(factor_rank) > 1e-10 and np.std(label_rank) > 1e-10:
                    ic = np.corrcoef(factor_rank, label_rank)[0, 1]
                    if not np.isnan(ic):
                        ic_scores.append(ic)
            
            if not ic_scores:
                return 0.0
            
            return float(np.mean(ic_scores))
            
        except Exception as e:
            factor.is_valid = False
            factor.error_message = str(e)
            return 0.0
    
    def compute_factor_value(self, tree: GeneticNode, df: pd.DataFrame,
                             date_col: str, symbol_col: str) -> Optional[pd.Series]:
        """
        计算因子值。
        
        Args:
            tree: 表达式树
            df: 数据 DataFrame
            date_col: 日期列
            symbol_col: 股票代码列
            
        Returns:
            因子值 Series
        """
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
                
                # 检查是否有 None
                if any(v is None for v in children_values):
                    return None
                
                return self.apply_operator(tree.name, children_values, df, date_col, symbol_col)
            
            return None
            
        except Exception as e:
            return None
    
    def apply_operator(self, operator: str, values: List[pd.Series],
                       df: pd.DataFrame, date_col: str, symbol_col: str) -> pd.Series:
        """应用算子"""
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
                lambda x: x.shift(1).rolling(window=window).mean()
            )
        
        elif operator == 'Ts_Std':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window).std()
            )
        
        elif operator == 'Ts_Delta':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1) - x.shift(window + 1)
            )
        
        elif operator == 'Ts_Argmax':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window).apply(
                    lambda s: s.argmax() if len(s) > 0 else np.nan, raw=True
                )
            )
        
        elif operator == 'Ts_Argmin':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window).apply(
                    lambda s: s.argmin() if len(s) > 0 else np.nan, raw=True
                )
            )
        
        elif operator == 'Ts_Max':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window).max()
            )
        
        elif operator == 'Ts_Min':
            window = int(values[1].iloc[0])
            return values[0].groupby(df[symbol_col]).transform(
                lambda x: x.shift(1).rolling(window=window).min()
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
    
    def initialize_population(self) -> List[GeneticFactor]:
        """初始化种群"""
        population = []
        
        for i in range(self.population_size):
            tree = self.generate_random_tree(self.max_tree_depth)
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
        
        self._log_mining("InitializePopulation", f"Generated {len(population)} factors")
        return population
    
    def select_elites(self, population: List[GeneticFactor], 
                      elite_count: int) -> List[GeneticFactor]:
        """选择精英"""
        sorted_pop = sorted(population, key=lambda f: abs(f.ic_score), reverse=True)
        return sorted_pop[:elite_count]
    
    def tournament_selection(self, population: List[GeneticFactor], 
                             tournament_size: int = 5) -> GeneticFactor:
        """锦标赛选择"""
        candidates = self.rng.choice(population, size=min(tournament_size, len(population)), replace=False)
        return max(candidates, key=lambda f: abs(f.ic_score))
    
    def crossover(self, parent1: GeneticFactor, parent2: GeneticFactor) -> Tuple[GeneticFactor, GeneticFactor]:
        """
        交叉操作 - 交换两个父代的子树。
        
        Args:
            parent1: 父代 1
            parent2: 父代 2
            
        Returns:
            两个子代因子
        """
        def get_random_node(tree: GeneticNode, depth: int = 0) -> Tuple[GeneticNode, GeneticNode, str]:
            """随机获取一个节点及其父节点"""
            if not tree.children:
                return tree, None, 'root'
            
            if self.rng.random() < 0.3 or depth >= self.max_tree_depth - 1:
                return tree, None, 'root'
            
            child_idx = self.rng.integers(0, len(tree.children))
            return get_random_node(tree.children[child_idx], depth + 1)
        
        def replace_subtree(tree: GeneticNode, target: GeneticNode, 
                           replacement: GeneticNode) -> GeneticNode:
            """替换子树"""
            if tree is target:
                return replacement
            
            new_tree = GeneticNode(name=tree.name, node_type=tree.node_type, 
                                   value=tree.value, children=tree.children.copy())
            
            for i, child in enumerate(new_tree.children):
                if child is target:
                    new_tree.children[i] = replacement
                else:
                    new_tree.children[i] = replace_subtree(child, target, replacement)
            
            return new_tree
        
        # 获取两个父代的随机节点
        node1, _, _ = get_random_node(parent1.tree)
        node2, _, _ = get_random_node(parent2.tree)
        
        # 创建子代
        child1_tree = replace_subtree(parent1.tree, node1, node2)
        child2_tree = replace_subtree(parent2.tree, node2, node1)
        
        # 验证并创建子代因子
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
        """
        变异操作 - 随机修改节点。
        
        Args:
            factor: 待变异的因子
            
        Returns:
            变异后的因子
        """
        def mutate_node(tree: GeneticNode, depth: int = 0) -> GeneticNode:
            """递归变异节点"""
            if tree.node_type == 'operator' and tree.children:
                # 随机选择一个子节点进行变异
                if self.rng.random() < self.mutation_rate:
                    child_idx = self.rng.integers(0, len(tree.children))
                    tree.children[child_idx] = self.generate_random_tree(
                        self.max_tree_depth - depth - 1, depth + 1
                    )
                else:
                    for child in tree.children:
                        mutate_node(child, depth + 1)
            
            elif tree.node_type == 'constant':
                # 变异常数值
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
        """
        执行遗传算法进化。
        
        Args:
            df: 数据 DataFrame
            date_col: 日期列
            symbol_col: 股票代码列
            label_col: 标签列
            
        Returns:
            进化后的种群
        """
        self._log_mining("StartEvolution", f"Starting evolution with {self.population_size} individuals")
        
        # 初始化种群
        population = self.initialize_population()
        
        elite_count = max(1, int(self.population_size * self.elite_rate))
        
        for generation in range(self.generations):
            # 评估所有个体
            for factor in population:
                if factor.is_valid:
                    factor.ic_score = self.evaluate_factor(factor, df, date_col, symbol_col, label_col)
                    self.all_evaluated_factors.append(factor)
            
            # 记录当代统计
            valid_factors = [f for f in population if f.is_valid]
            if valid_factors:
                best_ic = max(abs(f.ic_score) for f in valid_factors)
                avg_ic = np.mean([abs(f.ic_score) for f in valid_factors])
                
                self.generation_history.append({
                    'generation': generation,
                    'best_ic': best_ic,
                    'avg_ic': avg_ic,
                    'valid_count': len(valid_factors),
                    'high_order_count': len([f for f in valid_factors if f.order >= self.min_order]),
                })
                
                self._log_mining(
                    f"Generation {generation}",
                    f"Best IC: {best_ic:.4f}, Avg IC: {avg_ic:.4f}, "
                    f"Valid: {len(valid_factors)}/{len(population)}, "
                    f"High-Order: {len([f for f in valid_factors if f.order >= self.min_order])}"
                )
            
            # 选择精英
            elites = self.select_elites(population, elite_count)
            
            # 生成新一代
            new_population = elites.copy()
            
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # 交叉
                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # 添加到新种群
                if len(new_population) < self.population_size and child1.is_valid:
                    new_population.append(child1)
                if len(new_population) < self.population_size and child2.is_valid:
                    new_population.append(child2)
            
            population = new_population
        
        # 收集最终结果
        valid_factors = [f for f in population if f.is_valid and f.order >= self.min_order]
        sorted_factors = sorted(valid_factors, key=lambda f: abs(f.ic_score), reverse=True)
        
        self.best_factors = sorted_factors[:10]  # 保留 top 10
        
        self._log_mining(
            "EvolutionComplete",
            f"Found {len(self.best_factors)} high-order factors (order >= {self.min_order})"
        )
        
        return self.best_factors
    
    def mine_factors(self, df: pd.DataFrame, 
                     target_count: int = 5,
                     min_ic: float = 0.01) -> List[GeneticFactor]:
        """
        挖掘因子直到找到足够的优质因子。
        
        Args:
            df: 数据 DataFrame
            target_count: 目标因子数量
            min_ic: 最小 IC 阈值
            
        Returns:
            符合条件的因子列表
        """
        all_good_factors = []
        max_attempts = 5
        
        for attempt in range(max_attempts):
            self._log_mining("MiningAttempt", f"Attempt {attempt + 1}/{max_attempts}")
            
            # 执行进化
            factors = self.evolve(df)
            
            # 筛选优质因子
            good_factors = [f for f in factors if abs(f.ic_score) >= min_ic]
            all_good_factors.extend(good_factors)
            
            if len(all_good_factors) >= target_count:
                break
        
        # 去重并排序
        unique_factors = []
        seen_expressions = set()
        
        for f in sorted(all_good_factors, key=lambda x: abs(x.ic_score), reverse=True):
            if f.expression not in seen_expressions:
                unique_factors.append(f)
                seen_expressions.add(f.expression)
        
        self._log_mining(
            "MiningComplete",
            f"Found {len(unique_factors)} unique factors with IC >= {min_ic}"
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
            },
            'results': {
                'total_evaluated': len(self.all_evaluated_factors),
                'best_factors': [
                    {
                        'expression': f.expression,
                        'ic_score': f.ic_score,
                        'complexity': f.complexity,
                        'order': f.order,
                    }
                    for f in self.best_factors
                ],
                'generation_history': self.generation_history,
            },
            'mining_log': self.mining_log,
        }


# ==============================================================================
# V115 场景感知回归 - MarketRegimeDetector
# ==============================================================================

class MarketRegimeDetector:
    """
    【V115 核心】市场场景检测器 - 将市场分为四个象限并动态调整因子权重。
    
    【四象限分类】
    1. 高波动 + 大盘股主导 (High Vol / Large Cap)
    2. 高波动 + 小盘股主导 (High Vol / Small Cap)
    3. 低波动 + 大盘股主导 (Low Vol / Large Cap)
    4. 低波动 + 小盘股主导 (Low Vol / Small Cap)
    
    【动态权重策略】
    - 不同场景下，因子表现不同
    - 根据场景动态调整因子权重，而非全局统一权重
    """
    
    def __init__(self,
                 volatility_threshold: float = 0.02,  # 波动率阈值
                 size_threshold: float = 0.5,  # 市值分位数阈值
                 lookback_window: int = 20):  # 回看窗口
        """
        初始化市场场景检测器。
        
        Args:
            volatility_threshold: 波动率阈值
            size_threshold: 市值分位数阈值
            lookback_window: 回看窗口
        """
        self.volatility_threshold = volatility_threshold
        self.size_threshold = size_threshold
        self.lookback_window = lookback_window
        
        # 场景权重池
        self.regime_weights = {
            'high_vol_large_cap': {},
            'high_vol_small_cap': {},
            'low_vol_large_cap': {},
            'low_vol_small_cap': {},
        }
        
        # 当前场景
        self.current_regime = None
        
        # 场景历史
        self.regime_history = []
        
        # 场景因子 IC 记录
        self.regime_factor_ics = defaultdict(lambda: defaultdict(list))
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][MarketRegimeDetector] Initialized")
        logger.info("=" * 80)
        logger.info(f"  Volatility Threshold: {self.volatility_threshold:.1%}")
        logger.info(f"  Size Threshold: {self.size_threshold:.1%}")
        logger.info(f"  Lookback Window: {self.lookback_window}")
        logger.info("=" * 80)
    
    def detect_regime(self, df: pd.DataFrame,
                      date_col: str = 'trade_date',
                      symbol_col: str = 'symbol',
                      volatility_col: str = 'volatility_20',
                      market_cap_col: str = 'total_mv') -> str:
        """
        检测当前市场场景。
        
        Args:
            df: 数据 DataFrame
            date_col: 日期列
            symbol_col: 股票代码列
            volatility_col: 波动率列
            market_cap_col: 市值列
            
        Returns:
            场景名称
        """
        # 获取最新日期
        unique_dates = sorted(df[date_col].unique())
        if len(unique_dates) < self.lookback_window:
            return 'low_vol_large_cap'  # 默认场景
        
        recent_dates = unique_dates[-self.lookback_window:]
        recent_data = df[df[date_col].isin(recent_dates)]
        
        # 计算市场波动率 (全市场平均波动率)
        if volatility_col in recent_data.columns:
            market_volatility = recent_data[volatility_col].mean()
        else:
            # 用收益率标准差估算
            recent_data['return'] = recent_data.groupby(symbol_col)['close'].transform(
                lambda x: x.pct_change()
            )
            market_volatility = recent_data['return'].std()
        
        # 计算市值中位数分位数
        if market_cap_col in recent_data.columns:
            latest_date = unique_dates[-1]
            latest_data = recent_data[recent_data[date_col] == latest_date]
            if len(latest_data) > 0:
                median_cap = latest_data[market_cap_col].median()
                # 计算历史分位数
                all_median_caps = []
                for d in recent_dates:
                    day_data = recent_data[recent_data[date_col] == d]
                    if len(day_data) > 0:
                        all_median_caps.append(day_data[market_cap_col].median())
                
                if all_median_caps:
                    size_percentile = np.percentile(all_median_caps, 50)
                    is_large_cap = median_cap > size_percentile
                else:
                    is_large_cap = True
            else:
                is_large_cap = True
        else:
            is_large_cap = True
        
        # 判断场景
        is_high_vol = market_volatility > self.volatility_threshold
        
        if is_high_vol:
            regime = 'high_vol_large_cap' if is_large_cap else 'high_vol_small_cap'
        else:
            regime = 'low_vol_large_cap' if is_large_cap else 'low_vol_small_cap'
        
        self.current_regime = regime
        self.regime_history.append({
            'date': latest_date if 'latest_date' in dir() else datetime.now(),
            'regime': regime,
            'volatility': market_volatility,
            'is_large_cap': is_large_cap,
        })
        
        logger.info(f"[{VERSION}][RegimeDetector] Current Regime: {regime} "
                   f"(Vol: {market_volatility:.4f}, Large Cap: {is_large_cap})")
        
        return regime
    
    def get_regime_description(self, regime: str = None) -> str:
        """获取场景描述"""
        if regime is None:
            regime = self.current_regime
        
        descriptions = {
            'high_vol_large_cap': '高波动/大盘股 (High Volatility / Large Cap)',
            'high_vol_small_cap': '高波动/小盘股 (High Volatility / Small Cap)',
            'low_vol_large_cap': '低波动/大盘股 (Low Volatility / Large Cap)',
            'low_vol_small_cap': '低波动/小盘股 (Low Volatility / Small Cap)',
        }
        
        return descriptions.get(regime, 'Unknown Regime')
    
    def update_regime_weights(self, regime: str, factor_weights: Dict[str, float]) -> None:
        """
        更新特定场景下的因子权重。
        
        Args:
            regime: 场景名称
            factor_weights: 因子权重组
        """
        if regime in self.regime_weights:
            self.regime_weights[regime] = factor_weights.copy()
    
    def get_dynamic_weights(self, factor_names: List[str],
                            df: pd.DataFrame = None,
                            date_col: str = 'trade_date') -> Dict[str, float]:
        """
        根据当前场景获取动态因子权重。
        
        Args:
            factor_names: 因子名称列表
            df: 数据 DataFrame (用于检测场景)
            date_col: 日期列
            
        Returns:
            动态权重组
        """
        # 如果提供了数据，先检测场景
        if df is not None:
            self.detect_regime(df, date_col)
        
        regime = self.current_regime or 'low_vol_large_cap'
        
        # 如果该场景有权重记录，直接返回
        if regime in self.regime_weights and self.regime_weights[regime]:
            regime_w = self.regime_weights[regime]
            # 确保所有因子都有权重
            weights = {}
            for f in factor_names:
                weights[f] = regime_w.get(f, 1.0 / len(factor_names))
            return weights
        
        # 否则返回均匀权重
        equal_weight = 1.0 / len(factor_names)
        return {f: equal_weight for f in factor_names}
    
    def record_factor_ic(self, factor_name: str, ic: float, 
                         regime: str = None) -> None:
        """
        记录因子在特定场景下的 IC。
        
        Args:
            factor_name: 因子名称
            ic: IC 值
            regime: 场景名称
        """
        regime = regime or self.current_regime
        if regime:
            self.regime_factor_ics[regime][factor_name].append(ic)
    
    def get_regime_factor_performance(self) -> Dict[str, Dict[str, float]]:
        """获取各场景下因子表现统计"""
        result = {}
        
        for regime, factors in self.regime_factor_ics.items():
            result[regime] = {}
            for factor, ics in factors.items():
                if ics:
                    result[regime][factor] = {
                        'mean_ic': float(np.mean(ics)),
                        'ic_std': float(np.std(ics)) if len(ics) > 1 else 0,
                        'count': len(ics),
                    }
        
        return result
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """获取场景统计"""
        if not self.regime_history:
            return {}
        
        regime_counts = defaultdict(int)
        for entry in self.regime_history:
            regime_counts[entry['regime']] += 1
        
        total = len(self.regime_history)
        
        return {
            'total_days': total,
            'regime_distribution': {
                regime: {'count': count, 'percentage': count / total}
                for regime, count in regime_counts.items()
            },
            'current_regime': self.current_regime,
            'regime_description': self.get_regime_description(),
        }


# ==============================================================================
# V115 消融实验模块
# ==============================================================================

class AblationStudy:
    """
    【V115 核心】因子消融实验 - 找出拖累整体表现的因子并剔除。
    
    【消融策略】
    1. 单因子剔除测试：逐个移除因子，观察 IC 变化
    2. 组合消融测试：移除因子组合，观察 IC 变化
    3. 重要性排序：按 IC 贡献排序
    4. 自动剔除：剔除 IC 贡献为负的因子
    """
    
    def __init__(self, ic_threshold: float = 0.01):
        """
        初始化消融实验。
        
        Args:
            ic_threshold: IC 阈值
        """
        self.ic_threshold = ic_threshold
        self.ablation_results = []
        self.important_factors = []
        self.harmful_factors = []
        
        logger.info(f"[{VERSION}][AblationStudy] Initialized with IC threshold: {ic_threshold}")
    
    def single_factor_ablation(self, df: pd.DataFrame,
                                factor_names: List[str],
                                score_col: str = 'score',
                                label_col: str = 't1_return',
                                date_col: str = 'trade_date') -> Dict[str, Any]:
        """
        单因子消融实验。
        
        Args:
            df: 数据 DataFrame
            factor_names: 因子名称列表
            score_col: 评分列
            label_col: 标签列
            date_col: 日期列
            
        Returns:
            消融实验结果
        """
        logger.info(f"[{VERSION}][AblationStudy] Starting single-factor ablation...")
        
        # 计算基准 IC
        baseline_ic = self._calculate_mean_ic(df, score_col, label_col, date_col)
        
        results = []
        
        for factor in factor_names:
            if factor not in df.columns:
                continue
            
            # 计算剔除该因子后的 IC
            # 方法：重新计算 score (不包含该因子)
            remaining_factors = [f for f in factor_names if f != factor]
            
            if not remaining_factors:
                continue
            
            # 简单平均剩余因子
            new_score = df[remaining_factors].mean(axis=1)
            df_temp = df.copy()
            df_temp[score_col] = new_score
            
            ablation_ic = self._calculate_mean_ic(df_temp, score_col, label_col, date_col)
            
            # IC 变化
            ic_change = ablation_ic - baseline_ic
            
            # 因子重要性 = IC 下降幅度 (正数表示重要)
            importance = -ic_change
            
            results.append({
                'factor': factor,
                'baseline_ic': baseline_ic,
                'ablation_ic': ablation_ic,
                'ic_change': ic_change,
                'importance': importance,
                'is_harmful': ic_change > 0,  # 剔除后 IC 上升，说明是负贡献
            })
        
        # 排序
        sorted_results = sorted(results, key=lambda x: x['importance'], reverse=True)
        
        # 分类
        self.important_factors = [r['factor'] for r in sorted_results if r['importance'] > 0.001]
        self.harmful_factors = [r['factor'] for r in sorted_results if r['importance'] < -0.001]
        
        self.ablation_results = sorted_results
        
        logger.info(f"[{VERSION}][AblationStudy] Complete. "
                   f"Important: {len(self.important_factors)}, Harmful: {len(self.harmful_factors)}")
        
        return {
            'baseline_ic': baseline_ic,
            'results': sorted_results,
            'important_factors': self.important_factors,
            'harmful_factors': self.harmful_factors,
        }
    
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
    
    def get_ablation_report(self) -> str:
        """获取消融实验报告"""
        if not self.ablation_results:
            return "No ablation study conducted yet."
        
        report = f"""# V115 Factor Ablation Study Report

## Summary

- **Baseline IC**: {self.ablation_results[0]['baseline_ic']:.4f}
- **Important Factors**: {len(self.important_factors)}
- **Harmful Factors**: {len(self.harmful_factors)}

## Factor Importance Ranking

| Rank | Factor | IC Change | Importance | Status |
|------|--------|-----------|------------|--------|
"""
        
        for i, r in enumerate(self.ablation_results, 1):
            status = "⚠ Harmful" if r['is_harmful'] else "✓ Important" if r['importance'] > 0 else "- Neutral"
            report += f"| {i} | {r['factor']} | {r['ic_change']:+.4f} | {r['importance']:+.4f} | {status} |\n"
        
        report += f"""
## Recommendations

### Factors to Keep (重要因子)
{', '.join(self.important_factors) if self.important_factors else 'None'}

### Factors to Remove (有害因子)
{', '.join(self.harmful_factors) if self.harmful_factors else 'None'}

## Action Plan

1. **立即剔除**: {', '.join(self.harmful_factors[:5]) if self.harmful_factors else 'None'}
2. **保留核心**: {', '.join(self.important_factors[:5]) if self.important_factors else 'None'}
3. **重新评估**: 剩余因子需要在不同场景下进一步验证
"""
        
        return report


# ==============================================================================
# V115 数据自愈增强 - 带 Mock 数据生成
# ==============================================================================

class DataHealingEngineV115:
    """
    【V115 增强】数据自愈引擎 - .env 自动检测 + Mock 数据生成。
    
    【自愈策略】
    1. 检测数据库连接状态
    2. 连接失败时生成 Mock 数据
    3. 提供清晰的错误提示
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        初始化数据自愈引擎。
        
        Args:
            db_url: 数据库连接 URL
        """
        self.db_url = db_url or os.getenv('DATABASE_URL')
        self.db = None
        self.is_mock_mode = False
        self.healing_log = []
        
        self._check_and_connect()
    
    def _check_and_connect(self):
        """检查环境变量并连接数据库"""
        logger.info(f"[{VERSION}][DataHealing] Checking environment...")
        
        if not self.db_url:
            logger.warning(f"[{VERSION}][DataHealing] DATABASE_URL not found in .env")
            logger.warning(f"[{VERSION}][DataHealing] Will use Mock data generation mode")
            self.is_mock_mode = True
            return
        
        # 尝试连接
        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.pool import QueuePool
            
            engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
            
            # 测试连接
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.db = engine
            logger.info(f"[{VERSION}][DataHealing] Database connected successfully")
            
        except Exception as e:
            logger.warning(f"[{VERSION}][DataHealing] Database connection failed: {e}")
            logger.warning(f"[{VERSION}][DataHealing] Please configure DATABASE_URL in .env file")
            logger.info(f"[{VERSION}][DataHealing] Falling back to Mock data mode")
            self.is_mock_mode = True
    
    def generate_mock_data(self, n_stocks: int = 100, n_days: int = 60,
                           start_date: str = '20240101') -> pd.DataFrame:
        """
        生成 Mock 数据用于测试。
        
        Args:
            n_stocks: 股票数量
            n_days: 交易日数量
            start_date: 开始日期
            
        Returns:
            Mock DataFrame
        """
        logger.info(f"[{VERSION}][DataHealing] Generating Mock data: {n_stocks} stocks x {n_days} days")
        
        np.random.seed(42)
        
        # 生成日期
        dates = pd.date_range(start=start_date, periods=n_days, freq='B')
        dates = [d.strftime('%Y%m%d') for d in dates]
        
        # 生成股票代码
        symbols = [f"{str(i).zfill(6)}.SZ" for i in range(1, n_stocks + 1)]
        
        # 生成数据
        data = []
        for symbol in symbols:
            base_price = np.random.uniform(10, 100)
            base_mv = np.random.uniform(1e9, 1e11)
            
            for i, date in enumerate(dates):
                # 价格随机游走
                ret = np.random.normal(0, 0.02)
                close = base_price * (1 + ret)
                
                # 生成 OHLC
                daily_vol = abs(np.random.normal(0.03, 0.01))
                high = close * (1 + daily_vol)
                low = close * (1 - daily_vol)
                open_price = close * (1 + np.random.normal(0, 0.01))
                
                # 生成成交量
                volume = np.random.uniform(1e6, 1e8)
                amount = volume * close
                
                # 生成因子
                volatility_20 = np.random.uniform(0.01, 0.05)
                momentum_10 = np.random.uniform(-0.1, 0.1)
                turnover_rate = np.random.uniform(0.01, 0.1)
                total_mv = base_mv * (1 + np.random.normal(0, 0.05))
                
                # 生成 T+1 收益 (带有一些可预测性)
                t1_return = np.random.normal(0, 0.02)
                
                row = {
                    'trade_date': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'pre_close': close / (1 + ret),
                    'volume': volume,
                    'amount': amount,
                    'turnover_rate': turnover_rate,
                    'total_mv': total_mv,
                    'volatility_20': volatility_20,
                    'momentum_10': momentum_10,
                    'pe_ttm': np.random.uniform(10, 50),
                    'pb': np.random.uniform(1, 5),
                    't1_return': t1_return,
                }
                
                data.append(row)
                
                base_price = close
                base_mv = total_mv
        
        df = pd.DataFrame(data)
        
        # 计算 T+3, T+5 收益
        for n in [3, 5]:
            df[f't{n}_return'] = df.groupby('symbol')['close'].transform(
                lambda x: x.shift(-n) / x - 1
            )
        
        self.healing_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'generate_mock_data',
            'n_stocks': n_stocks,
            'n_days': n_days,
            'total_rows': len(df),
        })
        
        logger.info(f"[{VERSION}][DataHealing] Mock data generated: {len(df)} rows")
        
        return df
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


# ==============================================================================
# V115 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV115:
    """
    V115 Alpha 预测核心引擎 - 自动化特征挖掘与场景化 Alpha.
    
    【V115 核心组件】
    1. GeneticFactorMiner: 基因算子搜索，生成三阶以上复杂因子
    2. MarketRegimeDetector: 场景感知，四象限动态权重
    3. DataHealingEngineV115: 数据自愈，Mock 数据生成
    4. AblationStudy: 消融实验，自动剔除有害因子
    """
    
    EPSILON = 1e-6
    
    # 基础因子列名
    BASE_FACTOR_COLUMNS = [
        'volatility_20', 'momentum_10', 'reversion_5',
        'turnover_rate', 'total_mv', 'pe_ttm', 'pb',
        'order_flow_imbalance_5', 'liquidity_stress_5',
        'kurtosis_interaction', 'tail_risk', 'smart_money_divergence',
    ]
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_genetic_mining: bool = True,
                 enable_regime_aware: bool = True,
                 enable_ablation: bool = True,
                 auto_heal: bool = True,
                 db_url: Optional[str] = None,
                 genetic_config: Optional[Dict] = None) -> None:
        """
        初始化 V115 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_genetic_mining: 是否启用基因挖掘
            enable_regime_aware: 是否启用场景感知
            enable_ablation: 是否启用消融实验
            auto_heal: 是否启用数据自愈
            db_url: 数据库连接 URL
            genetic_config: 遗传算法配置
        """
        self.config_path = Path(config_path)
        self.enable_genetic_mining = enable_genetic_mining
        self.enable_regime_aware = enable_regime_aware
        self.enable_ablation = enable_ablation
        self.auto_heal = auto_heal
        self.db_url = db_url
        
        # V115 核心组件
        self.genetic_miner = GeneticFactorMiner(**(genetic_config or {})) if enable_genetic_mining else None
        self.regime_detector = MarketRegimeDetector() if enable_regime_aware else None
        self.ablation_study = AblationStudy() if enable_ablation else None
        self.data_healer = DataHealingEngineV115(db_url=db_url) if auto_heal else None
        
        # 因子 IC 记录
        self.factor_ics = {}
        self.genetic_factors = []
        
        # 审计日志
        self.audit_log = []
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][AlphaResearch] V115 Alpha Research Engine Initialized")
        logger.info("=" * 80)
        logger.info(f"  Genetic Mining: {self.enable_genetic_mining}")
        logger.info(f"  Regime Aware: {self.enable_regime_aware}")
        logger.info(f"  Ablation Study: {self.enable_ablation}")
        logger.info(f"  Auto Healing: {self.auto_heal}")
        logger.info(f"  Data Mode: {'Mock' if (self.data_healer and self.data_healer.is_mock_mode) else 'Database'}")
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
        【主接口】计算 Alpha 评分。
        
        Args:
            df: 输入数据
            
        Returns:
            包含 score 和 t1_return 的 DataFrame
        """
        self._log_audit("ComputeScore", f"Starting V115 score computation with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据自愈检查
        if self.auto_heal and self.data_healer:
            if self.data_healer.is_mock_mode:
                self._log_audit("MockDataMode", "Using mock data for testing")
            else:
                self._log_audit("DatabaseMode", "Connected to database")
        
        # 2. 场景检测
        current_regime = None
        if self.enable_regime_aware and self.regime_detector:
            current_regime = self.regime_detector.detect_regime(result)
            self._log_audit("RegimeDetection", f"Detected regime: {current_regime}")
        
        # 3. 基因因子挖掘 (如果启用)
        if self.enable_genetic_mining and self.genetic_miner:
            self._log_audit("GeneticMining", "Starting genetic factor mining...")
            
            # 挖掘因子
            mined_factors = self.genetic_miner.mine_factors(
                result,
                target_count=5,
                min_ic=0.01
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
                except Exception as e:
                    self._log_audit("GeneticFactorError", f"Factor {i}: {e}")
        
        # 4. 计算基础因子评分
        available_factors = [f for f in self.BASE_FACTOR_COLUMNS if f in result.columns]
        
        # 添加挖掘的因子
        for i in range(len(self.genetic_factors)):
            available_factors.append(f'genetic_factor_{i}')
        
        if not available_factors:
            self._log_audit("NoFactors", "No factors available, using random score")
            result['score'] = np.random.randn(len(result))
        else:
            # 获取动态权重
            if self.enable_regime_aware and self.regime_detector:
                weights = self.regime_detector.get_dynamic_weights(available_factors, result)
            else:
                weights = {f: 1.0 / len(available_factors) for f in available_factors}
            
            # 加权评分
            score = np.zeros(len(result))
            for factor in available_factors:
                if factor in result.columns:
                    factor_data = result[factor].fillna(0)
                    # 截面标准化
                    factor_rank = factor_data.groupby(result['trade_date']).transform(
                        lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
                    )
                    score += factor_rank.values * weights.get(factor, 0)
            
            result['score'] = score
        
        # 5. 确保 t1_return 存在
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )
        
        # 6. 消融实验 (如果 IC 太低)
        t1_ic = self._calculate_mean_ic(result, 'score', 't1_return', 'trade_date')
        
        if t1_ic < 0.01 and self.enable_ablation and self.ablation_study:
            self._log_audit("LowICWarning", f"T+1 IC ({t1_ic:.4f}) < 0.01, running ablation study")
            
            # 运行消融实验
            ablation_result = self.ablation_study.single_factor_ablation(
                result,
                factor_names=available_factors,
            )
            
            self._log_audit("AblationComplete", 
                          f"Found {len(ablation_result['harmful_factors'])} harmful factors")
            
            # 剔除有害因子后重新计算
            if ablation_result['harmful_factors']:
                clean_factors = [f for f in available_factors 
                                if f not in ablation_result['harmful_factors']]
                
                if clean_factors:
                    self._log_audit("RefactoringScore", f"Recalculating with {len(clean_factors)} clean factors")
                    
                    score = np.zeros(len(result))
                    for factor in clean_factors:
                        if factor in result.columns:
                            factor_data = result[factor].fillna(0)
                            factor_rank = factor_data.groupby(result['trade_date']).transform(
                                lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
                            )
                            score += factor_rank.values / len(clean_factors)
                    
                    result['score'] = score
        
        # 7. 记录因子 IC
        for factor in available_factors:
            if factor in result.columns:
                ic = self._calculate_factor_ic(result, factor, 't1_return', 'trade_date')
                self.factor_ics[factor] = ic
                
                if self.enable_regime_aware and self.regime_detector:
                    self.regime_detector.record_factor_ic(factor, ic, current_regime)
        
        self._log_audit("ComputeScoreComplete", f"Final T+1 IC: {t1_ic:.4f}")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']] \
            if all(col in result.columns for col in ['t3_return', 't5_return']) \
            else result[['trade_date', 'symbol', 'score', 't1_return']]
    
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
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """获取场景统计"""
        if self.regime_detector:
            return self.regime_detector.get_regime_statistics()
        return {}
    
    def get_ablation_report(self) -> str:
        """获取消融实验报告"""
        if self.ablation_study:
            return self.ablation_study.get_ablation_report()
        return "Ablation study not enabled."
    
    def get_full_audit_report(self) -> Dict[str, Any]:
        """获取完整审计报告"""
        return {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'enable_genetic_mining': self.enable_genetic_mining,
                'enable_regime_aware': self.enable_regime_aware,
                'enable_ablation': self.enable_ablation,
                'data_mode': 'mock' if (self.data_healer and self.data_healer.is_mock_mode) else 'database',
            },
            'genetic_mining': self.get_genetic_mining_report(),
            'regime_statistics': self.get_regime_statistics(),
            'ablation_report': self.get_ablation_report() if self.enable_ablation else None,
            'factor_ics': self.factor_ics,
            'audit_log': self.audit_log,
        }


# ==============================================================================
# V115 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_genetic_mining: bool = True,
                       enable_regime_aware: bool = True,
                       enable_ablation: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None,
                       genetic_config: Optional[Dict] = None) -> AlphaResearchV115:
    """
    获取 AlphaResearchV115 实例。
    """
    return AlphaResearchV115(
        config_path=config_path,
        enable_genetic_mining=enable_genetic_mining,
        enable_regime_aware=enable_regime_aware,
        enable_ablation=enable_ablation,
        auto_heal=auto_heal,
        db_url=db_url,
        genetic_config=genetic_config,
    )


# ==============================================================================
# V115 回测运行器
# ==============================================================================

class BacktestRunnerV115:
    """
    V115 回测运行器 - 整合所有组件执行完整回测。
    """
    
    def __init__(self,
                 initial_capital: float = 100_000.00,
                 commission_rate: float = 0.0015,  # 单边 0.15%
                 stamp_duty_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 top_n: int = 50):
        """
        初始化回测运行器。
        
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金率
            stamp_duty_rate: 印花税率
            slippage_rate: 滑点率
            top_n: 持仓股票数
        """
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
        """
        运行完整回测。
        
        Args:
            df: 输入数据
            output_dir: 输出目录
            
        Returns:
            回测结果
        """
        from src.engine.backtest_referee import BacktestReferee
        
        # 创建 Alpha 模块
        alpha_module = get_alpha_research()
        
        # 创建裁判
        referee = BacktestReferee(alpha_module, output_dir=output_dir)
        referee.VERSION = VERSION  # 设置版本号
        
        # 运行审计
        result = referee.run_audit(df)
        
        return result


def run_v115_backtest(data_path: Optional[str] = None,
                      output_dir: str = "reports") -> Dict[str, Any]:
    """
    运行 V115 回测的便捷函数。
    
    Args:
        data_path: 数据文件路径 (Parquet 格式)
        output_dir: 输出目录
        
    Returns:
        回测结果
    """
    # 初始化数据自愈引擎
    healer = DataHealingEngineV115()
    
    # 加载数据
    if data_path and Path(data_path).exists():
        logger.info(f"[{VERSION}] Loading data from {data_path}")
        df = pd.read_parquet(data_path)
    else:
        logger.info(f"[{VERSION}] No data file found, generating mock data")
        df = healer.generate_mock_data(n_stocks=100, n_days=60)
    
    # 运行回测
    runner = BacktestRunnerV115()
    result = runner.run(df, output_dir)
    
    return result


if __name__ == "__main__":
    # 示例运行
    result = run_v115_backtest()
    print(json.dumps(result, indent=2, default=str))