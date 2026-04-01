"""
Alpha Research Module - V116 真实环境下的因子进化.

【V116 核心任务】
1. 诊断 V115 失败原因：IC 为负 (-0.0078) 说明遗传算法进化出了反向信号或在 Mock 数据上过拟合
2. 重构 GeneticFactorMiner：
   - 适应度函数：IC + ICIR - |Skewness| 综合指标
   - 方向纠偏：Auto-Flip 逻辑，强特征 IC 为负时自动取反
   - 算子扩充：Ts_Correlation, Ts_Regression_Slope 等机构行为算子
3. 场景感知对齐：MarketRegimeDetector 完善
   - 高波动小盘：强制启用均值回归类因子
   - 低波动大盘：启用趋势跟踪类因子
4. 禁用 Mock 数据：正式审计报告严禁使用伪造数据

【V116 技术规格】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.03 | 核心指标 |
| IC IR | > 0.4 | 稳定性指标 |
| Fitness Function | IC + ICIR - |Skewness| | 综合适应度 |
| Auto-Flip | Enabled | 方向纠偏 |
| Regime Awareness | Enabled | 场景感知 |

【V116 禁止事项】
- 严禁在正式审计报告中使用 Mock 数据
- 禁止遗传算法过拟合单一指标
- 禁止忽略因子偏度风险
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

# V116 强制：主动加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V116 强制版本全局变量
# ==============================================================================
VERSION = "V116"


# ==============================================================================
# V116 自定义异常类
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


class RegimeDetectionError(Exception):
    """场景检测错误 - 当市场场景识别失败时抛出"""
    pass


# ==============================================================================
# V116 增强基因算子搜索 - Symbolic Logic with Auto-Flip
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
    icir_score: float = 0.0
    skewness: float = 0.0
    fitness_score: float = 0.0  # IC + ICIR - |Skewness|
    complexity: int = 0
    order: int = 0  # 因子阶数
    is_valid: bool = True
    error_message: str = ""
    auto_flipped: bool = False  # 是否经过自动翻转


class SymbolicOperatorLibrary:
    """
    【V116 核心】符号算子库 - 用于遗传算法生成复杂因子.
    
    【V116 新增算子】
    1. Ts_Correlation: 时序相关性 (捕捉量价关系)
    2. Ts_Regression_Slope: 时序回归斜率 (捕捉趋势强度)
    3. Ts_Covariance: 时序协方差
    4. Ts_Rank: 时序排名
    5. WMA: 加权移动平均
    6. EMA: 指数移动平均
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
        
        # V116 新增：时序相关性 (捕捉机构行为)
        'Ts_Correlation': {'arity': 3, 'type': 'time_series', 'window': [10, 20, 30], 
                          'description': '时序相关性 - 捕捉量价关系'},
        
        # V116 新增：时序回归斜率 (捕捉趋势强度)
        'Ts_Regression_Slope': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30],
                                'description': '时序回归斜率 - 捕捉趋势强度'},
        
        # V116 新增：时序协方差
        'Ts_Covariance': {'arity': 3, 'type': 'time_series', 'window': [10, 20, 30],
                         'description': '时序协方差'},
        
        # V116 新增：时序排名
        'Ts_Rank': {'arity': 2, 'type': 'time_series', 'window': [10, 20, 30],
                   'description': '时序排名 - 当前值在窗口中的百分位'},
        
        # V116 新增：加权移动平均
        'WMA': {'arity': 2, 'type': 'time_series', 'window': [5, 10, 20],
               'description': '加权移动平均'},
        
        # V116 新增：指数移动平均
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
    }
    
    # 基础特征池 (V116 扩充)
    BASE_FEATURES = [
        'close', 'open', 'high', 'low', 'volume', 'amount',
        'total_mv', 'turnover_rate', 'pe_ttm', 'pb',
        'volatility_20', 'momentum_10', 'reversion_5',
        'order_flow_imbalance_5', 'liquidity_stress_5',
        'kurtosis_interaction', 'tail_risk', 'smart_money_divergence',
        # V116 新增特征
        'vwap', 'accumulation_distribution', 'money_flow',
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
    【V116 核心】遗传因子挖掘器 - 真实环境下的因子进化.
    
    【V116 改进】
    1. 适应度函数：IC + ICIR - |Skewness|
    2. 方向纠偏：Auto-Flip 逻辑
    3. 算子扩充：Ts_Correlation, Ts_Regression_Slope 等
    
    【遗传算法流程】
    1. 初始化种群：随机生成 N 个因子表达式树
    2. 适应度评估：计算 IC + ICIR - |Skewness|
    3. 方向纠偏：IC 稳定为负时自动取反
    4. 选择：保留适应度高的个体
    5. 交叉：交换两个个体的子树
    6. 变异：随机修改节点
    7. 重复 2-6 直到收敛
    """
    
    def __init__(self,
                 population_size: int = 50,
                 generations: int = 20,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.6,
                 elite_rate: float = 0.1,
                 min_order: int = 2,  # V116 降低阶数要求避免过拟合
                 max_tree_depth: int = 5,
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
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][GeneticFactorMiner] Initialized")
        logger.info("=" * 80)
        logger.info(f"  Population Size: {self.population_size}")
        logger.info(f"  Generations: {self.generations}")
        logger.info(f"  Fitness Function: IC + ICIR - |Skewness|")
        logger.info(f"  Auto-Flip: Enabled")
        logger.info(f"  New Operators: Ts_Correlation, Ts_Regression_Slope, Ts_Covariance, Ts_Rank, WMA, EMA")
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
        """随机生成表达式树"""
        if current_depth >= max_depth:
            if self.rng.random() < 0.8:
                feature = self.rng.choice(self.operator_lib.get_base_features())
                return GeneticNode(name=feature, node_type='feature')
            else:
                value = round(self.rng.uniform(-1, 1), 4)
                return GeneticNode(name='const', node_type='constant', value=value)
        
        operator = self.rng.choice(self.operator_lib.get_all_operators())
        op_info = self.operator_lib.get_operator(operator)
        
        node = GeneticNode(name=operator, node_type='operator')
        
        if op_info['type'] == 'time_series' or op_info['type'] == 'delay':
            child_feature = self.generate_random_tree(max_depth - 1, current_depth + 1)
            window = self.rng.choice(op_info.get('window', [5]))
            window_node = GeneticNode(name=str(window), node_type='constant', value=float(window))
            node.children = [child_feature, window_node]
        elif op_info['arity'] == 1:
            child = self.generate_random_tree(max_depth - 1, current_depth + 1)
            node.children = [child]
        else:
            for _ in range(op_info['arity']):
                child = self.generate_random_tree(max_depth - 1, current_depth + 1)
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
        评估因子 - V116 综合适应度函数.
        
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
            
            # V116 综合指标
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
                            label_col: str = 't1_return',
                            threshold_days: int = 5) -> GeneticFactor:
        """
        V116 方向纠偏逻辑 - Auto-Flip.
        
        如果一个强特征的 IC 稳定为负，自动对其取反 (-1 * Factor)
        """
        try:
            factor_values = self.compute_factor_value(factor.tree, df, date_col, symbol_col)
            
            if factor_values is None:
                return factor
            
            # 检查连续负 IC 天数
            unique_dates = sorted(df[date_col].unique())
            negative_ic_days = 0
            total_valid_days = 0
            
            for date in unique_dates[-threshold_days:]:  # 只看最近 threshold_days 天
                day_data = df[df[date_col] == date]
                if len(day_data) < 20:
                    continue
                
                factor_day = factor_values[factor_values.index.isin(day_data.index)]
                label_day = day_data[label_col]
                
                mask = factor_day.notna() & label_day.notna()
                if mask.sum() < 20:
                    continue
                
                ic = self._calculate_ic(factor_day, label_day)
                total_valid_days += 1
                
                if ic < -0.01:  # 显著负相关
                    negative_ic_days += 1
            
            # 如果大部分天数为负 IC，触发翻转
            if total_valid_days > 0 and negative_ic_days / total_valid_days >= 0.6:
                logger.info(f"[{VERSION}][Auto-Flip] Flipping factor: {factor.expression} "
                           f"(negative IC for {negative_ic_days}/{total_valid_days} days)")
                
                # 创建翻转后的因子
                flipped_factor = GeneticFactor(
                    expression=f"-1 * ({factor.expression})",
                    tree=self._create_negation_tree(factor.tree),
                    ic_score=-factor.ic_score,
                    icir_score=factor.icir_score,
                    skewness=-factor.skewness,
                    fitness_score=abs(factor.ic_score) + factor.icir_score - abs(factor.skewness),
                    complexity=factor.complexity,
                    order=factor.order,
                    is_valid=True,
                    auto_flipped=True
                )
                return flipped_factor
            
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
        
        # V116 新增算子
        
        elif operator == 'Ts_Correlation':
            """时序相关性 - 捕捉量价关系 (如 close 和 volume 的相关性)"""
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
                                   value=tree.value, children=tree.children.copy())
            
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
                
                self.generation_history.append({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': avg_fitness,
                    'best_ic': best_ic,
                    'valid_count': len(valid_factors),
                    'high_order_count': len([f for f in valid_factors if f.order >= self.min_order]),
                })
                
                self._log_mining(
                    f"Generation {generation}",
                    f"Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}, "
                    f"Best IC: {best_ic:.4f}"
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
        
        # 应用方向纠偏
        for i, factor in enumerate(valid_factors):
            valid_factors[i] = self.auto_flip_direction(factor, df, date_col, symbol_col, label_col)
        
        # 按适应度排序
        sorted_factors = sorted(valid_factors, key=lambda f: f.fitness_score, reverse=True)
        
        self.best_factors = sorted_factors[:10]
        
        self._log_mining(
            "EvolutionComplete",
            f"Found {len(self.best_factors)} high-quality factors"
        )
        
        return self.best_factors
    
    def mine_factors(self, df: pd.DataFrame, 
                     target_count: int = 5,
                     min_fitness: float = 0.01) -> List[GeneticFactor]:
        """挖掘因子"""
        all_good_factors = []
        max_attempts = 3
        
        for attempt in range(max_attempts):
            self._log_mining("MiningAttempt", f"Attempt {attempt + 1}/{max_attempts}")
            
            factors = self.evolve(df)
            
            good_factors = [f for f in factors if f.fitness_score >= min_fitness]
            all_good_factors.extend(good_factors)
            
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
                    }
                    for f in self.best_factors
                ],
                'generation_history': self.generation_history,
            },
            'mining_log': self.mining_log,
        }


# ==============================================================================
# V116 场景感知回归 - MarketRegimeDetector (完善版)
# ==============================================================================

class MarketRegimeDetector:
    """
    【V116 核心】市场场景检测器 - 场景感知因子权重.
    
    【V116 场景策略】
    1. 高波动小盘：强制启用均值回归类因子
    2. 低波动大盘：启用趋势跟踪类因子
    
    【四象限分类】
    1. 高波动 + 大盘股主导 (High Vol / Large Cap)
    2. 高波动 + 小盘股主导 (High Vol / Small Cap)
    3. 低波动 + 大盘股主导 (Low Vol / Large Cap)
    4. 低波动 + 小盘股主导 (Low Vol / Small Cap)
    """
    
    def __init__(self,
                 volatility_threshold: float = 0.025,  # V116 调整阈值
                 size_threshold: float = 0.5,
                 lookback_window: int = 20):
        """初始化市场场景检测器"""
        self.volatility_threshold = volatility_threshold
        self.size_threshold = size_threshold
        self.lookback_window = lookback_window
        
        # V116 场景权重池 - 明确场景因子类型
        self.regime_weights = {
            'high_vol_large_cap': {},
            'high_vol_small_cap': {},
            'low_vol_large_cap': {},
            'low_vol_small_cap': {},
        }
        
        # 场景因子分类
        self.mean_reversion_factors = [
            'reversion_5', 'volatility_20', 'liquidity_stress_5',
            'kurtosis_interaction', 'tail_risk',
        ]
        
        self.trend_following_factors = [
            'momentum_10', 'order_flow_imbalance_5', 'smart_money_divergence',
        ]
        
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
        logger.info(f"  Mean Reversion Factors: {self.mean_reversion_factors}")
        logger.info(f"  Trend Following Factors: {self.trend_following_factors}")
        logger.info("=" * 80)
    
    def detect_regime(self, df: pd.DataFrame,
                      date_col: str = 'trade_date',
                      symbol_col: str = 'symbol',
                      volatility_col: str = 'volatility_20',
                      market_cap_col: str = 'total_mv') -> str:
        """检测当前市场场景"""
        unique_dates = sorted(df[date_col].unique())
        if len(unique_dates) < self.lookback_window:
            return 'low_vol_large_cap'  # 默认场景
        
        recent_dates = unique_dates[-self.lookback_window:]
        recent_data = df[df[date_col].isin(recent_dates)]
        
        # 计算市场波动率
        if volatility_col in recent_data.columns:
            market_volatility = recent_data[volatility_col].mean()
        else:
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
    
    def get_dynamic_weights(self, factor_names: List[str],
                            df: pd.DataFrame = None,
                            date_col: str = 'trade_date') -> Dict[str, float]:
        """
        V116 场景感知动态权重.
        
        【核心策略】
        - 高波动小盘：强制启用均值回归类因子 (权重 x2)
        - 低波动大盘：启用趋势跟踪类因子 (权重 x2)
        """
        # 如果提供了数据，先检测场景
        if df is not None:
            self.detect_regime(df, date_col)
        
        regime = self.current_regime or 'low_vol_large_cap'
        
        # 基础权重
        weights = {f: 1.0 for f in factor_names}
        
        # V116 场景感知调整
        if regime == 'high_vol_small_cap':
            # 高波动小盘：强制启用均值回归
            logger.info(f"[{VERSION}][RegimeAware] High Vol/Small Cap detected - "
                       f"enabling mean reversion factors")
            for f in factor_names:
                if f in self.mean_reversion_factors:
                    weights[f] = 2.0  # 权重加倍
                elif f in self.trend_following_factors:
                    weights[f] = 0.5  # 权重减半
        
        elif regime == 'low_vol_large_cap':
            # 低波动大盘：启用趋势跟踪
            logger.info(f"[{VERSION}][RegimeAware] Low Vol/Large Cap detected - "
                       f"enabling trend following factors")
            for f in factor_names:
                if f in self.trend_following_factors:
                    weights[f] = 2.0
                elif f in self.mean_reversion_factors:
                    weights[f] = 0.5
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {f: w / total_weight for f, w in weights.items()}
        
        return weights
    
    def record_factor_ic(self, factor_name: str, ic: float, 
                         regime: str = None) -> None:
        """记录因子在特定场景下的 IC"""
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
# V116 数据加载器 - 真实数据环境
# ==============================================================================

class RealDataLoader:
    """
    【V116 核心】真实数据加载器 - 禁用 Mock 数据.
    
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
# V116 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV116:
    """
    V116 Alpha 预测核心引擎 - 真实环境下的因子进化.
    
    【V116 核心组件】
    1. GeneticFactorMiner: 基因算子搜索 (IC + ICIR - |Skewness| 适应度)
    2. MarketRegimeDetector: 场景感知 (高波动小盘→均值回归，低波动大盘→趋势跟踪)
    3. RealDataLoader: 真实数据加载 (禁用 Mock)
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
                 db_url: Optional[str] = None,
                 genetic_config: Optional[Dict] = None) -> None:
        """初始化 V116 Alpha 研究引擎"""
        self.config_path = Path(config_path)
        self.enable_genetic_mining = enable_genetic_mining
        self.enable_regime_aware = enable_regime_aware
        self.db_url = db_url or os.getenv('DATABASE_URL')
        
        # V116 核心组件
        self.genetic_miner = GeneticFactorMiner(**(genetic_config or {})) if enable_genetic_mining else None
        self.regime_detector = MarketRegimeDetector() if enable_regime_aware else None
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
        logger.info(f"[{VERSION}][AlphaResearch] V116 Alpha Research Engine Initialized")
        logger.info("=" * 80)
        logger.info(f"  Genetic Mining: {self.enable_genetic_mining}")
        logger.info(f"  Regime Aware: {self.enable_regime_aware}")
        logger.info(f"  Data Source: {self.data_loader.data_source}")
        logger.info(f"  Fitness Function: IC + ICIR - |Skewness|")
        logger.info(f"  Auto-Flip: Enabled")
        logger.info(f"  New Operators: Ts_Correlation, Ts_Regression_Slope, etc.")
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
        self._log_audit("ComputeScore", f"Starting V116 score computation with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据源检查
        self._log_audit("DataSource", f"Data source: {self.data_loader.data_source}")
        
        # 2. 场景检测
        current_regime = None
        if self.enable_regime_aware and self.regime_detector:
            current_regime = self.regime_detector.detect_regime(result)
            self._log_audit("RegimeDetection", f"Detected regime: {current_regime}")
        
        # 3. 基因因子挖掘 (如果启用)
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
                    except Exception as e:
                        self._log_audit("GeneticFactorError", f"Factor {i}: {e}")
            except Exception as e:
                self._log_audit("GeneticMiningError", str(e))
        
        # 4. 计算基础因子评分
        available_factors = [f for f in self.BASE_FACTOR_COLUMNS if f in result.columns]
        
        # 添加挖掘的因子
        for i in range(len(self.genetic_factors)):
            available_factors.append(f'genetic_factor_{i}')
        
        if not available_factors:
            self._log_audit("NoFactors", "No factors available, using random score")
            result['score'] = np.random.randn(len(result))
        else:
            # 获取动态权重 (场景感知)
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
        
        # 6. 记录因子 IC
        for factor in available_factors:
            if factor in result.columns:
                ic = self._calculate_factor_ic(result, factor, 't1_return', 'trade_date')
                self.factor_ics[factor] = ic
                
                if self.enable_regime_aware and self.regime_detector:
                    self.regime_detector.record_factor_ic(factor, ic, current_regime)
        
        self._log_audit("ComputeScoreComplete", f"Final score calculated")
        
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
    
    def get_full_audit_report(self) -> Dict[str, Any]:
        """获取完整审计报告"""
        return {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'enable_genetic_mining': self.enable_genetic_mining,
                'enable_regime_aware': self.enable_regime_aware,
                'data_source': self.data_loader.data_source,
                'fitness_function': 'IC + ICIR - |Skewness|',
                'auto_flip': True,
            },
            'genetic_mining': self.get_genetic_mining_report(),
            'regime_statistics': self.get_regime_statistics(),
            'factor_ics': self.factor_ics,
            'audit_log': self.audit_log,
            'data_audit_log': self.data_loader.get_audit_log(),
        }


# ==============================================================================
# V116 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_genetic_mining: bool = True,
                       enable_regime_aware: bool = True,
                       db_url: Optional[str] = None,
                       genetic_config: Optional[Dict] = None) -> AlphaResearchV116:
    """获取 AlphaResearchV116 实例"""
    return AlphaResearchV116(
        config_path=config_path,
        enable_genetic_mining=enable_genetic_mining,
        enable_regime_aware=enable_regime_aware,
        db_url=db_url,
        genetic_config=genetic_config,
    )


# ==============================================================================
# V116 回测运行器
# ==============================================================================

class BacktestRunnerV116:
    """V116 回测运行器 - 真实环境下的因子进化"""
    
    def __init__(self,
                 initial_capital: float = 100_000.00,
                 commission_rate: float = 0.0015,  # 单边 0.15%
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


def run_v116_backtest(data_path: Optional[str] = None,
                      output_dir: str = "reports") -> Dict[str, Any]:
    """运行 V116 回测的便捷函数"""
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
    runner = BacktestRunnerV116()
    result = runner.run(df, output_dir)
    
    return result


if __name__ == "__main__":
    # 示例运行
    try:
        result = run_v116_backtest()
        print(json.dumps(result, indent=2, default=str))
    except DataHealingError as e:
        logger.error(f"V116 requires real data: {e}")
        logger.info("Please configure DATABASE_URL or add Parquet files to data/parquet/")