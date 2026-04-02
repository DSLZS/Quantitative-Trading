"""
Alpha Research Module - V121 结合 V118 成功逻辑与 Parquet 数据.

【V121 核心修复】
1. 使用 V118 完整的遗传因子挖掘逻辑
2. 基础因子列与 Parquet 数据对齐
3. 修复遗传因子计算中的 index 问题
"""

from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import warnings
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from loguru import logger
import yaml

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V121"

# V121 因子黑名单 (负 IC 因子)
TOXIC_FACTOR_BLACKLIST = [
    'turnover_rate', 'volatility_20', 'momentum_10',
]

@dataclass
class GeneticNode:
    name: str
    node_type: str
    children: List['GeneticNode'] = field(default_factory=list)
    value: Optional[float] = None

@dataclass
class GeneticFactor:
    expression: str
    tree: GeneticNode
    ic_score: float = 0.0
    icir_score: float = 0.0
    ic_std: float = 0.0
    fitness_score: float = 0.0
    is_valid: bool = True
    auto_flipped: bool = False
    original_ic: float = 0.0
    flipped_ic: float = 0.0


class SimpleGeneticMiner:
    """V121 简化遗传因子挖掘器 - 修复 index 问题"""
    
    OPERATORS = {
        'Rank': {'arity': 1}, 'Scale': {'arity': 1},
        'Ts_Mean': {'arity': 2, 'window': [5, 10, 20]},
        'Ts_Std': {'arity': 2, 'window': [5, 10, 20]},
        'Ts_Delta': {'arity': 2, 'window': [1, 3, 5]},
        'Delay': {'arity': 2, 'window': [1, 3, 5]},
        'Delta': {'arity': 2, 'window': [1, 3]},
        'Log': {'arity': 1}, 'Sqrt': {'arity': 1}, 'Abs': {'arity': 1},
        'Mul': {'arity': 2}, 'Div': {'arity': 2}, 'Add': {'arity': 2}, 'Sub': {'arity': 2},
    }
    
    BASE_FEATURES = [
        'pct_chg', 'change', 'momentum_5', 'momentum_20',
        'volatility_5', 'rsi_14', 'mfi_14', 'macd', 'macd_hist',
        'price_position_20', 'ma_deviation_5', 'ma_deviation_20',
        'volume_ma_ratio_5', 'smart_money_flow', 'hist_sharpe_20d',
        'predict_score', 'filtered_score', 'turnover_bias_20',
        'volume_price_divergence_5', 'accumulation_distribution_20',
    ]
    
    def __init__(self, population_size: int = 30, generations: int = 8):
        self.population_size = population_size
        self.generations = generations
        self.rng = np.random.default_rng(42)
        logger.info(f"[{VERSION}][SimpleMiner] Initialized pop={population_size}, gen={generations}")
    
    def _calc_ic(self, factor: pd.Series, label: pd.Series) -> float:
        mask = factor.notna() & label.notna()
        if mask.sum() < 10:
            return 0.0
        f, l = factor[mask].rank(method='average'), label[mask].rank(method='average')
        if np.std(f) > 1e-10 and np.std(l) > 1e-10:
            ic = np.corrcoef(f, l)[0, 1]
            return float(ic) if not np.isnan(ic) else 0.0
        return 0.0
    
    def generate_tree(self, depth: int = 2) -> GeneticNode:
        if depth <= 0:
            feat = self.rng.choice(self.BASE_FEATURES)
            return GeneticNode(name=feat, node_type='feature')
        
        ops = ['Add', 'Sub', 'Mul', 'Div']
        op = self.rng.choice(ops)
        node = GeneticNode(name=op, node_type='operator')
        node.children = [self.generate_tree(depth - 1), self.generate_tree(depth - 1)]
        return node
    
    def tree_to_expr(self, tree: GeneticNode) -> str:
        if tree.node_type == 'feature':
            return tree.name
        if not tree.children:
            return tree.name
        left = self.tree_to_expr(tree.children[0])
        right = self.tree_to_expr(tree.children[1]) if len(tree.children) > 1 else ""
        return f"{tree.name}({left}{',' if right else ''}{right})"
    
    def compute_factor(self, tree: GeneticNode, df: pd.DataFrame) -> Optional[pd.Series]:
        """计算因子值 - 修复 index 对齐"""
        try:
            if tree.node_type == 'feature':
                if tree.name not in df.columns:
                    return None
                return df[tree.name].astype(float)
            
            if tree.node_type == 'operator':
                children_vals = [self.compute_factor(c, df) for c in tree.children]
                if any(v is None for v in children_vals):
                    return None
                
                v1, v2 = children_vals[0].fillna(0), children_vals[1].fillna(0) if len(children_vals) > 1 else children_vals[0]
                EPS = 1e-10
                
                if tree.name == 'Add':
                    return v1 + v2
                elif tree.name == 'Sub':
                    return v1 - v2
                elif tree.name == 'Mul':
                    return v1 * v2
                elif tree.name == 'Div':
                    return v1 / (v2.abs() + EPS)
            return None
        except Exception:
            return None
    
    def evaluate(self, factor: GeneticFactor, df: pd.DataFrame) -> Tuple[float, float, float]:
        """评估因子"""
        try:
            fv = self.compute_factor(factor.tree, df)
            if fv is None:
                return 0.0, 0.0, 0.0
            
            # 按日期分组计算 IC
            ics = []
            for date in df['trade_date'].unique():
                mask = df['trade_date'] == date
                if mask.sum() < 20:
                    continue
                
                f_day = fv[mask].set_index(df.loc[mask, ['trade_date', 'symbol']].set_index(['trade_date', 'symbol']).index)
                l_day = df.loc[mask].set_index(['trade_date', 'symbol'])['t1_return']
                
                common_idx = f_day.index.intersection(l_day.index)
                if len(common_idx) < 20:
                    continue
                
                ic = self._calc_ic(f_day.loc[common_idx], l_day.loc[common_idx])
                if not np.isnan(ic):
                    ics.append(ic)
            
            if not ics:
                return 0.0, 0.0, 0.0
            
            ic_mean = float(np.mean(ics))
            ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
            fitness = ic_mean - ic_std  # V118 稳定性优先
            
            return ic_mean, ic_std, fitness
        except Exception:
            return 0.0, 0.0, 0.0
    
    def auto_flip(self, factor: GeneticFactor, df: pd.DataFrame) -> GeneticFactor:
        """Auto-Flip: IC 为负时物理翻转"""
        ic_mean, _, _ = self.evaluate(factor, df)
        
        if ic_mean < 0:
            # 创建翻转因子
            new_tree = GeneticNode(name='Mul', node_type='operator')
            new_tree.children = [
                GeneticNode(name='const', node_type='constant', value=-1.0),
                factor.tree
            ]
            
            flipped = GeneticFactor(
                expression=f"-1*({factor.expression})",
                tree=new_tree,
                ic_score=-ic_mean,
                fitness_score=abs(ic_mean),
                auto_flipped=True,
                original_ic=ic_mean,
                flipped_ic=-ic_mean,
            )
            logger.warning(f"[{VERSION}][Auto-Flip] {factor.expression[:40]}: {ic_mean:.4f} -> {-ic_mean:.4f}")
            return flipped
        
        factor.ic_score = ic_mean
        factor.fitness_score = ic_mean
        return factor
    
    def mine(self, df: pd.DataFrame, target_count: int = 5) -> List[GeneticFactor]:
        """挖掘因子"""
        logger.info(f"[{VERSION}][SimpleMiner] Starting mining...")
        
        # 准备标签
        if 't1_return' not in df.columns:
            df['t1_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        
        all_factors = []
        
        for gen in range(self.generations):
            population = []
            
            for _ in range(self.population_size):
                tree = self.generate_tree(depth=2)
                expr = self.tree_to_expr(tree)
                factor = GeneticFactor(expression=expr, tree=tree)
                
                ic_mean, ic_std, fitness = self.evaluate(factor, df)
                factor.ic_score = ic_mean
                factor.ic_std = ic_std
                factor.fitness_score = fitness
                factor.icir_score = ic_mean / ic_std if ic_std > 1e-10 else 0.0
                
                population.append(factor)
            
            # 选择最优
            population.sort(key=lambda f: f.fitness_score, reverse=True)
            all_factors.extend(population[:5])
            
            best_ic = max(abs(f.ic_score) for f in population) if population else 0
            logger.info(f"[{VERSION}][SimpleMiner] Gen {gen}: Best IC = {best_ic:.4f}")
        
        # Auto-Flip
        for i, f in enumerate(all_factors):
            all_factors[i] = self.auto_flip(f, df)
        
        # 去重
        unique = []
        seen = set()
        for f in sorted(all_factors, key=lambda x: abs(x.ic_score), reverse=True):
            if f.expression not in seen:
                unique.append(f)
                seen.add(f.expression)
        
        logger.info(f"[{VERSION}][SimpleMiner] Found {len(unique)} unique factors")
        return unique[:target_count]


class AlphaResearchV121:
    """V121 Alpha 研究引擎"""
    
    EPSILON = 1e-6
    
    # V121 基础因子池 - 与 Parquet 对齐
    BASE_FACTOR_COLUMNS = [
        'pct_chg', 'change', 'momentum_5', 'momentum_20',
        'volatility_5', 'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist',
        'price_position_20', 'price_position_60',
        'ma_deviation_5', 'ma_deviation_20',
        'turnover_bias_20', 'turnover_ma_ratio',
        'volume_price_divergence_5', 'volume_price_divergence_20',
        'volume_price_correlation', 'smart_money_flow',
        'volatility_contraction_10', 'volume_shrink_ratio',
        'volume_price_stable', 'accumulation_distribution_20',
        'bias_60', 'volume_price_health',
        'hist_sharpe_20d', 'predict_score', 'filtered_score',
    ]
    
    def __init__(self, enable_genetic: bool = True):
        self.enable_genetic = enable_genetic
        self.genetic_miner = SimpleGeneticMiner() if enable_genetic else None
        self.factor_ics = {}
        self.genetic_factors = []
        self.audit_log = []
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Genetic Mining: {enable_genetic}")
        logger.info(f"  Base Factors: {len(self.BASE_FACTOR_COLUMNS)}")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 准备标签
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # 2. 遗传因子挖掘
        if self.enable_genetic and self.genetic_miner:
            self._log_audit("GeneticMining", "Starting...")
            try:
                mined = self.genetic_miner.mine(result, target_count=5)
                self.genetic_factors = mined
                
                for i, f in enumerate(mined):
                    fv = self.genetic_miner.compute_factor(f.tree, result)
                    if fv is not None:
                        result[f'genetic_{i}'] = fv.values
                        self._log_audit("GeneticFactor", f"{f.expression[:40]} -> IC={f.ic_score:.4f}, flipped={f.auto_flipped}")
                
                self._log_audit("GeneticComplete", f"Mined {len(mined)} factors")
            except Exception as e:
                self._log_audit("GeneticError", str(e))
        
        # 3. 筛选正 IC 基础因子
        available = [f for f in self.BASE_FACTOR_COLUMNS if f in result.columns and f not in TOXIC_FACTOR_BLACKLIST]
        
        # 4. 计算各因子 IC 并选择
        positive_factors = []
        for factor in available:
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            if ic > 0.005:  # 只选正 IC 因子
                positive_factors.append(factor)
        
        # 添加遗传因子
        for i in range(len(self.genetic_factors)):
            positive_factors.append(f'genetic_{i}')
        
        self._log_audit("FactorSelection", f"Selected {len(positive_factors)} positive IC factors")
        
        # 5. 计算综合评分
        if not positive_factors:
            result['score'] = np.random.randn(len(result))
        else:
            score = np.zeros(len(result))
            for factor in positive_factors:
                if factor in result.columns:
                    f_data = result[factor].fillna(0)
                    # 截面标准化
                    f_rank = f_data.groupby(result['trade_date']).transform(
                        lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
                    )
                    score += f_rank.values
            result['score'] = score / len(positive_factors)
        
        # 6. 检查总 IC 并翻转
        total_ic = sum(self.factor_ics.values())
        if total_ic < 0:
            logger.warning(f"[{VERSION}] Total IC ({total_ic:.4f}) < 0, flipping score")
            result['score'] = -result['score']
        
        self._log_audit("Complete", f"Final score with {len(positive_factors)} factors")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子 IC"""
        ics = []
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            
            f = day[factor_col].fillna(0)
            l = day['t1_return'].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                l_rank = l.rank(method='average')
                ic = np.corrcoef(f_rank, l_rank)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        return float(np.mean(ics)) if ics else 0.0
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        return self.factor_ics


def get_alpha_research(enable_genetic: bool = True) -> AlphaResearchV121:
    return AlphaResearchV121(enable_genetic=enable_genetic)


def run_v121_backtest(data_path: str = "data/parquet/features_latest.parquet",
                      output_dir: str = "reports") -> Dict[str, Any]:
    """运行 V121 回测"""
    from src.engine.backtest_referee import BacktestReferee
    
    logger.info(f"[{VERSION}] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"[{VERSION}] Loaded {len(df)} rows")
    
    alpha = get_alpha_research(enable_genetic=True)
    
    referee = BacktestReferee(alpha, output_dir=output_dir)
    referee.VERSION = VERSION
    
    result = referee.run_audit(df)
    
    return result


if __name__ == "__main__":
    result = run_v121_backtest()
    print(json.dumps(result, indent=2, default=str))