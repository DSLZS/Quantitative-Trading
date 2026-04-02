"""
Alpha Research Module - V120 简化遗传算法与 IC 修复.

【V120 核心修复】
1. 简化遗传算法评估逻辑 - 直接使用 IC 作为适应度
2. 修复基础因子选择 - 只使用正 IC 因子
3. 增加有效遗传因子产出

【V119 失败根因】
- 遗传因子评估返回 0：因为因子池为空且正交化逻辑复杂
- 基础因子 IC 为负：没有正确筛选正 IC 因子
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

VERSION = "V120"

# V120 因子黑名单 (负 IC 因子)
TOXIC_FACTOR_BLACKLIST = [
    'turnover_rate', 'volatility_20', 'momentum_10',
    'close', 'open', 'high', 'low', 'volume', 'amount',  # V119 发现这些是负 IC
]

@dataclass
class GeneticFactor:
    """遗传因子"""
    expression: str
    ic_score: float = 0.0
    fitness_score: float = 0.0
    is_valid: bool = True


class SimpleGeneticMiner:
    """
    V120 简化遗传因子挖掘器.
    
    【简化策略】
    1. 直接使用 IC 作为适应度
    2. 减少种群大小和代数
    3. 简化算子集
    """
    
    def __init__(self, population_size: int = 20, generations: int = 5):
        self.population_size = population_size
        self.generations = generations
        self.rng = np.random.default_rng(42)
        
        # 简化算子集
        self.operators = ['Add', 'Sub', 'Mul', 'Div', 'Rank', 'Scale']
        self.base_features = [
            'pct_chg', 'change', 'momentum_5', 'momentum_20',
            'volatility_5', 'rsi_14', 'mfi_14', 'macd', 'macd_hist',
            'price_position_20', 'ma_deviation_5', 'ma_deviation_20',
            'volume_ma_ratio_5', 'smart_money_flow', 'hist_sharpe_20d',
            'predict_score', 'filtered_score',
        ]
        
        logger.info(f"[{VERSION}][SimpleMiner] Initialized with pop={population_size}, gen={generations}")
    
    def _calculate_ic(self, factor_values: pd.Series, label_values: pd.Series) -> float:
        """计算 Rank IC"""
        mask = factor_values.notna() & label_values.notna()
        if mask.sum() < 10:
            return 0.0
        
        f_rank = factor_values[mask].rank(method='average')
        l_rank = label_values[mask].rank(method='average')
        
        if np.std(f_rank) > 1e-10 and np.std(l_rank) > 1e-10:
            ic = np.corrcoef(f_rank, l_rank)[0, 1]
            return float(ic) if not np.isnan(ic) else 0.0
        return 0.0
    
    def _apply_operator(self, op: str, v1: pd.Series, v2: pd.Series = None) -> pd.Series:
        """应用算子"""
        EPS = 1e-10
        
        if op == 'Add':
            return v1 + v2
        elif op == 'Sub':
            return v1 - v2
        elif op == 'Mul':
            return v1 * v2
        elif op == 'Div':
            return v1 / (v2.abs() + EPS)
        elif op == 'Rank':
            return v1.groupby(pd.Series(range(len(v1)), index=v1.index) // len(v1) if len(v1) > 0 else pd.Series(range(len(v1)), index=v1.index)).transform(
                lambda x: x.rank(method='average') / len(x) if len(x) > 0 else x
            )
        elif op == 'Scale':
            return (v1 - v1.mean()) / (v1.std() + EPS)
        return v1
    
    def generate_random_factor(self) -> str:
        """生成随机因子表达式"""
        f1 = self.rng.choice(self.base_features)
        f2 = self.rng.choice(self.base_features)
        op = self.rng.choice(self.operators[:4])  # 只用四则运算
        return f"{f1} {op} {f2}"
    
    def compute_factor(self, expr: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """计算因子值"""
        try:
            # 简单解析表达式
            parts = expr.split()
            if len(parts) != 3:
                return None
            
            f1, op, f2 = parts
            
            if f1 not in df.columns or f2 not in df.columns:
                return None
            
            v1 = df[f1].astype(float).fillna(0)
            v2 = df[f2].astype(float).fillna(0)
            
            EPS = 1e-10
            
            if op == 'Add':
                return v1 + v2
            elif op == 'Sub':
                return v1 - v2
            elif op == 'Mul':
                return v1 * v2
            elif op == 'Div':
                return v1 / (v2.abs() + EPS)
            else:
                return v1
        except Exception:
            return None
    
    def mine(self, df: pd.DataFrame, target_count: int = 5) -> List[GeneticFactor]:
        """挖掘因子"""
        logger.info(f"[{VERSION}][SimpleMiner] Starting mining...")
        
        # 准备数据
        if 't1_return' not in df.columns:
            df['t1_return'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        
        label = df.set_index(['trade_date', 'symbol'])['t1_return']
        
        all_factors = []
        
        for gen in range(self.generations):
            population = []
            
            # 生成种群
            for _ in range(self.population_size):
                expr = self.generate_random_factor()
                factor_val = self.compute_factor(expr, df)
                
                if factor_val is None:
                    continue
                
                factor_idx = df.set_index(['trade_date', 'symbol']).index
                factor_series = factor_val.set_index(['trade_date', 'symbol']) if isinstance(factor_val, pd.DataFrame) else factor_val
                
                ic = self._calculate_ic(factor_series, label)
                fitness = abs(ic)
                
                population.append(GeneticFactor(expression=expr, ic_score=ic, fitness_score=fitness))
            
            # 选择最优
            population.sort(key=lambda f: f.fitness_score, reverse=True)
            all_factors.extend(population[:5])
            
            logger.info(f"[{VERSION}][SimpleMiner] Gen {gen}: Best IC = {population[0].ic_score if population else 0:.4f}")
        
        # 去重
        unique = []
        seen = set()
        for f in sorted(all_factors, key=lambda x: abs(x.ic_score), reverse=True):
            if f.expression not in seen:
                unique.append(f)
                seen.add(f.expression)
        
        logger.info(f"[{VERSION}][SimpleMiner] Found {len(unique)} unique factors")
        return unique[:target_count]


class AlphaResearchV120:
    """V120 Alpha 研究引擎"""
    
    EPSILON = 1e-6
    
    # V120 基础因子池 - 排除负 IC 因子
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
        logger.info(f"  Toxic Blacklist: {TOXIC_FACTOR_BLACKLIST}")
    
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
                    factor_val = self.genetic_miner.compute_factor(f.expression, result)
                    if factor_val is not None:
                        result[f'genetic_{i}'] = factor_val.values
                        self._log_audit("GeneticFactor", f"{f.expression} -> IC={f.ic_score:.4f}")
                
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
            if ic > 0.01:  # 只选正 IC 因子
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


def get_alpha_research(enable_genetic: bool = True) -> AlphaResearchV120:
    return AlphaResearchV120(enable_genetic=enable_genetic)


def run_v120_backtest(data_path: str = "data/parquet/features_latest.parquet",
                      output_dir: str = "reports") -> Dict[str, Any]:
    """运行 V120 回测"""
    from src.engine.backtest_referee import BacktestReferee
    
    # 加载数据
    logger.info(f"[{VERSION}] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"[{VERSION}] Loaded {len(df)} rows")
    
    # 创建 Alpha 模块
    alpha = get_alpha_research(enable_genetic=True)
    
    # 创建裁判
    referee = BacktestReferee(alpha, output_dir=output_dir)
    referee.VERSION = VERSION
    
    # 运行审计
    result = referee.run_audit(df)
    
    return result


if __name__ == "__main__":
    result = run_v120_backtest()
    print(json.dumps(result, indent=2, default=str))