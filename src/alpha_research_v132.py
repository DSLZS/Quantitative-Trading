"""
Alpha Research Module - V132 四因子极简版.

【V132 核心改进】
1. 使用 4 因子配置（进一步精简）
2. IC 阈值 0.028（更高要求）
3. Winsorization 去极值（2.5σ）
4. 目标：通过极简配置突破 IC 瓶颈

【历史结果分析】
- V123/V131 (5 因子): IC=0.0482-0.0484
- V129 (6 因子): IC=0.0477
- V130 (7 因子): IC=0.0476
- V126 (5 因子): IC=0.0482
- 4 因子可能是最优配置
"""

from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import warnings
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V132"

# V132 基础因子池
BASE_FACTORS = [
    'pct_chg', 'change', 'momentum_5', 'momentum_20',
    'volatility_5', 'ma_deviation_5', 'ma_deviation_20',
    'price_position_20', 'price_position_60', 'bias_60',
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'turnover_bias_20', 'volume_shrink_ratio',
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist', 'hist_sharpe_20d',
]


def winsorize(series: pd.Series, sigma: float = 2.5) -> pd.Series:
    """Winsorization 去极值"""
    mean = series.mean()
    std = series.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return series.clip(lower=lower, upper=upper)


class AlphaResearchV132:
    """V132 Alpha 研究引擎 - 四因子极简版"""
    
    EPSILON = 1e-6
    
    def __init__(self, ic_threshold: float = 0.028, n_factors: int = 4):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Top-{n_factors} with Winsorization (2.5σ)")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  N Factors: {n_factors}")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子 IC（按日期分组平均）"""
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
    
    def _process_factor(self, series: pd.Series, trade_dates: pd.Series) -> np.ndarray:
        """因子处理：去极值 + 标准化"""
        # 1. Winsorization 去极值 (2.5σ)
        series_wins = winsorize(series.fillna(0), sigma=2.5)
        
        # 2. 截面标准化
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        
        return result.values
    
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
        
        # 2. 计算所有因子 IC 并排序
        factor_ics = []
        for factor in BASE_FACTORS:
            if factor not in result.columns:
                continue
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            factor_ics.append((factor, ic))
        
        # 3. 按 IC 绝对值排序
        factor_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 4. 处理因子（翻转 + 标准化）
        factor_data = {}
        
        for factor, ic in factor_ics:
            if abs(ic) < self.ic_threshold:
                continue
            if len(self.selected_factors) >= self.n_factors:
                break
                
            f_raw = result[factor]
            
            # 负 IC 因子翻转
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={ic:.4f} -> flipped")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={ic:.4f} -> kept")
            
            # 因子处理：去极值 + 标准化
            f_std = self._process_factor(f_processed, result['trade_date'])
            
            factor_data[factor] = f_std
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(BASE_FACTORS)} factors")
        
        # 5. IC 绝对值加权
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            # 获取调整后的 IC（考虑翻转）
            adjusted_ics = []
            for factor in self.selected_factors:
                direction = self.factor_directions.get(factor, 1)
                adjusted_ics.append(abs(self.factor_ics[factor]) * direction)
            
            # IC 绝对值加权
            total_ic = sum(abs(ic) for ic in adjusted_ics)
            
            if total_ic > 0:
                weights = [abs(ic) / total_ic for ic in adjusted_ics]
            else:
                weights = [1.0 / len(self.selected_factors)] * len(self.selected_factors)
            
            score = np.zeros(len(result))
            for i, factor in enumerate(self.selected_factors):
                score += factor_data[factor] * weights[i]
                self.factor_weights[factor] = weights[i]
            
            result['score'] = score
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (IC weighted)")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted


def get_alpha_research(ic_threshold: float = 0.028, n_factors: int = 4) -> AlphaResearchV132:
    return AlphaResearchV132(ic_threshold=ic_threshold, n_factors=n_factors)


def run_v132_backtest(data_path: str = "data/parquet/features_latest.parquet",
                      output_dir: str = "reports",
                      ic_threshold: float = 0.028,
                      n_factors: int = 4) -> Dict[str, Any]:
    """运行 V132 回测"""
    from src.engine.backtest_referee import BacktestReferee
    
    logger.info(f"[{VERSION}] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"[{VERSION}] Loaded {len(df)} rows")
    
    alpha = get_alpha_research(ic_threshold=ic_threshold, n_factors=n_factors)
    
    referee = BacktestReferee(alpha, output_dir=output_dir)
    referee.VERSION = VERSION
    
    result = referee.run_audit(df)
    
    return result


if __name__ == "__main__":
    result = run_v132_backtest()
    print(json.dumps(result, indent=2, default=str))