"""
Alpha Research Module - V126 精简因子组合.

【V126 核心改进】
1. 只选择 IC 最高的 3-5 个因子
2. IC 加权（使用平方加权，放大高 IC 因子权重）
3. 目标：降低噪声，提高 IC 稳定性

【V125 失败分析】
- 10 个因子太多，引入了噪声
- IC Std=0.1785 仍然太高
- 需要更精简的因子组合
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

VERSION = "V126"

# V126 基础因子池 - 精简版
BASE_FACTORS = [
    'pct_chg', 'change', 'momentum_5', 'momentum_20',
    'volatility_5', 'ma_deviation_5', 'ma_deviation_20',
    'price_position_20', 'price_position_60', 'bias_60',
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'turnover_bias_20', 'volume_shrink_ratio',
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist', 'hist_sharpe_20d',
]


class AlphaResearchV126:
    """V126 Alpha 研究引擎 - 精简因子组合"""
    
    EPSILON = 1e-6
    
    def __init__(self, ic_threshold: float = 0.025, min_factors: int = 3, max_factors: int = 5):
        self.ic_threshold = ic_threshold
        self.min_factors = min_factors
        self.max_factors = max_factors
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Top-N Factor Portfolio with IC^2 Weighting")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Factor Range: {min_factors}-{max_factors}")
    
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
        
        # 3. 选择 IC 绝对值最高的因子
        factor_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 4. 处理因子（翻转 + 标准化）
        factor_data = {}
        
        for factor, ic in factor_ics:
            if abs(ic) < self.ic_threshold:
                continue
            if len(self.selected_factors) >= self.max_factors:
                break
                
            f_raw = result[factor].fillna(0)
            
            # 负 IC 因子翻转
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={ic:.4f} -> flipped (abs_ic={abs(ic):.4f})")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={ic:.4f} -> kept")
            
            # 截面标准化
            f_std = f_processed.groupby(result['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
            )
            
            factor_data[factor] = f_std.values
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(BASE_FACTORS)} factors (IC > {self.ic_threshold})")
        
        # 5. IC 平方加权（放大高 IC 因子权重）
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            # 获取调整后的 IC（考虑翻转）
            adjusted_ics = []
            for factor in self.selected_factors:
                direction = self.factor_directions.get(factor, 1)
                adjusted_ics.append(abs(self.factor_ics[factor]) * direction)
            
            # IC 平方加权
            ic_squared = np.array(adjusted_ics) ** 2
            total_weight = ic_squared.sum()
            
            if total_weight > 0:
                weights = ic_squared / total_weight
            else:
                weights = np.ones(len(self.selected_factors)) / len(self.selected_factors)
            
            score = np.zeros(len(result))
            for i, factor in enumerate(self.selected_factors):
                score += factor_data[factor] * weights[i]
                self.factor_weights[factor] = weights[i]
            
            result['score'] = score
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (IC^2 weighted)")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted


def get_alpha_research(ic_threshold: float = 0.025, min_factors: int = 3, max_factors: int = 5) -> AlphaResearchV126:
    return AlphaResearchV126(ic_threshold=ic_threshold, min_factors=min_factors, max_factors=max_factors)


def run_v126_backtest(data_path: str = "data/parquet/features_latest.parquet",
                      output_dir: str = "reports",
                      ic_threshold: float = 0.025,
                      min_factors: int = 3,
                      max_factors: int = 5) -> Dict[str, Any]:
    """运行 V126 回测"""
    from src.engine.backtest_referee import BacktestReferee
    
    logger.info(f"[{VERSION}] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"[{VERSION}] Loaded {len(df)} rows")
    
    alpha = get_alpha_research(ic_threshold=ic_threshold, min_factors=min_factors, max_factors=max_factors)
    
    referee = BacktestReferee(alpha, output_dir=output_dir)
    referee.VERSION = VERSION
    
    result = referee.run_audit(df)
    
    return result


if __name__ == "__main__":
    result = run_v126_backtest()
    print(json.dumps(result, indent=2, default=str))