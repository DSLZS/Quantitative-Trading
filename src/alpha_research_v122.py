"""
Alpha Research Module - V122 纯负翻转策略.

【V122 核心逻辑】
1. 只使用正 IC 因子
2. 对负 IC 因子进行物理翻转 (-1 * factor)
3. 简单加权平均作为最终评分

【V121 失败根因】
- 遗传因子评估 index 对齐问题导致 IC=0
- 没有正确处理负 IC 因子
"""

from typing import Any, Optional, Dict, List
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

VERSION = "V122"

# V122 基础因子池
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


class AlphaResearchV122:
    """V122 Alpha 研究引擎 - 纯负翻转策略"""
    
    EPSILON = 1e-6
    
    def __init__(self):
        self.factor_ics = {}
        self.factor_directions = {}  # 记录因子是否被翻转
        self.audit_log = []
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Negative IC Flip")
        logger.info(f"  Base Factors: {len(BASE_FACTOR_COLUMNS)}")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_ic(self, factor: pd.Series, label: pd.Series) -> float:
        """计算 Rank IC"""
        mask = factor.notna() & label.notna()
        if mask.sum() < 10:
            return 0.0
        
        f_rank = factor[mask].rank(method='average')
        l_rank = label[mask].rank(method='average')
        
        if np.std(f_rank) > 1e-10 and np.std(l_rank) > 1e-10:
            ic = np.corrcoef(f_rank, l_rank)[0, 1]
            return float(ic) if not np.isnan(ic) else 0.0
        return 0.0
    
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
        
        # 2. 计算各因子 IC
        available = [f for f in BASE_FACTOR_COLUMNS if f in result.columns]
        
        factor_data = {}  # 存储处理后的因子值
        positive_factors = []
        
        for factor in available:
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            
            f_raw = result[factor].fillna(0)
            
            # V122 核心：负 IC 因子物理翻转
            if ic < -0.005:
                # 负 IC 因子：翻转符号
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={ic:.4f} -> flipped")
            elif ic > 0.005:
                # 正 IC 因子：保持
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={ic:.4f} -> kept")
            else:
                # IC 接近 0 的因子：跳过
                self.factor_directions[factor] = 0
                continue
            
            # 截面标准化
            f_std = f_processed.groupby(result['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
            )
            
            factor_data[factor] = f_std.values
            positive_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(positive_factors)} factors (flipped {sum(1 for d in self.factor_directions.values() if d < 0)})")
        
        # 3. 计算综合评分
        if not positive_factors:
            result['score'] = np.random.randn(len(result))
        else:
            score = np.zeros(len(result))
            for factor in positive_factors:
                score += factor_data[factor]
            result['score'] = score / len(positive_factors)
        
        # 4. 验证总 IC
        total_ic = sum(self.factor_ics[f] * self.factor_directions.get(f, 1) for f in positive_factors)
        self._log_audit("TotalIC", f"Total IC after flip: {total_ic:.4f}")
        
        self._log_audit("Complete", f"Final score with {len(positive_factors)} factors")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        # 返回翻转后的 IC
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted


def get_alpha_research() -> AlphaResearchV122:
    return AlphaResearchV122()


def run_v122_backtest(data_path: str = "data/parquet/features_latest.parquet",
                      output_dir: str = "reports") -> Dict[str, Any]:
    """运行 V122 回测"""
    from src.engine.backtest_referee import BacktestReferee
    
    logger.info(f"[{VERSION}] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"[{VERSION}] Loaded {len(df)} rows")
    
    alpha = get_alpha_research()
    
    referee = BacktestReferee(alpha, output_dir=output_dir)
    referee.VERSION = VERSION
    
    result = referee.run_audit(df)
    
    return result


if __name__ == "__main__":
    result = run_v122_backtest()
    print(json.dumps(result, indent=2, default=str))