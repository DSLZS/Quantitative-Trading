"""
Alpha Research Module - V128 ICIR 加权组合.

【V128 核心改进】
1. 使用 5 因子配置（基于 V126 最优结果）
2. ICIR 加权（IC/IC_Std）- 考虑因子稳定性
3. 目标：降低 IC Std，提高 IC IR

【V127 失败分析】
- 3 因子太少，IC 下降到 0.0424
- IC Std 反而上升到 0.2022
- 5 因子配置更优
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

VERSION = "V128"

# V128 基础因子池 - 精简版
BASE_FACTORS = [
    'pct_chg', 'change', 'momentum_5', 'momentum_20',
    'volatility_5', 'ma_deviation_5', 'ma_deviation_20',
    'price_position_20', 'price_position_60', 'bias_60',
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'turnover_bias_20', 'volume_shrink_ratio',
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist', 'hist_sharpe_20d',
]


class AlphaResearchV128:
    """V128 Alpha 研究引擎 - ICIR 加权组合"""
    
    EPSILON = 1e-6
    
    def __init__(self, ic_threshold: float = 0.025, n_factors: int = 5):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.factor_ics = {}
        self.factor_ic_stds = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Top-5 Factor Portfolio with ICIR Weighting")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  N Factors: {n_factors}")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_factor_ic_series(self, df: pd.DataFrame, factor_col: str) -> List[float]:
        """计算因子 IC 序列（按日期）"""
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
        
        return ics
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> Tuple[float, float]:
        """计算因子 IC 均值和标准差"""
        ics = self._calc_factor_ic_series(df, factor_col)
        
        if not ics:
            return 0.0, 0.0
        
        mean_ic = float(np.mean(ics))
        std_ic = float(np.std(ics)) if len(ics) > 1 else 1.0
        
        return mean_ic, std_ic
    
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
        
        # 2. 计算所有因子 IC 和 ICIR 并排序
        factor_metrics = []
        for factor in BASE_FACTORS:
            if factor not in result.columns:
                continue
            mean_ic, std_ic = self._calc_factor_ic(result, factor)
            icir = mean_ic / (std_ic + self.EPSILON) if std_ic > 0 else 0
            self.factor_ics[factor] = mean_ic
            self.factor_ic_stds[factor] = std_ic
            factor_metrics.append((factor, mean_ic, std_ic, icir))
        
        # 3. 按 ICIR 排序选择因子
        factor_metrics.sort(key=lambda x: abs(x[3]), reverse=True)
        
        # 4. 处理因子（翻转 + 标准化）
        factor_data = {}
        
        for factor, mean_ic, std_ic, icir in factor_metrics:
            if abs(mean_ic) < self.ic_threshold:
                continue
            if len(self.selected_factors) >= self.n_factors:
                break
                
            f_raw = result[factor].fillna(0)
            
            # 负 IC 因子翻转
            if mean_ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={mean_ic:.4f}, ICIR={icir:.3f} -> flipped")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={mean_ic:.4f}, ICIR={icir:.3f} -> kept")
            
            # 截面标准化
            f_std = f_processed.groupby(result['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
            )
            
            factor_data[factor] = f_std.values
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(BASE_FACTORS)} factors")
        
        # 5. ICIR 绝对值加权
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            # 获取调整后的 ICIR（考虑翻转）
            adjusted_icirs = []
            for factor in self.selected_factors:
                direction = self.factor_directions.get(factor, 1)
                icir = self.factor_ics[factor] / (self.factor_ic_stds[factor] + self.EPSILON)
                adjusted_icirs.append(abs(icir) * direction)
            
            # ICIR 绝对值加权
            total_icir = sum(abs(icir) for icir in adjusted_icirs)
            
            if total_icir > 0:
                weights = [abs(icir) / total_icir for icir in adjusted_icirs]
            else:
                weights = [1.0 / len(self.selected_factors)] * len(self.selected_factors)
            
            score = np.zeros(len(result))
            for i, factor in enumerate(self.selected_factors):
                score += factor_data[factor] * weights[i]
                self.factor_weights[factor] = weights[i]
            
            result['score'] = score
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (ICIR weighted)")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted


def get_alpha_research(ic_threshold: float = 0.025, n_factors: int = 5) -> AlphaResearchV128:
    return AlphaResearchV128(ic_threshold=ic_threshold, n_factors=n_factors)


def run_v128_backtest(data_path: str = "data/parquet/features_latest.parquet",
                      output_dir: str = "reports",
                      ic_threshold: float = 0.025,
                      n_factors: int = 5) -> Dict[str, Any]:
    """运行 V128 回测"""
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
    result = run_v128_backtest()
    print(json.dumps(result, indent=2, default=str))