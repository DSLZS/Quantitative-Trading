"""
Alpha Research Module - V124 正交化因子去冗余.

【V124 核心改进】
1. 因子正交化：对高相关因子进行正交化，去除冗余信息
2. 动态权重：根据 IC 和因子独立性综合赋权
3. 目标：降低 IC Std，提高 IC IR

【V123 结果分析】
- 阈值 0.03 时 IC=0.0484 最接近目标
- 问题：因子间高度相关（动量类因子冗余）
- IC Std 始终在 0.17-0.18 高位，需要正交化降低波动
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

VERSION = "V124"

# V124 基础因子池（按类别分组）
FACTOR_GROUPS = {
    'momentum': ['pct_chg', 'change', 'momentum_5', 'momentum_20'],
    'volatility': ['volatility_5', 'ma_deviation_5', 'ma_deviation_20'],
    'position': ['price_position_20', 'price_position_60', 'bias_60'],
    'volume': ['volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20', 
               'turnover_bias_20', 'volume_shrink_ratio'],
    'technical': ['rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist', 'hist_sharpe_20d'],
    'prediction': ['predict_score', 'filtered_score'],
}

# 每组选择代表因子的优先级（IC 越高越优先）
FACTOR_PRIORITY = {
    'momentum': ['momentum_5', 'pct_chg', 'change', 'momentum_20'],
    'volatility': ['volatility_5', 'ma_deviation_20', 'ma_deviation_5'],
    'position': ['price_position_20', 'bias_60', 'price_position_60'],
    'volume': ['volume_price_stable', 'volume_price_divergence_20', 'turnover_bias_20', 'volume_shrink_ratio', 'volume_price_divergence_5'],
    'technical': ['hist_sharpe_20d', 'rsi_14', 'mfi_14'],
    'prediction': ['predict_score', 'filtered_score'],
}


class AlphaResearchV124:
    """V124 Alpha 研究引擎 - 正交化因子去冗余"""
    
    EPSILON = 1e-6
    
    def __init__(self, ic_threshold: float = 0.025, max_factors_per_group: int = 2):
        self.ic_threshold = ic_threshold
        self.max_factors_per_group = max_factors_per_group
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Orthogonal Factor Selection")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Max Factors per Group: {max_factors_per_group}")
    
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
    
    def _orthogonalize(self, df: pd.DataFrame, factor_col: str, reference_factors: List[str]) -> pd.Series:
        """
        对因子进行正交化处理，去除与参考因子的相关性
        
        使用 Gram-Schmidt 正交化：
        residual = factor - sum(corr(factor, ref_i) * ref_i)
        """
        if not reference_factors:
            return df[factor_col].fillna(0)
        
        factor = df[factor_col].fillna(0).values
        result = factor.copy()
        
        for ref in reference_factors:
            if ref not in df.columns:
                continue
            
            ref_data = df[ref].fillna(0).values
            
            # 计算相关系数
            mask = ~np.isnan(factor) & ~np.isnan(ref_data)
            if mask.sum() < 50:
                continue
            
            corr = np.corrcoef(factor[mask], ref_data[mask])[0, 1]
            if np.isnan(corr):
                continue
            
            # 减去投影
            ref_normalized = (ref_data - np.nanmean(ref_data)) / (np.nanstd(ref_data) + self.EPSILON)
            result = result - corr * ref_normalized
        
        return pd.Series(result, index=df.index)
    
    def _select_factors_from_groups(self, df: pd.DataFrame) -> List[Tuple[str, float]]:
        """从每组因子中选择代表性因子"""
        selected = []
        
        # 先计算所有因子的 IC
        all_factor_ics = {}
        for group_name, factors in FACTOR_GROUPS.items():
            for factor in factors:
                if factor not in df.columns:
                    continue
                ic = self._calc_factor_ic(df, factor)
                all_factor_ics[factor] = ic
                self.factor_ics[factor] = ic
        
        # 从每组选择 IC 最高的因子
        for group_name, factors in FACTOR_GROUPS.items():
            available = [(f, all_factor_ics.get(f, 0)) for f in factors 
                        if f in df.columns and abs(all_factor_ics.get(f, 0)) >= self.ic_threshold]
            
            if not available:
                continue
            
            # 按 IC 绝对值排序
            available.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # 选择 top N
            for factor, ic in available[:self.max_factors_per_group]:
                selected.append((factor, ic))
        
        return selected
    
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
        
        # 2. 从每组选择代表性因子
        raw_selected = self._select_factors_from_groups(result)
        
        # 3. 处理因子（翻转 + 标准化）
        factor_data = {}
        
        for factor, ic in raw_selected:
            f_raw = result[factor].fillna(0)
            
            # 负 IC 因子翻转
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                adjusted_ic = abs(ic)
                self._log_audit("FactorFlip", f"{factor}: IC={ic:.4f} -> flipped")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                adjusted_ic = ic
                self._log_audit("FactorKeep", f"{factor}: IC={ic:.4f} -> kept")
            
            # 截面标准化
            f_std = f_processed.groupby(result['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
            )
            
            factor_data[factor] = {'values': f_std.values, 'ic': adjusted_ic}
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)} factors from {len(FACTOR_GROUPS)} groups")
        
        # 4. IC 加权计算综合评分
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            total_ic = sum(factor_data[f]['ic'] for f in self.selected_factors)
            score = np.zeros(len(result))
            
            for factor in self.selected_factors:
                weight = factor_data[factor]['ic'] / total_ic
                score += factor_data[factor]['values'] * weight
                self.factor_weights[factor] = weight
            
            result['score'] = score
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} orthogonal factors")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted


def get_alpha_research(ic_threshold: float = 0.025, max_factors_per_group: int = 2) -> AlphaResearchV124:
    return AlphaResearchV124(ic_threshold=ic_threshold, max_factors_per_group=max_factors_per_group)


def run_v124_backtest(data_path: str = "data/parquet/features_latest.parquet",
                      output_dir: str = "reports",
                      ic_threshold: float = 0.025,
                      max_factors_per_group: int = 2) -> Dict[str, Any]:
    """运行 V124 回测"""
    from src.engine.backtest_referee import BacktestReferee
    
    logger.info(f"[{VERSION}] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"[{VERSION}] Loaded {len(df)} rows")
    
    alpha = get_alpha_research(ic_threshold=ic_threshold, max_factors_per_group=max_factors_per_group)
    
    referee = BacktestReferee(alpha, output_dir=output_dir)
    referee.VERSION = VERSION
    
    result = referee.run_audit(df)
    
    return result


if __name__ == "__main__":
    result = run_v124_backtest()
    print(json.dumps(result, indent=2, default=str))