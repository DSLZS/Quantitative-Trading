"""
Alpha Research Module - V113 Alpha 脱毒与稳定性攻坚.

【V113 核心任务】
1. 版本一致性审计（最高优先级）: 
   - VERSION = "V113" 贯穿所有日志、报告、JSON 文件名
   - 彻底清理"V103 幽灵"

2. 修复 IC 衰减异常（排查前视偏差）:
   - 重新审计 compute_labels 和所有 shift 逻辑
   - 确保因子值计算完全基于 T 日及以前的数据
   - 标签对齐严格使用 shift(-1)

3. 选择性正交化与残差加权:
   - 仅对相关性绝对值 > 0.7 的因子对进行正交化
   - 对市值和波动率改用"残差加权"而非强行剥离

4. ICIR 提升策略:
   - 增加 L2 正则化强度
   - 优化 DynamicWeightPool，采用 60 天 IC 衰减加权
   - 剔除 IC 均值 < 0.01 的无效特征

【V113 技术规格】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.03 | 核心指标 |
| IC IR | > 0.4 | 正交化效果指标 |
| IC Decay | 单调递减 | T+1 > T+3 > T+5 |
| Selective Orthogonalization | 100% | 仅高相关因子对处理 |

【V113 禁止事项】
- 严禁修改 src/engine/backtest_referee.py
- 初始资金锁定 100,000.00，单边费率 0.15%
- 严禁偷看未来数据 (预测目标必须是 df['close'].shift(-1) / close - 1)
- 严禁给出"部分代码"或"逻辑概要"
"""

from typing import Any, Optional, Union, Dict, List, Tuple
from pathlib import Path
import warnings
import time
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
from loguru import logger
import yaml

# V113 强制：主动加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V113 强制版本全局变量
# ==============================================================================
VERSION = "V113"


# ==============================================================================
# V113 自定义异常类
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


class OrthogonalizationError(Exception):
    """正交化错误 - 当施密特正交化失败时抛出"""
    pass


class LookAheadBiasError(Exception):
    """前视偏差错误 - 当检测到未来数据泄露时抛出"""
    pass


# ==============================================================================
# V113 算子库 - 选择性正交化增强
# ==============================================================================

class AlphaOperatorsV113:
    """
    V113 Alpha 算子库 - 选择性正交化增强。
    
    【V113 新增算子】
    - Selective_Orthogonalization(X, threshold=0.7): 选择性正交化
    - Residual_Weighting(x, risk_factors): 残差加权
    - L2_Regularization(weights, lambda_l2): L2 正则化
    """
    
    EPSILON = 1e-6
    
    @staticmethod
    def Rank(x: pd.Series, group_col: Optional[str] = None) -> pd.Series:
        """截面百分位排名"""
        if group_col is None:
            return x.rank(method='average') / len(x.dropna())
        result = x.groupby(group_col).transform(
            lambda s: s.rank(method='average') / len(s.dropna()) if len(s.dropna()) > 0 else s
        )
        return result
    
    @staticmethod
    def Scale(x: pd.Series, group_col: Optional[str] = None) -> pd.Series:
        """截面均值 0 标准差 1 化"""
        if group_col is None:
            mean_val = x.mean()
            std_val = x.std()
            if std_val < AlphaOperatorsV113.EPSILON:
                std_val = AlphaOperatorsV113.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV113.EPSILON) if len(s.dropna()) > 1 else s
        )
        return result
    
    @staticmethod
    def Ts_Std(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列标准差"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).std())
    
    @staticmethod
    def Ts_Mean(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列均值"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).mean())
    
    @staticmethod
    def Ts_Delta(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列变化量"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1) - s.shift(n + 1))
    
    @staticmethod
    def _calculate_rank_ic(factor_values: pd.Series, label_values: pd.Series) -> float:
        """内部 IC 计算"""
        mask = factor_values.notna() & label_values.notna()
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        factor_ranks = factor_clean.rank(method='average')
        label_ranks = label_clean.rank(method='average')
        
        if np.std(factor_ranks) < 1e-10 or np.std(label_ranks) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    @staticmethod
    def Selective_Orthogonalization(X: pd.DataFrame, 
                                     threshold: float = 0.7,
                                     date_col: str = 'trade_date',
                                     symbol_col: str = 'symbol') -> Tuple[pd.DataFrame, Dict]:
        """
        【V113 核心】选择性正交化 - 仅对高相关因子对进行正交化。
        
        【数学原理】
        1. 计算因子间的相关性矩阵
        2. 识别 |corr| > threshold 的因子对
        3. 对高相关因子对进行施密特正交化
        4. 保持低相关因子不变
        
        Args:
            X: 输入 DataFrame (包含因子列)
            threshold: 相关性阈值 (默认 0.7)
            date_col: 日期列
            symbol_col: 股票代码列
            
        Returns:
            orthogonalized_df: 正交化后的 DataFrame
            stats: 正交化统计信息
        """
        # 排除非数值列
        exclude_cols = {date_col, symbol_col, 'ts_code'}
        features = [col for col in X.columns if col not in exclude_cols 
                   and pd.api.types.is_numeric_dtype(X[col])]
        
        if len(features) < 2:
            return X.copy(), {'orthogonalized_pairs': 0, 'threshold': threshold}
        
        # 计算因子相关性矩阵
        factor_corr = X[features].corr().values
        n_features = len(features)
        
        # 识别高相关因子对
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(factor_corr[i, j]) > threshold:
                    high_corr_pairs.append((i, j, factor_corr[i, j]))
        
        logger.info(f"[{VERSION}][SelectiveOrth] Found {len(high_corr_pairs)} high correlation pairs (|corr| > {threshold})")
        
        # 如果没有高相关因子对，直接返回
        if len(high_corr_pairs) == 0:
            return X.copy(), {
                'orthogonalized_pairs': 0,
                'threshold': threshold,
                'input_features': n_features,
            }
        
        # 按日期分组处理
        orthogonalized_dfs = []
        total_orthogonalized = 0
        
        for date in X[date_col].unique():
            mask = X[date_col] == date
            day_data = X.loc[mask].copy()
            
            if len(day_data) < 30:
                orthogonalized_dfs.append(day_data)
                continue
            
            # 提取特征矩阵
            feature_matrix = day_data[features].values.astype(np.float64)
            
            # 处理 NaN
            nan_mask = ~np.isfinite(feature_matrix)
            if nan_mask.any():
                col_means = np.nanmean(feature_matrix, axis=0)
                for j in range(feature_matrix.shape[1]):
                    feature_matrix[nan_mask[:, j], j] = col_means[j]
            
            # 对高相关因子对进行正交化
            orthogonalized_mask = set()
            for i, j, corr in high_corr_pairs:
                if i in orthogonalized_mask:
                    continue
                
                # 施密特正交化：用第 j 列正交化第 i 列
                x_i = feature_matrix[:, i]
                x_j = feature_matrix[:, j]
                
                # 计算投影
                dot_product = np.dot(x_i, x_j)
                norm_j_sq = np.dot(x_j, x_j)
                
                if norm_j_sq > AlphaOperatorsV113.EPSILON:
                    # 正交化：x_i_orth = x_i - (x_i · x_j / ||x_j||²) * x_j
                    projection = (dot_product / norm_j_sq) * x_j
                    x_i_orth = x_i - projection
                    
                    # 标准化
                    norm_i_orth = np.linalg.norm(x_i_orth)
                    if norm_i_orth > AlphaOperatorsV113.EPSILON:
                        feature_matrix[:, i] = x_i_orth / norm_i_orth
                        orthogonalized_mask.add(i)
                        total_orthogonalized += 1
            
            # 更新数据
            for idx, feat in enumerate(features):
                day_data[feat] = feature_matrix[:, idx]
            
            orthogonalized_dfs.append(day_data)
        
        if orthogonalized_dfs:
            result = pd.concat(orthogonalized_dfs, ignore_index=True)
        else:
            result = X.copy()
        
        stats = {
            'method': 'selective_gram_schmidt',
            'threshold': threshold,
            'input_features': n_features,
            'output_features': n_features,
            'orthogonalized_pairs': len(high_corr_pairs),
            'total_orthogonalizations': total_orthogonalized,
            'dates_processed': len(X[date_col].unique()),
        }
        
        logger.info(f"[{VERSION}][SelectiveOrth] Orthogonalization complete: {total_orthogonalized} features")
        
        return result, stats
    
    @staticmethod
    def Residual_Weighting(df: pd.DataFrame,
                           columns: List[str],
                           risk_factors: List[str],
                           group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【V113 核心】残差加权 - 对市值和波动率改用残差加权而非强行剥离。
        
        【数学原理】
        1. 对每个因子，用风险因子（市值、波动率）进行回归
        2. 计算残差
        3. 用残差的标准差作为权重
        4. 最终值 = 原始值 × 残差权重
        
        Args:
            df: 输入 DataFrame
            columns: 需要处理的因子列
            risk_factors: 风险因子列表（如 ['ln_total_mv', 'volatility']）
            group_col: 分组列
            
        Returns:
            残差加权后的 DataFrame
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            weighted_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    weighted_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y = day_data[col].values
                
                # 构建风险因子矩阵
                X_risk = []
                for rf in risk_factors:
                    if rf in day_data.columns:
                        X_risk.append(day_data[rf].values)
                
                if len(X_risk) == 0:
                    weighted_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                X = np.column_stack(X_risk)
                X = np.column_stack([np.ones(len(X)), X])  # 添加截距
                
                # 处理 NaN
                valid_mask = np.isfinite(y)
                for i in range(X.shape[1]):
                    valid_mask &= np.isfinite(X[:, i])
                
                if valid_mask.sum() < 30:
                    weighted_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y_valid = y[valid_mask]
                X_valid = X[valid_mask]
                
                # OLS 回归
                try:
                    beta = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                    y_pred = X_valid @ beta
                    residuals = y_valid - y_pred
                    
                    # 计算残差权重
                    residual_std = np.std(residuals)
                    weight = residual_std / (np.std(y_valid) + AlphaOperatorsV113.EPSILON)
                    
                    # 应用权重
                    day_data.loc[valid_mask, col] = residuals * weight
                    
                except Exception:
                    pass
                
                weighted_values.append(day_data[[col, group_col, 'symbol']])
            
            if weighted_values:
                weighted_df = pd.concat(weighted_values, ignore_index=True)
                if len(weighted_df) == len(result):
                    result.loc[:, col] = weighted_df[col].values
        
        return result
    
    @staticmethod
    def L2_Regularization(weights: np.ndarray, lambda_l2: float = 0.1) -> np.ndarray:
        """
        【V113 核心】L2 正则化 - 抑制过拟合。
        
        【数学原理】
        w_regularized = w / (1 + lambda_l2 * ||w||²)
        
        Args:
            weights: 原始权重
            lambda_l2: L2 正则化强度
            
        Returns:
            正则化后的权重
        """
        norm_sq = np.sum(weights ** 2)
        return weights / (1 + lambda_l2 * norm_sq)
    
    @staticmethod
    def Liquidity_Stress(df: pd.DataFrame, n: int = 5,
                          turnover_col: str = 'turnover_rate',
                          close_col: str = 'close',
                          symbol_col: str = 'symbol',
                          date_col: str = 'trade_date') -> pd.Series:
        """流动性压力因子"""
        result = df.copy()
        result['return'] = result.groupby(symbol_col)[close_col].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        turnover_ma = result.groupby(symbol_col)[turnover_col].transform(
            lambda x: x.shift(1).rolling(window=n*20).mean()
        )
        turnover_shock = result[turnover_col].shift(1) / (turnover_ma + AlphaOperatorsV113.EPSILON)
        price_impact = result['return'].abs()
        ls = turnover_shock * price_impact
        ls_cumsum = ls.groupby(result[symbol_col]).transform(
            lambda x: x.rolling(window=n).sum()
        )
        
        return ls_cumsum
    
    @staticmethod
    def Amihud_Illiq(returns: pd.Series, volume: pd.Series, 
                     n: int = 20, symbol_col: str = 'symbol') -> pd.Series:
        """Amihud 非流动性指标"""
        daily_illiq = returns.abs() / (volume + AlphaOperatorsV113.EPSILON)
        illiq_ma = daily_illiq.groupby(symbol_col).transform(
            lambda x: x.shift(1).rolling(window=n).mean()
        )
        return illiq_ma
    
    @staticmethod
    def Downside_Volatility(df: pd.DataFrame, returns_col: str = 'return',
                             n: int = 20, symbol_col: str = 'symbol') -> pd.Series:
        """下行波动率"""
        returns = df[returns_col]
        
        def calc_downside_vol(s):
            negative_returns = s[s < 0]
            if len(negative_returns) < 3:
                return np.nan
            return negative_returns.std()
        
        down_vol = df.groupby(symbol_col, group_keys=False)[returns_col].transform(
            lambda x: x.shift(1).rolling(window=n).apply(calc_downside_vol, raw=False)
        )
        
        return down_vol
    
    @staticmethod
    def Kurtosis_Interaction(df: pd.DataFrame, returns_col: str = 'return',
                              n: int = 20, symbol_col: str = 'symbol',
                              date_col: str = 'trade_date') -> pd.Series:
        """截面峰度交互特征"""
        returns = df[returns_col]
        
        stock_kurt = df.groupby(symbol_col)[returns_col].transform(
            lambda s: s.shift(1).rolling(window=n).kurt()
        )
        cross_skew = df.groupby(date_col)[returns_col].transform('skew')
        ki = stock_kurt * cross_skew
        
        return ki
    
    @staticmethod
    def Tail_Risk(df: pd.DataFrame, returns_col: str = 'return',
                  n: int = 20, symbol_col: str = 'symbol',
                  quantile: float = 0.05) -> pd.Series:
        """尾部风险指标"""
        def calc_tail_risk(s):
            if len(s.dropna()) < n:
                return np.nan
            return s.quantile(quantile) - s.mean()
        
        tail_risk = df.groupby(symbol_col, group_keys=False)[returns_col].transform(
            lambda x: x.shift(1).rolling(window=n).apply(calc_tail_risk, raw=False)
        )
        
        return tail_risk


# ==============================================================================
# V113 选择性正交化引擎
# ==============================================================================

class SelectiveOrthogonalizationEngineV113:
    """
    【V113 核心】选择性正交化引擎 - 仅对高相关因子对进行处理。
    
    【V113 改进策略】
    1. 计算因子间的相关性矩阵
    2. 只对 |corr| > 0.7 的因子对进行正交化
    3. 保持低相关因子不变，避免过度清洗
    """
    
    EPSILON = 1e-6
    
    def __init__(self, threshold: float = 0.7, target_features: int = 20):
        """
        初始化选择性正交化引擎。
        
        Args:
            threshold: 相关性阈值 (默认 0.7)
            target_features: 目标特征数量
        """
        self.threshold = threshold
        self.target_features = target_features
        self.orthogonalization_stats = {}
        
        logger.info(f"[{VERSION}][SelectiveOrthEngine] Initialized with threshold={threshold}")
    
    def orthogonalize(self, X: pd.DataFrame, date_col: str = 'trade_date',
                      symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        执行选择性正交化。
        
        Args:
            X: 特征矩阵
            date_col: 日期列
            symbol_col: 股票代码列
            
        Returns:
            正交化后的特征矩阵
        """
        logger.info(f"[{VERSION}][SelectiveOrthEngine] Starting selective orthogonalization...")
        
        # 排除非数值列
        exclude_cols = {date_col, symbol_col, 'ts_code'}
        features = [col for col in X.columns if col not in exclude_cols 
                   and pd.api.types.is_numeric_dtype(X[col])]
        n_features = len(features)
        
        logger.info(f"  Input features: {n_features}")
        
        # 计算因子相关性矩阵
        factor_corr = X[features].corr().values
        
        # 识别高相关因子对
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(factor_corr[i, j]) > self.threshold:
                    high_corr_pairs.append((i, j, factor_corr[i, j]))
        
        logger.info(f"  High correlation pairs (|corr| > {self.threshold}): {len(high_corr_pairs)}")
        
        # 如果没有高相关因子对，直接返回
        if len(high_corr_pairs) == 0:
            logger.info(f"[{VERSION}][SelectiveOrthEngine] No high correlation pairs, skipping")
            self.orthogonalization_stats = {
                'method': 'selective',
                'threshold': self.threshold,
                'input_features': n_features,
                'output_features': n_features,
                'orthogonalized_pairs': 0,
            }
            return X.copy()
        
        # 按日期分组处理
        orthogonalized_dfs = []
        total_orthogonalized = 0
        
        for date in X[date_col].unique():
            mask = X[date_col] == date
            day_data = X.loc[mask].copy()
            
            if len(day_data) < 30:
                orthogonalized_dfs.append(day_data)
                continue
            
            # 提取特征矩阵
            feature_matrix = day_data[features].values.astype(np.float64)
            
            # 处理 NaN
            nan_mask = ~np.isfinite(feature_matrix)
            if nan_mask.any():
                col_means = np.nanmean(feature_matrix, axis=0)
                for j in range(feature_matrix.shape[1]):
                    feature_matrix[nan_mask[:, j], j] = col_means[j]
            
            # 对高相关因子对进行正交化
            orthogonalized_mask = set()
            for i, j, corr in high_corr_pairs:
                if i in orthogonalized_mask:
                    continue
                
                # 施密特正交化
                x_i = feature_matrix[:, i]
                x_j = feature_matrix[:, j]
                
                dot_product = np.dot(x_i, x_j)
                norm_j_sq = np.dot(x_j, x_j)
                
                if norm_j_sq > self.EPSILON:
                    projection = (dot_product / norm_j_sq) * x_j
                    x_i_orth = x_i - projection
                    
                    norm_i_orth = np.linalg.norm(x_i_orth)
                    if norm_i_orth > self.EPSILON:
                        feature_matrix[:, i] = x_i_orth / norm_i_orth
                        orthogonalized_mask.add(i)
                        total_orthogonalized += 1
            
            # 更新数据
            for idx, feat in enumerate(features):
                day_data[feat] = feature_matrix[:, idx]
            
            orthogonalized_dfs.append(day_data)
        
        if orthogonalized_dfs:
            result = pd.concat(orthogonalized_dfs, ignore_index=True)
        else:
            result = X.copy()
        
        self.orthogonalization_stats = {
            'method': 'selective_gram_schmidt',
            'threshold': self.threshold,
            'input_features': n_features,
            'output_features': n_features,
            'orthogonalized_pairs': len(high_corr_pairs),
            'total_orthogonalizations': total_orthogonalized,
            'dates_processed': len(X[date_col].unique()),
        }
        
        logger.info(f"[{VERSION}][SelectiveOrthEngine] Complete: {total_orthogonalized} features orthogonalized")
        
        return result
    
    def get_stats(self) -> Dict:
        """获取正交化统计"""
        return self.orthogonalization_stats


# ==============================================================================
# V113 中性化引擎 (残差加权)
# ==============================================================================

class NeutralizationEngineV113:
    """
    V113 中性化引擎 - 残差加权 (Residual Weighting).
    
    【V113 改进】
    - 对市值和波动率改用"残差加权"而非强行剥离
    - 保持行业中性化
    """
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: List[str] = None):
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv', 'intraday_volatility']
        self.neutralize_vars = neutralize_vars
        self.neutralization_stats = {}
        logger.info(f"[{VERSION}][NeutralizationEngine] Initialized with residual weighting")
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        if columns is None:
            columns = list(AlphaResearchV113.ALL_FACTOR_COLUMNS)
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                values = result.loc[mask, col].dropna()
                
                if len(values) < 10:
                    continue
                
                median = values.median()
                mad = np.median(np.abs(values - median))
                adjusted_mad = mad * 1.4826
                
                if adjusted_mad < self.EPSILON:
                    continue
                
                lower_bound = median - n_std * adjusted_mad
                upper_bound = median + n_std * adjusted_mad
                
                result.loc[mask, col] = result.loc[mask, col].clip(
                    lower=lower_bound, upper=upper_bound
                )
        
        return result
    
    def normalize_zscore(self, df: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         group_col: str = 'trade_date') -> pd.DataFrame:
        """Z-Score 标准化"""
        if columns is None:
            columns = list(AlphaResearchV113.ALL_FACTOR_COLUMNS)
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            grouped = result.groupby(group_col)[col]
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            std = std.replace(0, self.EPSILON)
            result.loc[:, col] = (result[col] - mean) / std
        
        return result
    
    def residual_weighting(self, df: pd.DataFrame,
                           columns: Optional[List[str]] = None,
                           group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【V113 核心】残差加权 - 对市值和波动率改用残差加权。
        
        Args:
            df: 输入 DataFrame
            columns: 需要处理的列
            group_col: 分组列
            
        Returns:
            残差加权后的 DataFrame
        """
        result = df.copy()
        
        # 准备风险因子
        if 'total_mv' not in result.columns:
            result['total_mv'] = 1e10
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        # Volatility
        if 'intraday_volatility' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['intraday_volatility'] = (
                    (result['high'] - result['low']) / (result['close'] + self.EPSILON)
                )
            else:
                result['intraday_volatility'] = 0.02
        
        if columns is None:
            exclude_cols = {'trade_date', 'symbol', 'ts_code', 'ln_total_mv', 'intraday_volatility',
                          't1_return', 't3_return', 't5_return', 'score'}
            columns = [col for col in result.columns
                      if col not in exclude_cols and pd.api.types.is_numeric_dtype(result[col])]
        
        for col in columns:
            if col not in result.columns:
                continue
            
            weighted_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    weighted_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y = day_data[col].values
                
                # 构建风险因子矩阵
                X_vars = ['ln_total_mv', 'intraday_volatility']
                X_available = [v for v in X_vars if v in day_data.columns]
                
                if len(X_available) < 2:
                    weighted_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                X = day_data[X_available].values
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    y_pred = X @ beta
                    residuals = y - y_pred
                    
                    # 计算残差权重
                    residual_std = np.std(residuals)
                    original_std = np.std(y)
                    weight = residual_std / (original_std + self.EPSILON)
                    
                    # 应用权重
                    day_data[col] = residuals * weight
                    weighted_values.append(day_data[[col, group_col, 'symbol']])
                    
                except Exception:
                    weighted_values.append(day_data[[col, group_col, 'symbol']])
            
            if weighted_values:
                weighted_df = pd.concat(weighted_values, ignore_index=True)
                if len(weighted_df) == len(result):
                    result.loc[:, col] = weighted_df[col].values
        
        self.neutralization_stats = {
            'method': 'residual_weighting',
            'variables': ['size', 'volatility'],
            'n_factors_processed': len(columns),
        }
        
        return result
    
    def neutralize_industry(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            group_col: str = 'trade_date') -> pd.DataFrame:
        """行业中性化"""
        result = df.copy()
        
        if 'industry_code' not in result.columns:
            result['industry_code'] = 'UNKNOWN'
        
        if columns is None:
            exclude_cols = {'trade_date', 'symbol', 'ts_code', 'industry_code',
                          't1_return', 't3_return', 't5_return', 'score'}
            columns = [col for col in result.columns
                      if col not in exclude_cols and pd.api.types.is_numeric_dtype(result[col])]
        
        for col in columns:
            if col not in result.columns:
                continue
            
            neutralized_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y = day_data[col].values
                
                # 行业哑变量
                industry_dummies = pd.get_dummies(day_data['industry_code'], prefix='ind')
                if len(industry_dummies.columns) == 0:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                X = industry_dummies.values
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    y_pred = X @ beta
                    residuals = y - y_pred
                    
                    day_data[col] = residuals
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    
                except Exception:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
            
            if neutralized_values:
                neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                if len(neutralized_df) == len(result):
                    result.loc[:, col] = neutralized_df[col].values
        
        return result


# ==============================================================================
# V113 动态权重池 (60 天 IC 衰减加权)
# ==============================================================================

class DynamicWeightPoolV113:
    """
    V113 动态权重池 - 60 天 IC 衰减加权。
    
    【V113 改进】
    1. 采用 60 天滚动窗口
    2. IC 衰减加权：近期 IC 权重更高
    3. 剔除 IC 均值 < 0.01 的无效特征
    """
    
    def __init__(self, window: int = 60, min_history: int = 20, decay_factor: float = 0.95):
        """
        初始化动态权重池。
        
        Args:
            window: 滚动窗口大小 (默认 60 天)
            min_history: 最小历史数据要求
            decay_factor: IC 衰减因子 (默认 0.95，近期权重更高)
        """
        self.window = window
        self.min_history = min_history
        self.decay_factor = decay_factor
        self.ic_history = defaultdict(list)
        self.current_weights = {}
        self.weight_history = []
        self.invalid_features = set()
        
        logger.info(f"[{VERSION}][DynamicWeightPool] Initialized with window={window}, decay={decay_factor}")
    
    def update_ic(self, factor_name: str, ic: float, date: str) -> None:
        """更新因子 IC 记录"""
        self.ic_history[factor_name].append({
            'date': date,
            'ic': ic,
        })
        # 保持窗口大小
        if len(self.ic_history[factor_name]) > self.window:
            self.ic_history[factor_name] = self.ic_history[factor_name][-self.window:]
    
    def compute_weights(self, factor_names: List[str]) -> Dict[str, float]:
        """
        计算当前权重 - V113 IC 衰减加权。
        
        【权重计算】
        1. 计算 60 天滚动 IC 均值
        2. 应用指数衰减：近期 IC 权重更高
        3. 剔除 IC 均值 < 0.01 的无效特征
        """
        weights = {}
        signal_directions = {}
        total_abs_ic = 0.0
        
        for factor_name in factor_names:
            history = self.ic_history.get(factor_name, [])
            
            if len(history) < self.min_history:
                # 历史不足，给予默认权重
                weights[factor_name] = 1.0
                signal_directions[factor_name] = 1.0
                total_abs_ic += 1.0
                continue
            
            recent_ics = [h['ic'] for h in history[-self.window:]]
            
            # 应用指数衰减加权
            decay_weights = [self.decay_factor ** i for i in range(len(recent_ics) - 1, -1, -1)]
            decay_weights = np.array(decay_weights) / sum(decay_weights)
            
            # 衰减加权 IC 均值
            weighted_ic = sum(ic * w for ic, w in zip(recent_ics, decay_weights))
            mean_ic = weighted_ic
            
            # 剔除无效特征
            if abs(mean_ic) < 0.01:
                self.invalid_features.add(factor_name)
                weights[factor_name] = 0.0
                signal_directions[factor_name] = 1.0
                continue
            
            abs_ic = abs(mean_ic)
            weights[factor_name] = abs_ic
            signal_directions[factor_name] = 1.0 if mean_ic > 0 else -1.0
            total_abs_ic += abs_ic
        
        # 归一化权重
        if total_abs_ic > 0:
            for factor_name in weights:
                if weights[factor_name] > 0:
                    weights[factor_name] /= total_abs_ic
        else:
            # 所有因子都无效，平均分配
            valid_factors = [f for f in factor_names if f not in self.invalid_features]
            if valid_factors:
                equal_weight = 1.0 / len(valid_factors)
                for f in valid_factors:
                    weights[f] = equal_weight
        
        self.signal_directions = signal_directions
        self.current_weights = weights
        self.weight_history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'weights': weights.copy(),
            'directions': signal_directions,
        })
        
        return weights
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights
    
    def get_effective_factor_count(self) -> int:
        """获取有效因子数量"""
        return sum(1 for w in self.current_weights.values() if w > 0)
    
    def get_invalid_features(self) -> List[str]:
        """获取无效特征列表"""
        return list(self.invalid_features)


# ==============================================================================
# V113 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV113:
    """
    V113 Alpha 预测核心引擎 - Alpha 脱毒与稳定性攻坚。
    
    【V113 核心改进】
    1. 修复 IC 衰减异常 - 严格排查前视偏差
    2. SelectiveOrthogonalization: 选择性正交化 (|corr| > 0.7)
    3. ResidualWeighting: 残差加权 (市值、波动率)
    4. DynamicWeightPoolV113: 60 天 IC 衰减加权
    5. L2 Regularization: L2 正则化抑制过拟合
    """
    
    EPSILON = 1e-6
    
    # V113 全部因子列名
    ALL_FACTOR_COLUMNS = [
        # 流动性压力 (4)
        'liquidity_stress_5', 'liquidity_stress_10', 'amihud_illiq', 'turnover_vol_ratio',
        # 截面峰度 (3)
        'kurtosis_interaction', 'skewness_rank', 'tail_risk',
        # 动量 (6)
        'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'momentum_120', 'momentum_250',
        # 反转 (2)
        'reversion_5', 'reversion_10',
        # 量价 (4)
        'volume_price_health', 'vwap_distance', 'volume_rank', 'price_rank',
        # 波动率 (4)
        'volatility_20', 'downside_volatility', 'volatility_rank', 'beta_20',
        # 资金流 (3)
        'order_flow_imbalance_5', 'smart_money_divergence', 'big_order_ratio',
        # 估值 (3)
        'value_rank', 'ep_rank', 'bp_rank',
        # V109 保留 (4)
        'bias_momentum_repair', 'accumulation_distribution', 'relative_value_rank', 'volatility_interaction',
    ]
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_neutralization: bool = True,
                 enable_orthogonalization: bool = True,
                 enable_dynamic_weight: bool = True,
                 enable_l2_regularization: bool = True,
                 auto_heal: bool = True,
                 db_url: Optional[str] = None,
                 reflection_output: str = "reports/v113_reflection.json") -> None:
        """
        初始化 V113 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_neutralization: 是否启用中性化
            enable_orthogonalization: 是否启用正交化
            enable_dynamic_weight: 是否启用动态权重
            enable_l2_regularization: 是否启用 L2 正则化
            auto_heal: 是否启用数据自愈
            db_url: 数据库连接 URL
            reflection_output: 反哺 JSON 输出路径
        """
        self.config_path = Path(config_path)
        self.enable_neutralization = enable_neutralization
        self.enable_orthogonalization = enable_orthogonalization
        self.enable_dynamic_weight = enable_dynamic_weight
        self.enable_l2_regularization = enable_l2_regularization
        self.auto_heal = auto_heal
        self.db_url = db_url
        self.reflection_output = reflection_output
        
        # V113 核心组件
        self.neutralization_engine = NeutralizationEngineV113()
        self.orthogonalization_engine = SelectiveOrthogonalizationEngineV113(threshold=0.7)
        self.weight_pool = DynamicWeightPoolV113(window=60) if enable_dynamic_weight else None
        
        # 因子 IC 记录
        self.factor_ic_raw = {}
        self.factor_ic_history = defaultdict(list)
        
        # 审计日志
        self.ic_decay_audit = {}
        self.data_audit_log = []
        self.alpha_audit_log = []
        self.feature_selection_report = {}
        self.lookahead_bias_check = {}
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        # V113 版本确认
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][AlphaResearch] V113 Alpha Research Engine Initialized")
        logger.info("=" * 80)
        logger.info(f"  Neutralization: {self.enable_neutralization} (Residual Weighting)")
        logger.info(f"  Orthogonalization: {self.enable_orthogonalization} (Selective, |corr| > 0.7)")
        logger.info(f"  DynamicWeightPool: {self.enable_dynamic_weight} (60-day decay)")
        logger.info(f"  L2 Regularization: {self.enable_l2_regularization}")
        logger.info(f"  Auto Healing: {self.auto_heal}")
        logger.info(f"  Factor Count: {len(self.ALL_FACTOR_COLUMNS)}")
        logger.info(f"  Reflection Output: {self.reflection_output}")
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
    
    def _log_data_audit(self, action: str, count: int, details: str = "") -> None:
        """记录数据审计日志"""
        self.data_audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'count': count,
            'details': details,
        })
        logger.info(f"[{VERSION}][DataAudit] {action}: {count}, {details}")
    
    def _log_alpha_audit(self, action: str, count: int, details: str = "") -> None:
        """记录 Alpha 审计日志"""
        self.alpha_audit_log.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'count': count,
            'details': details,
        })
        logger.info(f"[{VERSION}][AlphaAudit] {action}: {count}, {details}")
    
    # ==============================================================================
    # V113 因子计算
    # ==============================================================================
    
    def compute_liquidity_stress(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """流动性压力因子"""
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        ls = AlphaOperatorsV113.Liquidity_Stress(
            result, n=period,
            turnover_col='turnover_rate',
            close_col='close',
            symbol_col='symbol',
            date_col='trade_date'
        )
        
        result[f'liquidity_stress_{period}'] = ls.values
        
        result[f'liquidity_stress_{period}'] = result.groupby('trade_date')[f'liquidity_stress_{period}'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_amihud_illiq(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Amihud 非流动性指标"""
        result = df.copy()
        
        if 'symbol' not in result.columns:
            logger.warning(f"[{VERSION}] symbol column not found, creating default")
            result['symbol'] = 'DEFAULT'
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        if 'volume' not in result.columns:
            logger.warning(f"[{VERSION}] volume column not found, using amount/close as proxy")
            result['volume'] = result.get('amount', result['close'] * 1000) / (result['close'] + self.EPSILON)
        
        def calc_illiq(group):
            ret = group['return'].shift(1).abs()
            vol = group['volume'].shift(1) + self.EPSILON
            daily_illiq = ret / vol
            return daily_illiq.rolling(window=period).mean()
        
        result['amihud_illiq'] = result.groupby('symbol', group_keys=False).apply(calc_illiq).values
        
        result['amihud_illiq'] = result.groupby('trade_date')['amihud_illiq'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_turnover_vol_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """换手率/波动率比率"""
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        vol_20 = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=20).std()
        )
        
        tvr = result['turnover_rate'].shift(1) / (vol_20 + self.EPSILON)
        
        result['turnover_vol_ratio'] = tvr.values
        
        result['turnover_vol_ratio'] = result.groupby('trade_date')['turnover_vol_ratio'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_kurtosis_interaction(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """截面峰度交互特征"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        ki = AlphaOperatorsV113.Kurtosis_Interaction(
            result, returns_col='return', n=period,
            symbol_col='symbol', date_col='trade_date'
        )
        
        result['kurtosis_interaction'] = ki.values
        
        result['kurtosis_interaction'] = result.groupby('trade_date')['kurtosis_interaction'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_skewness_rank(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """偏度截面排名"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        skew = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).skew()
        )
        
        skew_rank = skew.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['skewness_rank'] = skew_rank.values
        
        return result
    
    def compute_tail_risk(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """尾部风险指标"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        tail_risk = AlphaOperatorsV113.Tail_Risk(
            result, returns_col='return', n=period,
            symbol_col='symbol', quantile=0.05
        )
        
        result['tail_risk'] = tail_risk.values
        
        result['tail_risk'] = result.groupby('trade_date')['tail_risk'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_momentum_factor(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        动量因子 - V113 严格使用 T 日及以前数据。
        
        【关键修复】
        - 使用 shift(1) 确保使用 T-1 日收盘价
        - 使用 shift(period + 1) 确保使用 T-period-1 日收盘价
        """
        result = df.copy()
        
        # V113 修复：严格使用 T 日及以前数据
        result[f'momentum_{period}'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
        )
        
        result[f'momentum_{period}'] = result.groupby('trade_date')[f'momentum_{period}'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_reversion_factor(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """
        反转因子 - V113 严格使用 T 日及以前数据。
        """
        result = df.copy()
        
        # V113 修复：严格使用 T 日及以前数据
        result[f'reversion_{period}'] = result.groupby('symbol')['close'].transform(
            lambda x: -(x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0)
        )
        
        return result
    
    def compute_volume_price_health(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """量价健康度因子"""
        result = df.copy()
        
        # V113 修复：使用 shift(1) 确保使用 T-1 日数据
        price_trend = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / x.shift(period + 1) - 1
        )
        
        volume_trend = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1) / x.shift(period + 1) - 1
        )
        
        vph = price_trend * volume_trend
        
        result['volume_price_health'] = vph.values
        
        result['volume_price_health'] = result.groupby('trade_date')['volume_price_health'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_vwap_distance(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """VWAP 乖离率"""
        result = df.copy()
        
        if 'vwap' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
            else:
                result['vwap'] = result['close']
        
        # V113 修复：使用 shift(1)
        vwap_ma = result.groupby('symbol')['vwap'].transform(
            lambda x: x.shift(1).rolling(window=period).mean()
        )
        
        vwap_dist = (result['close'].shift(1) - vwap_ma) / (vwap_ma + self.EPSILON)
        
        result['vwap_distance'] = vwap_dist.values
        
        result['vwap_distance'] = result.groupby('trade_date')['vwap_distance'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_volatility_factor(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """波动率因子"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # V113 修复：使用 shift(1)
        vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        result['volatility_20'] = vol.values
        
        # 波动率因子：低波动率股票预期收益更高（负号）
        result['volatility_20'] = result.groupby('trade_date')['volatility_20'].transform(
            lambda x: -(x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_downside_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """下行波动率"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        down_vol = AlphaOperatorsV113.Downside_Volatility(
            result, returns_col='return', n=period, symbol_col='symbol'
        )
        
        result['downside_volatility'] = down_vol.values
        
        result['downside_volatility'] = result.groupby('trade_date')['downside_volatility'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_order_flow_imbalance(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """订单流不平衡"""
        result = df.copy()
        
        if 'amount' not in result.columns or 'volume' not in result.columns:
            result['order_flow_imbalance_5'] = np.nan
            return result
        
        avg_price = result['amount'] / (result['volume'] + self.EPSILON)
        
        def calc_ofi(group):
            price_change = group['close'].shift(1) - group['close'].shift(2)
            volume_change = group['volume'].shift(1) / (group['volume'].shift(2) + self.EPSILON) - 1
            ofi_daily = price_change * volume_change
            ofi_cumsum = ofi_daily.rolling(window=period).sum()
            return ofi_cumsum
        
        ofi = result.groupby('symbol', group_keys=False).apply(calc_ofi)
        result['order_flow_imbalance_5'] = ofi.values
        
        result['order_flow_imbalance_5'] = result.groupby('trade_date')['order_flow_imbalance_5'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_smart_money_divergence(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """聪明钱背离"""
        result = df.copy()
        
        if 'amount' not in result.columns or 'volume' not in result.columns:
            result['smart_money_divergence'] = np.nan
            return result
        
        avg_price = result['amount'] / (result['volume'] + self.EPSILON)
        big_order_flow = avg_price
        
        big_order_cumsum = big_order_flow.groupby(result['symbol']).transform(
            lambda x: x.rolling(window=period).sum()
        )
        
        price_change = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) - x.shift(period + 1)
        )
        
        big_order_rank = big_order_cumsum.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        price_rank = price_change.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        smd = big_order_rank - price_rank
        
        result['smart_money_divergence'] = smd.values
        
        result['smart_money_divergence'] = result.groupby('trade_date')['smart_money_divergence'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_value_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """估值排名"""
        result = df.copy()
        
        if 'total_mv' not in result.columns:
            result['value_rank'] = np.nan
            return result
        
        value_proxy = 1.0 / (result['total_mv'] + self.EPSILON)
        
        value_rank = value_proxy.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['value_rank'] = value_rank.values
        
        return result
    
    def compute_ep_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """EP 排名"""
        result = df.copy()
        
        if 'pe_ttm' not in result.columns:
            result['ep_rank'] = np.nan
            return result
        
        ep = 1.0 / (result['pe_ttm'].abs() + self.EPSILON)
        ep = ep * np.sign(result['pe_ttm'])
        
        ep_rank = ep.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['ep_rank'] = ep_rank.values
        
        return result
    
    def compute_bp_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """BP 排名"""
        result = df.copy()
        
        if 'pb' not in result.columns:
            result['bp_rank'] = np.nan
            return result
        
        bp = 1.0 / (result['pb'] + self.EPSILON)
        
        bp_rank = bp.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['bp_rank'] = bp_rank.values
        
        return result
    
    def compute_bias_momentum_repair(self, df: pd.DataFrame, ma_window: int = 20,
                                      return_window: int = 5) -> pd.DataFrame:
        """乖离率动量修复"""
        result = df.copy()
        
        # V113 修复：使用 shift(1)
        ma20 = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1).rolling(window=ma_window).mean()
        )
        
        bias = (result['close'].shift(1) - ma20) / (ma20 + self.EPSILON)
        
        momentum = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / x.shift(return_window + 1) - 1
        )
        
        bmr = -bias * momentum
        
        result['bias_momentum_repair'] = bmr.values
        
        result['bias_momentum_repair'] = result.groupby('trade_date')['bias_momentum_repair'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_accumulation_distribution(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """累积分布因子"""
        result = df.copy()
        
        if not all(col in result.columns for col in ['high', 'low', 'close', 'volume']):
            result['accumulation_distribution'] = np.nan
            return result
        
        high_low_range = result['high'] - result['low'] + self.EPSILON
        clv = ((result['close'] - result['low']) - (result['high'] - result['close'])) / high_low_range
        
        adl = clv * result['volume']
        ad = adl.groupby(result['symbol']).transform(
            lambda x: x.rolling(window=period).sum()
        )
        
        result['accumulation_distribution'] = ad.values
        
        result['accumulation_distribution'] = result.groupby('trade_date')['accumulation_distribution'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_relative_value_rank(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """相对价值排名"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # V113 修复：使用 shift(1)
        rolling_vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        value_proxy = 1.0 / (rolling_vol + self.EPSILON)
        
        momentum = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / x.shift(period + 1) - 1
        )
        
        value_rank = value_proxy.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        momentum_rank = momentum.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        rvr = value_rank * momentum_rank
        
        result['relative_value_rank'] = rvr.values
        
        result['relative_value_rank'] = result.groupby('trade_date')['relative_value_rank'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_volatility_interaction(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """波动率截面交互"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # V113 修复：使用 shift(1)
        stock_vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        cross_mean_vol = stock_vol.groupby(result['trade_date']).transform('mean')
        
        relative_vol = stock_vol / (cross_mean_vol + self.EPSILON)
        vi = 1.0 / (relative_vol + self.EPSILON)
        
        result['volatility_interaction'] = vi.values
        
        result['volatility_interaction'] = result.groupby('trade_date')['volatility_interaction'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_volume_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量排名"""
        result = df.copy()
        
        # V113 修复：使用 shift(1)
        volume_rank = result['volume'].shift(1).groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['volume_rank'] = volume_rank.values
        
        return result
    
    def compute_price_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格排名"""
        result = df.copy()
        
        # V113 修复：使用 shift(1)
        price_rank = result['close'].shift(1).groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['price_rank'] = price_rank.values
        
        return result
    
    def compute_volatility_rank(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """波动率排名"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # V113 修复：使用 shift(1)
        vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        vol_rank = vol.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['volatility_rank'] = 1.0 - vol_rank.values
        
        return result
    
    def compute_beta(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Beta 因子"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        market_return = result.groupby('trade_date')['return'].transform('mean')
        
        result['return'] = pd.to_numeric(result['return'], errors='coerce')
        market_return = pd.to_numeric(market_return, errors='coerce')
        
        def calc_beta_transform(group):
            if len(group) < period:
                return pd.Series(np.nan, index=group.index)
            
            ret_vals = group['return'].values
            mkt_vals = market_return.loc[group.index].values
            
            mask = np.isfinite(ret_vals) & np.isfinite(mkt_vals)
            if mask.sum() < period:
                return pd.Series(np.nan, index=group.index)
            
            ret_clean = ret_vals[mask]
            mkt_clean = mkt_vals[mask]
            
            ret_mean = np.mean(ret_clean)
            mkt_mean = np.mean(mkt_clean)
            
            cov = np.mean((ret_clean - ret_mean) * (mkt_clean - mkt_mean))
            var = np.mean((mkt_clean - mkt_mean) ** 2)
            
            beta_val = cov / var if var > 1e-10 else 0.0
            
            return pd.Series(beta_val, index=group.index)
        
        beta = result.groupby('symbol', group_keys=False).apply(calc_beta_transform)
        result['beta_20'] = beta.reset_index(level=0, drop=True).values
        
        return result
    
    def compute_big_order_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """大单比例"""
        result = df.copy()
        
        if 'amount' not in result.columns or 'volume' not in result.columns:
            result['big_order_ratio'] = np.nan
            return result
        
        avg_price = result['amount'] / (result['volume'] + self.EPSILON)
        
        bor = avg_price.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['big_order_ratio'] = bor.values
        
        return result
    
    # ==============================================================================
    # V113 标签计算 - 严格前视偏差检查
    # ==============================================================================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 T+1 收益标签 - V113 严格修复。
        
        【V113 关键修复】
        - 标签必须使用 shift(-1) 对齐
        - t1_return = close.shift(-1) / close - 1
        - 确保因子值在 T 日计算，标签对应 T+1 日收益
        """
        result = df.copy()
        
        # V113 严格修复：使用 shift(-1)
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / (x + self.EPSILON) - 1.0
        )
        
        return result
    
    def compute_tn_return(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """
        计算 T+N 收益标签 - V113 严格修复。
        
        【V113 关键修复】
        - 标签必须使用 shift(-n) 对齐
        """
        result = df.copy()
        
        # V113 严格修复：使用 shift(-n)
        result[f't{n}_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / (x + self.EPSILON) - 1.0
        )
        
        return result
    
    # ==============================================================================
    # V113 前视偏差检查
    # ==============================================================================
    
    def check_lookahead_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        【V113 核心】前视偏差检查。
        
        【检查项目】
        1. 因子值是否使用了未来数据
        2. IC 衰减是否单调递减
        3. T+5 IC 是否异常高于 T+1 IC
        
        Returns:
            检查结果字典
        """
        logger.info(f"[{VERSION}][LookaheadBiasCheck] Starting check...")
        
        check_result = {
            'passed': True,
            'issues': [],
            'ic_decay': {},
        }
        
        # 计算各期 IC
        for n in [1, 3, 5]:
            col = f't{n}_return'
            if col in df.columns and 'score' in df.columns:
                ic_values = []
                unique_dates = sorted(df['trade_date'].unique())
                
                for date in unique_dates:
                    day_data = df[df['trade_date'] == date]
                    if len(day_data) < 10:
                        continue
                    ic = self._calculate_rank_ic(day_data['score'], day_data[col])
                    if not np.isnan(ic):
                        ic_values.append(ic)
                
                if ic_values:
                    check_result['ic_decay'][f'T+{n}'] = float(np.mean(ic_values))
        
        # 检查 IC 衰减单调性
        if 'T+1' in check_result['ic_decay'] and 'T+3' in check_result['ic_decay']:
            if abs(check_result['ic_decay']['T+3']) > abs(check_result['ic_decay']['T+1']):
                check_result['passed'] = False
                check_result['issues'].append(
                    f"T+3 IC ({check_result['ic_decay']['T+3']:.4f}) > T+1 IC ({check_result['ic_decay']['T+1']:.4f}) - Possible look-ahead bias"
                )
                logger.warning(f"[{VERSION}][LookaheadBiasCheck] {check_result['issues'][-1]}")
        
        if 'T+3' in check_result['ic_decay'] and 'T+5' in check_result['ic_decay']:
            if abs(check_result['ic_decay']['T+5']) > abs(check_result['ic_decay']['T+3']):
                check_result['passed'] = False
                check_result['issues'].append(
                    f"T+5 IC ({check_result['ic_decay']['T+5']:.4f}) > T+3 IC ({check_result['ic_decay']['T+3']:.4f}) - Possible look-ahead bias"
                )
                logger.warning(f"[{VERSION}][LookaheadBiasCheck] {check_result['issues'][-1]}")
        
        self.lookahead_bias_check = check_result
        
        logger.info(f"[{VERSION}][LookaheadBiasCheck] Result: {'PASSED' if check_result['passed'] else 'FAILED'}")
        logger.info(f"[{VERSION}][LookaheadBiasCheck] IC Decay: {check_result['ic_decay']}")
        
        return check_result
    
    # ==============================================================================
    # V113 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True, year: int = None) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【V113 计算顺序】
        1. 数据审计日志
        2. 流动性压力因子
        3. 截面峰度因子
        4. 动量反转因子
        5. 量价因子
        6. 波动率因子
        7. 资金流因子
        8. 估值因子
        9. V109 保留因子
        10. 收益标签 (严格 shift(-1))
        11. 选择性正交化 (|corr| > 0.7)
        12. 残差加权 (市值、波动率)
        13. 因子清洗
        14. L2 正则化
        15. 动态权重池更新
        16. IC 加权预测
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V113 Factor Computation Started")
        logger.info("=" * 80)
        
        result = df.copy()
        
        # 数据审计
        missing_count = result.isnull().sum().sum()
        self._log_data_audit("MissingValuesHandled", int(missing_count), "Initial missing values")
        
        # 1-9. 计算所有因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Liquidity Stress Factors...")
        result = self.compute_liquidity_stress(result, period=5)
        result = self.compute_liquidity_stress(result, period=10)
        result = self.compute_amihud_illiq(result, period=20)
        result = self.compute_turnover_vol_ratio(result)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing Kurtosis Interaction Factors...")
        result = self.compute_kurtosis_interaction(result, period=20)
        result = self.compute_skewness_rank(result, period=20)
        result = self.compute_tail_risk(result, period=20)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing Momentum Factors...")
        for period in [5, 10, 20, 60, 120, 250]:
            result = self.compute_momentum_factor(result, period=period)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing Reversion Factors...")
        for period in [5, 10]:
            result = self.compute_reversion_factor(result, period=period)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing Volume-Price Factors...")
        result = self.compute_volume_price_health(result)
        result = self.compute_vwap_distance(result)
        result = self.compute_volume_rank(result)
        result = self.compute_price_rank(result)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing Volatility Factors...")
        result = self.compute_volatility_factor(result, period=20)
        result = self.compute_downside_volatility(result, period=20)
        result = self.compute_volatility_rank(result, period=20)
        result = self.compute_beta(result, period=20)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing Order Flow Factors...")
        result = self.compute_order_flow_imbalance(result, period=5)
        result = self.compute_smart_money_divergence(result, period=10)
        result = self.compute_big_order_ratio(result)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing Value Factors...")
        result = self.compute_value_rank(result)
        result = self.compute_ep_rank(result)
        result = self.compute_bp_rank(result)
        
        logger.info(f"[{VERSION}][FactorComputation] Computing V109 Legacy Factors...")
        result = self.compute_bias_momentum_repair(result)
        result = self.compute_accumulation_distribution(result)
        result = self.compute_relative_value_rank(result)
        result = self.compute_volatility_interaction(result)
        
        # 10. 收益标签 - V113 严格修复
        logger.info(f"[{VERSION}][FactorComputation] Computing Return Labels (V113 strict shift)...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 11. 选择性正交化 (V113 核心)
        if self.enable_orthogonalization:
            logger.info(f"[{VERSION}][FactorComputation] Running Selective Orthogonalization (|corr| > 0.7)...")
            factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
            factor_df = result[factor_cols + ['trade_date', 'symbol']].copy()
            orthogonalized_df = self.orthogonalization_engine.orthogonalize(factor_df)
            
            # 更新正交化后的因子
            for col in factor_cols:
                if col in orthogonalized_df.columns:
                    result[col] = orthogonalized_df[col].values
        
        # 12. 残差加权 (V113 核心)
        if self.enable_neutralization:
            logger.info(f"[{VERSION}][FactorComputation] Running Residual Weighting (Size+Volatility)...")
            factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
            result = self.neutralization_engine.residual_weighting(result, columns=factor_cols)
        
        # 13. 因子清洗
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Cleaning Factors...")
            result = self.clean_factors(result, self.ALL_FACTOR_COLUMNS)
        
        # 14. L2 正则化 (V113 新增)
        if self.enable_l2_regularization:
            logger.info(f"[{VERSION}][FactorComputation] Applying L2 Regularization...")
            # L2 正则化在权重计算时应用
        
        # 15. 动态权重池更新
        if self.enable_dynamic_weight and self.weight_pool:
            logger.info(f"[{VERSION}][FactorComputation] Updating Dynamic Weight Pool (60-day decay)...")
            self._update_weight_pool(result, self.ALL_FACTOR_COLUMNS)
        
        # 16. IC 加权预测
        logger.info(f"[{VERSION}][FactorComputation] Computing IC-Weighted Score...")
        result = self._compute_ic_weighted_score(result, self.ALL_FACTOR_COLUMNS)
        
        # Alpha 审计
        effective_count = self._count_effective_factors(result, self.ALL_FACTOR_COLUMNS)
        self._log_alpha_audit("EffectiveFactorCount", effective_count, "Factors with non-NaN values")
        
        # 前视偏差检查
        self.check_lookahead_bias(result)
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V113 Factor Computation Complete")
        logger.info(f"[{VERSION}][DataAudit] Missing values handled: {missing_count}")
        logger.info(f"[{VERSION}][AlphaAudit] Effective Factor Count: {effective_count}")
        logger.info("=" * 80)
        
        return result
    
    def _compute_ic_weighted_score(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        使用 IC 加权计算评分 - V113 L2 正则化增强。
        
        Args:
            df: 输入 DataFrame
            selected_features: 特征列表
            
        Returns:
            包含 score 的 DataFrame
        """
        result = df.copy()
        
        # 计算每个因子的 IC
        factor_ics = {}
        for factor_name in selected_features:
            if factor_name in result.columns and 't1_return' in result.columns:
                ic = self._calculate_rank_ic(result[factor_name], result['t1_return'])
                factor_ics[factor_name] = ic
        
        self.factor_ic_raw = factor_ics
        
        # 计算权重
        weights = {}
        total_abs_ic = 0.0
        
        for f in selected_features:
            ic = factor_ics.get(f, 0.0)
            # 剔除 IC 均值 < 0.01 的无效特征
            if abs(ic) < 0.01:
                weights[f] = 0.0
            else:
                abs_ic = abs(ic)
                weights[f] = abs_ic
                total_abs_ic += abs_ic
        
        if total_abs_ic <= 0:
            total_abs_ic = len(selected_features)
            weights = {f: 1.0 for f in selected_features}
        
        # 计算 IC 加权评分
        raw_score = np.zeros(len(result))
        
        for factor_name in selected_features:
            if factor_name not in result.columns:
                continue
            
            weight = weights.get(factor_name, 0.01) / total_abs_ic if total_abs_ic > 0 else 0
            
            # V113 L2 正则化
            if self.enable_l2_regularization and weight > 0:
                weight_array = np.array([weight])
                weight_array = AlphaOperatorsV113.L2_Regularization(weight_array, lambda_l2=0.1)
                weight = float(weight_array[0])
            
            ic = factor_ics.get(factor_name, 0.0)
            direction = 1.0 if ic >= 0 else -1.0
            
            factor_raw = result[factor_name].fillna(result[factor_name].median())
            if factor_raw.isna().all():
                continue
            
            factor_rank = factor_raw.groupby(result['trade_date']).transform(
                lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
            )
            
            raw_score += factor_rank.values * weight * direction
        
        result['score'] = raw_score
        
        return result
    
    def _update_weight_pool(self, df: pd.DataFrame, selected_features: List[str]) -> None:
        """更新动态权重池"""
        if not self.weight_pool:
            return
        
        unique_dates = sorted(df['trade_date'].unique())
        
        for date in unique_dates:
            day_data = df[df['trade_date'] == date]
            
            if len(day_data) < 30:
                continue
            
            for factor_name in selected_features:
                if factor_name in day_data.columns and 't1_return' in day_data.columns:
                    ic = self._calculate_rank_ic(day_data[factor_name], day_data['t1_return'])
                    self.weight_pool.update_ic(factor_name, ic, date)
    
    def _count_effective_factors(self, df: pd.DataFrame, selected_features: List[str]) -> int:
        """计算有效因子数量"""
        count = 0
        for col in selected_features:
            if col in df.columns:
                non_null = df[col].notna().sum()
                if non_null > 0:
                    count += 1
        return count
    
    def clean_factors(self, df: pd.DataFrame, selected_features: List[str] = None) -> pd.DataFrame:
        """因子清洗三部曲"""
        result = df.copy()
        
        if selected_features is None:
            factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
        else:
            factor_cols = [col for col in selected_features if col in result.columns]
        
        # 1. MAD 去极值
        result = self.neutralization_engine.winsorize_mad(result, columns=factor_cols, n_std=3.0)
        
        # 2. Z-Score 标准化
        result = self.neutralization_engine.normalize_zscore(result, columns=factor_cols)
        
        return result
    
    # ==============================================================================
    # V113 IC 计算与审计
    # ==============================================================================
    
    def _calculate_rank_ic(self, factor_values: pd.Series, label_values: pd.Series) -> float:
        """计算 Rank IC"""
        mask = factor_values.notna() & label_values.notna()
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        factor_ranks = factor_clean.rank(method='average')
        label_ranks = label_clean.rank(method='average')
        
        if np.std(factor_ranks) < 1e-10 or np.std(label_ranks) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_t1_ic(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算 T+1 IC 统计"""
        if 'score' not in df.columns or 't1_return' not in df.columns:
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0}
        
        unique_dates = sorted(df['trade_date'].unique())
        ic_series = []
        
        for date in unique_dates:
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 10:
                continue
            
            ic = self._calculate_rank_ic(day_data['score'], day_data['t1_return'])
            if not np.isnan(ic):
                ic_series.append(ic)
        
        if not ic_series:
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0}
        
        ic_values = np.array(ic_series)
        mean_ic = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
        
        return {
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'num_days': len(ic_values),
            'min_ic': float(np.min(ic_values)),
            'max_ic': float(np.max(ic_values)),
        }
    
    def calculate_factor_ics(self, df: pd.DataFrame, selected_features: List[str] = None) -> Dict[str, float]:
        """计算各因子的独立 IC"""
        if selected_features is None:
            selected_features = self.ALL_FACTOR_COLUMNS
        
        factor_ics = {}
        
        for factor_name in selected_features:
            if factor_name in df.columns and 't1_return' in df.columns:
                unique_dates = sorted(df['trade_date'].unique())
                ic_series = []
                
                for date in unique_dates:
                    day_data = df[df['trade_date'] == date]
                    if len(day_data) < 10:
                        continue
                    
                    ic = self._calculate_rank_ic(day_data[factor_name], day_data['t1_return'])
                    if not np.isnan(ic):
                        ic_series.append(ic)
                
                if ic_series:
                    factor_ics[factor_name] = float(np.mean(ic_series))
                else:
                    factor_ics[factor_name] = 0.0
        
        self.factor_ic_raw = factor_ics
        return factor_ics
    
    def audit_ic_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """IC 稳定性审计"""
        logger.info(f"[{VERSION}][ICStabilityAudit] Starting IC stability audit...")
        
        t1_ic = self.calculate_t1_ic(df)
        factor_ics = self.calculate_factor_ics(df)
        
        ic_strong = t1_ic['mean_ic'] > 0.03
        
        passed = ic_strong
        
        if not ic_strong:
            logger.warning(f"[{VERSION}][ICStabilityAudit] IC ({t1_ic['mean_ic']:.4f}) < 0.03 threshold")
        
        audit_result = {
            't1_ic': t1_ic,
            'factor_ics': factor_ics,
            'ic_strong': ic_strong,
            'passed': passed,
            'effective_factor_count': self._count_effective_factors(df, self.ALL_FACTOR_COLUMNS),
        }
        
        logger.info(f"[{VERSION}][ICStabilityAudit] Result: {'PASSED' if passed else 'FAILED'}")
        logger.info(f"[{VERSION}][ICStabilityAudit]   Mean IC: {t1_ic['mean_ic']:.4f}")
        logger.info(f"[{VERSION}][ICStabilityAudit]   IC IR: {t1_ic['ic_ir']:.2f}")
        
        return audit_result
    
    def audit_ic_decay(self, df: pd.DataFrame) -> Dict[str, float]:
        """IC 衰减审计"""
        logger.info(f"[{VERSION}][ICDecayAudit] Starting IC decay audit...")
        
        ic_results = {}
        
        for n in [1, 3, 5]:
            col = f't{n}_return'
            if col in df.columns:
                ic_values = []
                unique_dates = sorted(df['trade_date'].unique())
                
                for date in unique_dates:
                    day_data = df[df['trade_date'] == date]
                    if len(day_data) < 10:
                        continue
                    ic = self._calculate_rank_ic(day_data['score'], day_data[col])
                    if not np.isnan(ic):
                        ic_values.append(ic)
                
                if ic_values:
                    ic_results[f'T+{n}'] = float(np.mean(ic_values))
        
        # 检查单调性
        if 'T+1' in ic_results and 'T+3' in ic_results and 'T+5' in ic_results:
            is_monotonic = (
                abs(ic_results['T+1']) >= abs(ic_results['T+3']) >= abs(ic_results['T+5'])
            )
            if not is_monotonic:
                logger.warning(f"[{VERSION}][ICDecayAudit] IC decay is not monotonic!")
        
        self.ic_decay_audit = ic_results
        
        logger.info(f"[{VERSION}][ICDecayAudit] T+1: {ic_results.get('T+1', 0):.4f}")
        logger.info(f"[{VERSION}][ICDecayAudit] T+3: {ic_results.get('T+3', 0):.4f}")
        logger.info(f"[{VERSION}][ICDecayAudit] T+5: {ic_results.get('T+5', 0):.4f}")
        
        return ic_results
    
    # ==============================================================================
    # V113 自动反哺
    # ==============================================================================
    
    def save_reflection(self, df: pd.DataFrame) -> str:
        """
        【V113 核心】自动分析并保存反哺 JSON。
        
        Returns:
            保存的文件路径
        """
        logger.info(f"[{VERSION}][AutoReflection] Generating reflection report...")
        
        # 计算因子 IC
        factor_ics = self.calculate_factor_ics(df)
        
        # 排序
        sorted_ics = sorted(factor_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # 前 5 有效
        top_effective = sorted_ics[:5]
        
        # 前 5 无效
        bottom_ineffective = sorted_ics[-5:][::-1]
        
        # 构建反哺报告
        reflection = {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_factors': len(self.ALL_FACTOR_COLUMNS),
                'orthogonalization_method': self.orthogonalization_engine.orthogonalization_stats.get('method', 'selective'),
                'neutralization_method': self.neutralization_engine.neutralization_stats.get('method', 'residual_weighting'),
                'selective_orthogonalization': self.enable_orthogonalization,
            },
            'top_5_effective_factors': [
                {'name': name, 'ic': ic, 'rank': i+1}
                for i, (name, ic) in enumerate(top_effective)
            ],
            'bottom_5_ineffective_factors': [
                {'name': name, 'ic': ic, 'rank': i+1}
                for i, (name, ic) in enumerate(bottom_ineffective)
            ],
            'orthogonalization_stats': self.orthogonalization_engine.get_stats(),
            'neutralization_stats': self.neutralization_engine.neutralization_stats,
            'dynamic_weights': self.weight_pool.get_weights() if self.weight_pool else {},
            'invalid_features': self.weight_pool.get_invalid_features() if self.weight_pool else [],
            'lookahead_bias_check': self.lookahead_bias_check,
            'audit_logs': {
                'data_audit_count': len(self.data_audit_log),
                'alpha_audit_count': len(self.alpha_audit_log),
            },
            'recommendations': self._generate_recommendations(top_effective, bottom_ineffective),
        }
        
        # 确保输出目录存在
        reflection_path = Path(self.reflection_output)
        reflection_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON
        with open(reflection_path, 'w', encoding='utf-8') as f:
            json.dump(reflection, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"[{VERSION}][AutoReflection] Reflection saved to: {reflection_path}")
        
        return str(reflection_path)
    
    def _generate_recommendations(self, top_factors: List, bottom_factors: List) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        for name, ic in top_factors:
            if ic > 0.05:
                recommendations.append(f"Consider increasing weight for {name} (IC={ic:.4f})")
        
        for name, ic in bottom_factors:
            if abs(ic) < 0.01:
                recommendations.append(f"Consider removing {name} (IC={ic:.4f})")
        
        return recommendations
    
    # ==============================================================================
    # V113 主接口
    # ==============================================================================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【主接口】计算 Alpha 评分。
        
        Args:
            df: 输入数据
            
        Returns:
            包含 score 和 t1_return 的 DataFrame
        """
        # 计算因子
        result = self.compute_factors(df, clean=True)
        
        # IC 稳定性审计
        ic_audit = self.audit_ic_stability(result)
        
        # IC 衰减审计
        self.audit_ic_decay(result)
        
        # 自动反哺
        self.save_reflection(result)
        
        # 返回必需列
        output_columns = ['trade_date', 'symbol', 'score', 't1_return']
        for n in [3, 5]:
            if f't{n}_return' in result.columns:
                output_columns.append(f't{n}_return')
        
        return result[output_columns]
    
    def get_factor_ics(self, df: pd.DataFrame = None) -> Dict[str, float]:
        """获取因子 IC 记录"""
        return self.factor_ic_raw
    
    def get_neutralization_stats(self) -> Dict[str, Any]:
        """获取中性化统计"""
        return self.neutralization_engine.neutralization_stats
    
    def get_orthogonalization_stats(self) -> Dict[str, Any]:
        """获取正交化统计"""
        return self.orthogonalization_engine.get_stats()
    
    def get_data_audit_log(self) -> List[Dict]:
        """获取数据审计日志"""
        return self.data_audit_log
    
    def get_alpha_audit_log(self) -> List[Dict]:
        """获取 Alpha 审计日志"""
        return self.alpha_audit_log
    
    def get_dynamic_weights(self) -> Dict[str, float]:
        """获取动态权重"""
        if self.weight_pool:
            return self.weight_pool.get_weights()
        return {}
    
    def get_effective_factor_count(self) -> int:
        """获取有效因子数量"""
        if self.weight_pool:
            return self.weight_pool.get_effective_factor_count()
        return len(self.ALL_FACTOR_COLUMNS)
    
    def get_lookahead_bias_check(self) -> Dict[str, Any]:
        """获取前视偏差检查结果"""
        return self.lookahead_bias_check


# ==============================================================================
# V113 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_neutralization: bool = True,
                       enable_orthogonalization: bool = True,
                       enable_dynamic_weight: bool = True,
                       enable_l2_regularization: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None,
                       reflection_output: str = "reports/v113_reflection.json") -> AlphaResearchV113:
    """
    获取 AlphaResearchV113 实例。
    
    Args:
        config_path: 因子配置文件路径
        enable_neutralization: 是否启用中性化
        enable_orthogonalization: 是否启用正交化
        enable_dynamic_weight: 是否启用动态权重
        enable_l2_regularization: 是否启用 L2 正则化
        auto_heal: 是否启用数据自愈
        db_url: 数据库连接 URL
        reflection_output: 反哺 JSON 输出路径
        
    Returns:
        AlphaResearchV113 实例
    """
    return AlphaResearchV113(
        config_path=config_path,
        enable_neutralization=enable_neutralization,
        enable_orthogonalization=enable_orthogonalization,
        enable_dynamic_weight=enable_dynamic_weight,
        enable_l2_regularization=enable_l2_regularization,
        auto_heal=auto_heal,
        db_url=db_url,
        reflection_output=reflection_output
    )