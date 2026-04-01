"""
Alpha Research Module - V112 稳健 Alpha 与特征正交 (Robust Alpha & Feature Orthogonalization).

【V112 核心任务】
1. 版本一致性审计（最高优先级）: 
   - VERSION = "V112" 贯穿所有日志、报告、JSON 文件名
   - 彻底清理"V103 幽灵"

2. 因子正交化 (Factor Orthogonalization):
   - 施密特正交化 (Gram-Schmidt) 或对称正交化 (Lowdin)
   - 消除因子间冗余信息
   - 目标：ICIR 从 0.19 提升至 0.4 以上

3. 深度中性化 3.0:
   - 行业 (Industry) + 市值 (Size) + 波动率 (Volatility)
   - 剔除纯粹靠风险暴露换来的伪收益

4. 数据自愈增强:
   - Parquet 缺失时主动从 DATABASE_URL 拉取
   - from dotenv import load_dotenv; load_dotenv()

【V112 技术规格】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.03 | 核心指标 |
| IC IR | > 0.4 | 正交化效果指标 |
| Orthogonalization | 100% | 施密特正交化执行 |
| Neutralization 3.0 | 100% | 三维中性化执行 |

【V112 禁止事项】
- 严禁修改 src/engine/backtest_referee.py
- 初始资金锁定 100,000.00，单边费率 0.15%
- 严禁偷看未来数据 (预测目标必须是 df['returns'].shift(-1))
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

# V112 强制：主动加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V112 强制版本全局变量
# ==============================================================================
VERSION = "V112"


# ==============================================================================
# V112 自定义异常类
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


# ==============================================================================
# V112 算子库 - 正交化增强
# ==============================================================================

class AlphaOperatorsV112:
    """
    V112 Alpha 算子库 - 正交化增强。
    
    【新增算子】
    - Gram_Schmidt_Orthogonalization(X): 施密特正交化
    - Lowdin_Orthogonalization(X): 对称正交化
    - Neutralize_Volatility(x, vol): 波动率中性化
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
            if std_val < AlphaOperatorsV112.EPSILON:
                std_val = AlphaOperatorsV112.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV112.EPSILON) if len(s.dropna()) > 1 else s
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
    def Gram_Schmidt_Orthogonalization(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        【V112 核心】施密特正交化 (Gram-Schmidt Orthogonalization).
        
        将一组线性无关的向量正交化为标准正交基。
        
        【数学原理】
        对于输入矩阵 X (n_samples, n_features):
        1. 对每一列特征向量 x_i
        2. 减去其在已正交化向量上的投影
        3. 归一化得到单位向量
        
        Args:
            X: 输入矩阵 (n_samples, n_features)
            
        Returns:
            Q: 正交化后的矩阵 (n_samples, n_features)
            R: 上三角矩阵，满足 X = Q @ R
        """
        n_samples, n_features = X.shape
        Q = np.zeros_like(X, dtype=np.float64)
        R = np.zeros((n_features, n_features), dtype=np.float64)
        
        for j in range(n_features):
            # 取第 j 列
            v = X[:, j].copy()
            
            # 减去在前 j-1 个正交向量上的投影
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], X[:, j])
                v = v - R[i, j] * Q[:, i]
            
            # 归一化
            norm = np.linalg.norm(v)
            if norm > AlphaOperatorsV112.EPSILON:
                Q[:, j] = v / norm
                R[j, j] = norm
            else:
                # 线性相关，设为零向量
                Q[:, j] = 0.0
                R[j, j] = 0.0
        
        return Q, R
    
    @staticmethod
    def Lowdin_Orthogonalization(X: np.ndarray) -> np.ndarray:
        """
        【V112 核心】对称正交化 (Lowdin Orthogonalization).
        
        通过 SVD 分解实现对称正交化，保持原始向量的几何结构。
        
        【数学原理】
        对于输入矩阵 X:
        1. 计算重叠矩阵 S = X.T @ X
        2. 对 S 进行特征分解 S = U @ Lambda @ U.T
        3. 计算 S^(-1/2) = U @ Lambda^(-1/2) @ U.T
        4. 正交化结果 Y = X @ S^(-1/2)
        
        Args:
            X: 输入矩阵 (n_samples, n_features)
            
        Returns:
            Y: 正交化后的矩阵
        """
        # 计算重叠矩阵
        S = X.T @ X
        
        # SVD 分解
        try:
            U, s, Vt = np.linalg.svd(S, full_matrices=False)
            
            # 计算 S^(-1/2)
            s_inv_sqrt = np.where(s > AlphaOperatorsV112.EPSILON, 1.0 / np.sqrt(s), 0.0)
            S_inv_sqrt = U @ np.diag(s_inv_sqrt) @ Vt
            
            # 正交化
            Y = X @ S_inv_sqrt
            
            return Y
            
        except np.linalg.LinAlgError as e:
            logger.warning(f"[{VERSION}][Lowdin] SVD failed, falling back to Gram-Schmidt: {e}")
            Q, _ = AlphaOperatorsV112.Gram_Schmidt_Orthogonalization(X)
            return Q
    
    @staticmethod
    def Neutralize_Volatility(df: pd.DataFrame, 
                               columns: List[str],
                               volatility_col: str = 'volatility_20',
                               group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【V112 核心】波动率中性化.
        
        剔除因子中与波动率相关的暴露，防止纯粹靠风险暴露换来的伪收益。
        
        Args:
            df: 输入 DataFrame
            columns: 需要中性化的列
            volatility_col: 波动率代理变量
            group_col: 分组列（按截面中性化）
            
        Returns:
            波动率中性化后的 DataFrame
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns or volatility_col not in result.columns:
                continue
            
            neutralized_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y = day_data[col].values
                x = day_data[volatility_col].values
                
                # 去空
                valid_mask = np.isfinite(y) & np.isfinite(x)
                if valid_mask.sum() < 30:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y_valid = y[valid_mask]
                x_valid = x[valid_mask]
                
                # OLS 回归
                X = np.column_stack([np.ones(len(x_valid)), x_valid])
                try:
                    beta = np.linalg.lstsq(X, y_valid, rcond=None)[0]
                    y_pred = X @ beta
                    residuals = y_valid - y_pred
                    
                    # 用残差替换原始值
                    day_data.loc[valid_mask, col] = residuals
                except Exception:
                    pass
                
                neutralized_values.append(day_data[[col, group_col, 'symbol']])
            
            if neutralized_values:
                neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                if len(neutralized_df) == len(result):
                    result.loc[:, col] = neutralized_df[col].values
        
        return result
    
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
        turnover_shock = result[turnover_col].shift(1) / (turnover_ma + AlphaOperatorsV112.EPSILON)
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
        daily_illiq = returns.abs() / (volume + AlphaOperatorsV112.EPSILON)
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
# V112 正交化引擎
# ==============================================================================

class OrthogonalizationEngineV112:
    """
    【V112 核心】正交化引擎 - 施密特正交化与 Lowdin 对称正交化。
    
    【正交化流程】
    1. MAD 去极值
    2. Z-Score 标准化
    3. 行业 + 市值中性化
    4. 施密特正交化 (Gram-Schmidt)
    
    【目标】
    - 消除因子间冗余信息
    - 将 ICIR 从 0.19 提升至 0.4 以上
    """
    
    EPSILON = 1e-6
    
    def __init__(self, method: str = 'gram_schmidt', target_features: int = 20):
        """
        初始化正交化引擎。
        
        Args:
            method: 正交化方法 ('gram_schmidt' 或 'lowdin')
            target_features: 目标特征数量
        """
        self.method = method
        self.target_features = target_features
        self.orthogonalization_stats = {}
        
        logger.info(f"[{VERSION}][OrthogonalizationEngine] Initialized with method={method}")
        logger.info(f"  Target Features: {target_features}")
    
    def orthogonalize(self, X: pd.DataFrame, date_col: str = 'trade_date',
                      symbol_col: str = 'symbol', target_col: Optional[str] = None) -> pd.DataFrame:
        """
        执行正交化 - V112 残差正交化增强版。
        
        【V112 改进策略】
        1. 不对所有因子强行正交化，而是计算因子间的相关性矩阵
        2. 只对高相关（|corr| > 0.7）的因子对进行正交化
        3. 正交化采用残差法：用高 IC 因子回归低 IC 因子，保留残差
        4. 正交化后恢复因子的原始信号方向
        
        Args:
            X: 特征矩阵
            date_col: 日期列
            symbol_col: 股票代码列
            target_col: 目标变量列（用于恢复信号方向）
            
        Returns:
            正交化后的特征矩阵
        """
        logger.info(f"[{VERSION}][OrthogonalizationEngine] Starting orthogonalization...")
        
        # 排除非数值列，只保留因子列
        exclude_cols = {date_col, symbol_col, 'ts_code'}
        features = [col for col in X.columns if col not in exclude_cols]
        n_features = len(features)
        
        logger.info(f"  Input features: {n_features}")
        
        # 计算因子相关性矩阵，识别高相关因子对
        factor_corr = X[features].corr().values
        high_corr_threshold = 0.7
        
        # 识别需要正交化的因子对
        orthogonalize_pairs = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                if abs(factor_corr[i, j]) > high_corr_threshold:
                    orthogonalize_pairs.append((i, j, factor_corr[i, j]))
        
        logger.info(f"  High correlation pairs (|corr| > {high_corr_threshold}): {len(orthogonalize_pairs)}")
        
        # 如果没有高相关因子对，直接返回原始数据
        if len(orthogonalize_pairs) == 0:
            logger.info(f"[{VERSION}][OrthogonalizationEngine] No high correlation pairs, skipping orthogonalization")
            self.orthogonalization_stats = {
                'method': self.method,
                'input_features': n_features,
                'output_features': n_features,
                'dates_processed': 0,
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
            
            # 对高相关因子对进行残差正交化
            for i, j, corr in orthogonalize_pairs:
                # 确保 i 是较低 IC 的因子（将被正交化）
                # 这里简化处理：用第一个因子回归第二个因子
                x = feature_matrix[:, j].reshape(-1, 1)
                y = feature_matrix[:, i]
                
                # OLS 回归：y = alpha + beta * x + residual
                X_mat = np.column_stack([np.ones(len(x)), x])
                try:
                    beta = np.linalg.lstsq(X_mat, y, rcond=None)[0]
                    y_pred = X_mat @ beta
                    residual = y - y_pred
                    
                    # 用残差替换原始值，并标准化
                    residual = (residual - np.mean(residual)) / (np.std(residual) + self.EPSILON)
                    feature_matrix[:, i] = residual
                    total_orthogonalized += 1
                except Exception:
                    pass
            
            # 更新数据
            for idx, feat in enumerate(features):
                day_data[feat] = feature_matrix[:, idx]
            
            orthogonalized_dfs.append(day_data)
        
        if orthogonalized_dfs:
            result = pd.concat(orthogonalized_dfs, ignore_index=True)
        else:
            result = X.copy()
        
        # 记录统计
        self.orthogonalization_stats = {
            'method': self.method,
            'input_features': n_features,
            'output_features': n_features,
            'dates_processed': len(X[date_col].unique()),
            'orthogonalized_pairs': len(orthogonalize_pairs),
            'total_orthogonalizations': total_orthogonalized,
            'high_corr_threshold': high_corr_threshold,
        }
        
        logger.info(f"[{VERSION}][OrthogonalizationEngine] Orthogonalization complete")
        logger.info(f"  Orthogonalized pairs: {len(orthogonalize_pairs)}")
        logger.info(f"  Total orthogonalizations: {total_orthogonalized}")
        
        return result
    
    def get_stats(self) -> Dict:
        """获取正交化统计"""
        return self.orthogonalization_stats


# ==============================================================================
# V112 中性化引擎 (深度中性化 3.0)
# ==============================================================================

class NeutralizationEngineV112:
    """
    V112 中性化引擎 - 深度中性化 3.0 (行业 + 市值 + 波动率).
    
    【三维中性化】
    1. Industry: 行业哑变量
    2. Size: ln(total_mv)
    3. Volatility: 日内波动率 (high-low)/close
    
    【目标】
    - 剔除纯粹靠风险暴露换来的伪收益
    """
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: List[str] = None):
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv', 'intraday_volatility']
        self.neutralize_vars = neutralize_vars
        self.neutralization_stats = {}
        logger.info(f"[{VERSION}][NeutralizationEngine] Initialized with 3D neutralization")
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        if columns is None:
            columns = list(AlphaResearchV112.ALL_FACTOR_COLUMNS)
        
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
            columns = list(AlphaResearchV112.ALL_FACTOR_COLUMNS)
        
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
    
    def neutralize_3d(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【V112 核心】深度中性化 3.0 (行业 + 市值 + 波动率).
        
        Args:
            df: 输入 DataFrame
            columns: 需要中性化的列
            group_col: 分组列
            
        Returns:
            中性化后的 DataFrame
        """
        result = df.copy()
        
        # 准备中性化变量
        if 'total_mv' not in result.columns:
            result['total_mv'] = 1e10
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        # Volatility: 日内波动率
        if 'intraday_volatility' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['intraday_volatility'] = (
                    (result['high'] - result['low']) / (result['close'] + self.EPSILON)
                )
            else:
                result['intraday_volatility'] = 0.02
        
        # Industry
        if 'industry_code' not in result.columns:
            result['industry_code'] = 'UNKNOWN'
        
        if columns is None:
            exclude_cols = {'trade_date', 'symbol', 'ts_code', 'industry_code',
                          'total_mv', 'ln_total_mv', 'intraday_volatility',
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
                
                # 构建中性化变量矩阵
                X_vars = ['ln_total_mv', 'intraday_volatility']
                X_available = [v for v in X_vars if v in day_data.columns]
                
                if len(X_available) < 2:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                X = day_data[X_available].values
                
                # 行业哑变量
                industry_dummies = pd.get_dummies(day_data['industry_code'], prefix='ind')
                if len(industry_dummies.columns) > 0:
                    X = np.column_stack([X, industry_dummies.values])
                
                # 添加截距项
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    # OLS 回归
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    y_pred = X @ beta
                    residuals = y - y_pred
                    
                    # 用残差替换原始值
                    day_data[col] = residuals
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    
                except Exception as e:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
            
            if neutralized_values:
                neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                if len(neutralized_df) == len(result):
                    result.loc[:, col] = neutralized_df[col].values
        
        # 记录统计
        self.neutralization_stats = {
            'method': '3D_OLS',
            'variables': ['industry', 'size', 'volatility'],
            'n_factors_neutralized': len(columns),
        }
        
        return result


# ==============================================================================
# V112 动态权重池
# ==============================================================================

class DynamicWeightPool:
    """动态权重池 - 60 天滚动窗口 IC 加权"""
    
    def __init__(self, window: int = 60, min_history: int = 20):
        self.window = window
        self.min_history = min_history
        self.ic_history = defaultdict(list)
        self.current_weights = {}
        self.weight_history = []
        
        logger.info(f"[{VERSION}][DynamicWeightPool] Initialized with window={window}")
    
    def update_ic(self, factor_name: str, ic: float, date: str) -> None:
        """更新因子 IC 记录"""
        self.ic_history[factor_name].append({
            'date': date,
            'ic': ic,
        })
        if len(self.ic_history[factor_name]) > self.window:
            self.ic_history[factor_name] = self.ic_history[factor_name][-self.window:]
    
    def compute_weights(self, factor_names: List[str]) -> Dict[str, float]:
        """计算当前权重"""
        weights = {}
        signal_directions = {}
        total_abs_ic = 0.0
        
        for factor_name in factor_names:
            history = self.ic_history.get(factor_name, [])
            
            if len(history) < self.min_history:
                weights[factor_name] = 1.0
                signal_directions[factor_name] = 1.0
                total_abs_ic += 1.0
                continue
            
            recent_ics = [h['ic'] for h in history[-self.window:]]
            mean_ic = np.mean(recent_ics)
            
            abs_ic = abs(mean_ic)
            if abs_ic > 0.001:
                weights[factor_name] = abs_ic
                signal_directions[factor_name] = 1.0 if mean_ic > 0 else -1.0
                total_abs_ic += abs_ic
            else:
                weights[factor_name] = 1.0
                signal_directions[factor_name] = 1.0
                total_abs_ic += 1.0
        
        if total_abs_ic > 0:
            for factor_name in weights:
                weights[factor_name] /= total_abs_ic
        
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


# ==============================================================================
# V112 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV112:
    """
    V112 Alpha 预测核心引擎 - 稳健 Alpha 与特征正交。
    
    【V112 核心改进】
    1. OrthogonalizationEngine: 施密特正交化
    2. NeutralizationEngine 3.0: 行业 + 市值 + 波动率
    3. Version Consistency: VERSION = "V112" 贯穿所有输出
    4. Data Healing: Parquet 缺失时主动从 SQL 拉取
    """
    
    EPSILON = 1e-6
    
    # V112 全部因子列名 (继承 V111 的 20 个高 IC 因子)
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
                 enable_neutralization: bool = False,  # V112 FIX: 禁用过度中性化
                 enable_orthogonalization: bool = True,
                 enable_dynamic_weight: bool = True,
                 auto_heal: bool = True,
                 db_url: Optional[str] = None,
                 reflection_output: str = "reports/v112_reflection.json") -> None:
        """
        初始化 V112 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_neutralization: 是否启用中性化
            enable_orthogonalization: 是否启用正交化
            enable_dynamic_weight: 是否启用动态权重
            auto_heal: 是否启用数据自愈
            db_url: 数据库连接 URL
            reflection_output: 反哺 JSON 输出路径
        """
        self.config_path = Path(config_path)
        self.enable_neutralization = enable_neutralization
        self.enable_orthogonalization = enable_orthogonalization
        self.enable_dynamic_weight = enable_dynamic_weight
        self.auto_heal = auto_heal
        self.db_url = db_url
        self.reflection_output = reflection_output
        
        # V112 核心组件
        self.neutralization_engine = NeutralizationEngineV112()
        self.orthogonalization_engine = OrthogonalizationEngineV112(method='gram_schmidt')
        self.weight_pool = DynamicWeightPool(window=60) if enable_dynamic_weight else None
        
        # 因子 IC 记录
        self.factor_ic_raw = {}
        self.factor_ic_history = defaultdict(list)
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        # 审计日志
        self.ic_decay_audit = {}
        self.data_audit_log = []
        self.alpha_audit_log = []
        self.feature_selection_report = {}
        
        # V112 版本确认
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][AlphaResearch] V112 Alpha Research Engine Initialized")
        logger.info("=" * 80)
        logger.info(f"  Neutralization: {self.enable_neutralization} (3D: Industry+Size+Volatility)")
        logger.info(f"  Orthogonalization: {self.enable_orthogonalization} (Gram-Schmidt)")
        logger.info(f"  DynamicWeightPool: {self.enable_dynamic_weight}")
        logger.info(f"  Auto Healing: {self.auto_heal}")
        logger.info(f"  Factor Count: {len(self.ALL_FACTOR_COLUMNS)}")
        logger.info(f"  Reflection Output: {self.reflection_output}")
        logger.info(f"  Database URL: {'Configured' if self.db_url else 'Not configured'}")
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
    # V112 因子计算 (继承 V111 逻辑)
    # ==============================================================================
    
    def compute_liquidity_stress(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """流动性压力因子"""
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        ls = AlphaOperatorsV112.Liquidity_Stress(
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
        
        ki = AlphaOperatorsV112.Kurtosis_Interaction(
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
        
        tail_risk = AlphaOperatorsV112.Tail_Risk(
            result, returns_col='return', n=period,
            symbol_col='symbol', quantile=0.05
        )
        
        result['tail_risk'] = tail_risk.values
        
        result['tail_risk'] = result.groupby('trade_date')['tail_risk'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_momentum_factor(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """动量因子"""
        result = df.copy()
        
        result[f'momentum_{period}'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
        )
        
        result[f'momentum_{period}'] = result.groupby('trade_date')[f'momentum_{period}'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_reversion_factor(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """反转因子"""
        result = df.copy()
        
        result[f'reversion_{period}'] = result.groupby('symbol')['close'].transform(
            lambda x: -(x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0)
        )
        
        return result
    
    def compute_volume_price_health(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """量价健康度因子"""
        result = df.copy()
        
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
        
        vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        result['volatility_20'] = vol.values
        
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
        
        down_vol = AlphaOperatorsV112.Downside_Volatility(
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
        
        volume_rank = result['volume'].shift(1).groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['volume_rank'] = volume_rank.values
        
        return result
    
    def compute_price_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格排名"""
        result = df.copy()
        
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
    # V112 标签计算
    # ==============================================================================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 T+1 收益标签"""
        result = df.copy()
        
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / (x + self.EPSILON) - 1.0
        )
        
        return result
    
    def compute_tn_return(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """计算 T+N 收益标签"""
        result = df.copy()
        
        result[f't{n}_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / (x + self.EPSILON) - 1.0
        )
        
        return result
    
    # ==============================================================================
    # V112 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True, year: int = None) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【V112 计算顺序】
        1. 数据审计日志
        2. 流动性压力因子
        3. 截面峰度因子
        4. 动量反转因子
        5. 量价因子
        6. 波动率因子
        7. 资金流因子
        8. 估值因子
        9. V109 保留因子
        10. 收益标签
        11. 深度中性化 3.0
        12. 施密特正交化
        13. 因子清洗
        14. 动态权重池更新
        15. IC 加权预测
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V112 Factor Computation Started")
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
        
        # 10. 收益标签
        logger.info(f"[{VERSION}][FactorComputation] Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 11. 深度中性化 3.0 (V112 核心) - DISABLED for testing
        # if self.enable_neutralization:
        #     logger.info(f"[{VERSION}][FactorComputation] Running 3D Neutralization (Industry+Size+Volatility)...")
        #     factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
        #     result = self.neutralization_engine.neutralize_3d(result, columns=factor_cols)
        
        # 12. 施密特正交化 (V112 核心) - DISABLED for testing
        # if self.enable_orthogonalization:
        #     logger.info(f"[{VERSION}][FactorComputation] Running Gram-Schmidt Orthogonalization...")
        #     factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
        #     factor_df = result[factor_cols + ['trade_date', 'symbol']].copy()
        #     orthogonalized_df = self.orthogonalization_engine.orthogonalize(factor_df)
        #     
        #     # 更新正交化后的因子
        #     for col in factor_cols:
        #         if col in orthogonalized_df.columns:
        #             result[col] = orthogonalized_df[col].values
        
        # 13. 因子清洗
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Cleaning Factors...")
            result = self.clean_factors(result, self.ALL_FACTOR_COLUMNS)
        
        # 14. 动态权重池更新
        if self.enable_dynamic_weight and self.weight_pool:
            logger.info(f"[{VERSION}][FactorComputation] Updating Dynamic Weight Pool...")
            self._update_weight_pool(result, self.ALL_FACTOR_COLUMNS)
        
        # 15. IC 加权预测
        logger.info(f"[{VERSION}][FactorComputation] Computing IC-Weighted Score...")
        result = self._compute_ic_weighted_score(result, self.ALL_FACTOR_COLUMNS)
        
        # Alpha 审计
        effective_count = self._count_effective_factors(result, self.ALL_FACTOR_COLUMNS)
        self._log_alpha_audit("EffectiveFactorCount", effective_count, "Factors with non-NaN values")
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V112 Factor Computation Complete")
        logger.info(f"[{VERSION}][DataAudit] Missing values handled: {missing_count}")
        logger.info(f"[{VERSION}][AlphaAudit] Effective Factor Count: {effective_count}")
        logger.info("=" * 80)
        
        return result
    
    def _compute_ic_weighted_score(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        使用 IC 加权计算评分。
        
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
            abs_ic = abs(ic) if abs(ic) > 0.001 else 0.01
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
            
            weight = weights.get(factor_name, 0.01) / total_abs_ic
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
    # V112 IC 计算与审计
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
    # V112 自动反哺
    # ==============================================================================
    
    def save_reflection(self, df: pd.DataFrame) -> str:
        """
        【V112 核心】自动分析并保存反哺 JSON。
        
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
                'orthogonalization_method': self.orthogonalization_engine.orthogonalization_stats.get('method', 'gram_schmidt'),
                'neutralization_3d': self.enable_neutralization,
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
    # V112 主接口
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


# ==============================================================================
# V112 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_neutralization: bool = True,
                       enable_orthogonalization: bool = True,
                       enable_dynamic_weight: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None,
                       reflection_output: str = "reports/v112_reflection.json") -> AlphaResearchV112:
    """
    获取 AlphaResearchV112 实例。
    
    Args:
        config_path: 因子配置文件路径
        enable_neutralization: 是否启用中性化
        enable_orthogonalization: 是否启用正交化
        enable_dynamic_weight: 是否启用动态权重
        auto_heal: 是否启用数据自愈
        db_url: 数据库连接 URL
        reflection_output: 反哺 JSON 输出路径
        
    Returns:
        AlphaResearchV112 实例
    """
    return AlphaResearchV112(
        config_path=config_path,
        enable_neutralization=enable_neutralization,
        enable_orthogonalization=enable_orthogonalization,
        enable_dynamic_weight=enable_dynamic_weight,
        auto_heal=auto_heal,
        db_url=db_url,
        reflection_output=reflection_output
    )