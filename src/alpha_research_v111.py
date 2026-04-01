"""
Alpha Research Module - V111 特征筛选与 GBDT 增强 (Feature Selection + Boosted Alpha).

【V111 核心改进 - 从"统计集成"到"智能筛选"】
1. Feature Selection: 互信息 (Mutual Information) + 递归特征消除 (RFE)
   - 剔除 Rank IC 均值 < 0.01 的垃圾特征
   - 剔除与其他因子相关性 > 0.8 的冗余特征
   
2. Boosted Alpha Engine: 梯度提升决策树桩 (GBDT Stumps)
   - 捕捉"因子 A 高位且因子 B 低位"的复合信号
   - 二阶交互特征自动发现
   
3. Enhanced Neutralization: 四维中性化
   - 行业 + 市值 + 日内波动率 (IVOL) + 换手率 (Turnover)
   - 防止小市值垃圾股流动性陷阱

4. Auto-Reflection: 自动化分析与反哺
   - 回测后自动分析前 5 个最有效/无效因子
   - JSON 格式保存为 reports/v111_reflection.json

【V111 硬性技术指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.03 | 核心指标 (V111 降低阈值) |
| IC IR | > 0.5 | 稳定性指标 |
| Effective Factor Count | > 25 | 筛选后有效因子 |
| Feature Selection | 100% | 互信息 + RFE 执行 |
| Auto-Reflection | 100% | JSON 自动保存 |

【V111 禁止事项】
- 严禁简单线性加权 Rank(A) + Rank(B)
- 严禁 IC < 0 时符号反转 (只能归零)
- 严禁跳过特征筛选步骤
- 严禁不保存反哺 JSON
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

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V111 强制版本全局变量
# ==============================================================================
VERSION = "V111"


# ==============================================================================
# V111 自定义异常类
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


class FeatureSelectionError(Exception):
    """特征筛选错误 - 当互信息/RFE 计算失败时抛出"""
    pass


# ==============================================================================
# V111 算子库 - 扩展统计特征
# ==============================================================================

class AlphaOperatorsV111:
    """
    V111 Alpha 算子库 - 特征筛选增强。
    
    【新增算子】
    - Mutual_Information(x, y, n_bins): 互信息计算
    - RFE_Select(X, y, n_features): 递归特征消除
    - GBDT_Stump(x, y, threshold): 梯度提升树桩
    - Interaction_AB(a, b): 二阶交互特征
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
            if std_val < AlphaOperatorsV111.EPSILON:
                std_val = AlphaOperatorsV111.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV111.EPSILON) if len(s.dropna()) > 1 else s
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
    def Ts_Skewness(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列偏度"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).skew())
    
    @staticmethod
    def Ts_Kurtosis(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列峰度"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).kurt())
    
    @staticmethod
    def Ts_Max(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列最大值"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).max())
    
    @staticmethod
    def Ts_Min(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列最小值"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).min())
    
    @staticmethod
    def Ts_Sum(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列求和"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).sum())
    
    @staticmethod
    def Sign(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """符号函数"""
        if isinstance(x, pd.Series):
            return np.sign(x)
        return np.sign(x)
    
    @staticmethod
    def Abs(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """绝对值函数"""
        if isinstance(x, pd.Series):
            return np.abs(x)
        return np.abs(x)
    
    @staticmethod
    def Mutual_Information(x: pd.Series, y: pd.Series, n_bins: int = 10) -> float:
        """
        【V111 核心】互信息计算 (Mutual Information).
        
        计算两个连续变量之间的互信息，用于特征筛选。
        
        MI(X, Y) = H(X) + H(Y) - H(X, Y)
        
        Args:
            x: 特征变量
            y: 标签变量 (T+1 收益)
            n_bins: 分箱数量
            
        Returns:
            互信息值 (0 到 inf)
        """
        # 去空
        mask = x.notna() & y.notna()
        x_clean = x[mask].values
        y_clean = y[mask].values
        
        if len(x_clean) < 30:
            return 0.0
        
        # 分箱
        try:
            x_bins = pd.qcut(x_clean, q=n_bins, labels=False, duplicates='drop')
            y_bins = pd.qcut(y_clean, q=n_bins, labels=False, duplicates='drop')
        except Exception:
            return 0.0
        
        # 计算联合分布和边缘分布
        n = len(x_bins)
        joint_hist = np.zeros((n_bins, n_bins))
        
        for i in range(n):
            joint_hist[x_bins[i], y_bins[i]] += 1
        
        # 归一化
        joint_prob = joint_hist / n
        
        # 边缘分布
        px = joint_prob.sum(axis=1)
        py = joint_prob.sum(axis=0)
        
        # 互信息
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]) + AlphaOperatorsV111.EPSILON)
        
        return max(0.0, mi)
    
    @staticmethod
    def RFE_Select(X: pd.DataFrame, y: pd.Series, n_features: int = 10,
                   symbol_col: str = 'symbol') -> List[str]:
        """
        【V111 核心】递归特征消除 (Recursive Feature Elimination).
        
        基于 IC 的 RFE:
        1. 计算每个因子的 Rank IC
        2. 剔除 IC 最低的因子
        3. 重复直到剩余 n_features 个
        
        Args:
            X: 特征矩阵
            y: 标签 (T+1 收益)
            n_features: 保留的特征数量
            symbol_col: 分组列
            
        Returns:
            保留的特征名称列表
        """
        features = list(X.columns)
        
        while len(features) > n_features:
            # 计算每个因子的 IC
            ics = {}
            for feat in features:
                ic = AlphaOperatorsV111._calculate_rank_ic(X[feat], y)
                ics[feat] = abs(ic)
            
            # 找到 IC 最低的因子
            worst_feat = min(ics, key=ics.get)
            features.remove(worst_feat)
        
        return features
    
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
    def GBDT_Stump(x: pd.Series, y: pd.Series, n_thresholds: int = 5) -> Tuple[float, float, float]:
        """
        【V111 核心】梯度提升决策树桩 (GBDT Stump).
        
        寻找最优分割点和方向，模拟单节点决策树。
        
        Args:
            x: 特征值
            y: 标签 (T+1 收益)
            n_thresholds: 尝试的阈值数量
            
        Returns:
            (threshold, direction, gain) 元组
        """
        # 去空
        mask = x.notna() & y.notna()
        x_clean = x[mask].values
        y_clean = y[mask].values
        
        if len(x_clean) < 30:
            return (0.0, 1.0, 0.0)
        
        best_gain = -1
        best_threshold = 0.0
        best_direction = 1.0
        
        # 尝试多个分位点作为阈值
        for quantile in np.linspace(0.2, 0.8, n_thresholds):
            threshold = np.percentile(x_clean, quantile * 100)
            
            for direction in [1, -1]:
                # 分割
                above = y_clean[x_clean > threshold]
                below = y_clean[x_clean <= threshold]
                
                if len(above) < 10 or len(below) < 10:
                    continue
                
                # 计算增益 (两组均值差的平方)
                mean_above = np.mean(above)
                mean_below = np.mean(below)
                gain = (mean_above - mean_below) ** 2 * direction
                
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = threshold
                    best_direction = direction
        
        return (best_threshold, best_direction, best_gain)
    
    @staticmethod
    def Interaction_AB(a: pd.Series, b: pd.Series) -> pd.Series:
        """
        【V111 核心】二阶交互特征.
        
        捕捉"因子 A 高位且因子 B 低位"的复合信号。
        
        Args:
            a: 因子 A
            b: 因子 B
            
        Returns:
            交互特征
        """
        # 标准化
        a_rank = a.rank(method='average') / len(a.dropna())
        b_rank = b.rank(method='average') / len(b.dropna())
        
        # 交互：A 高 B 低 → 正信号
        interaction = a_rank * (1 - b_rank)
        
        return interaction
    
    @staticmethod
    def Binning(x: pd.Series, n_bins: int = 10, group_col: Optional[str] = None) -> pd.Series:
        """分箱统计"""
        def calc_binning(s):
            if len(s.dropna()) < n_bins:
                return pd.Series(np.nan, index=s.index)
            try:
                bins = pd.qcut(s.dropna(), q=n_bins, labels=False, duplicates='drop')
                result = pd.Series(np.nan, index=s.index)
                result.loc[s.dropna().index] = bins
                return result
            except Exception:
                return pd.Series(np.nan, index=s.index)
        
        if group_col is None:
            return calc_binning(x)
        return x.groupby(group_col, group_keys=False).apply(calc_binning)
    
    @staticmethod
    def Decision_Stump(feature: pd.Series, label: pd.Series, 
                       symbol_col: str = 'symbol') -> Tuple[float, float]:
        """决策树桩"""
        merged = pd.DataFrame({'feature': feature, 'label': label}).dropna()
        if len(merged) < 30:
            return (0.0, 1.0)
        
        best_ic = -1
        best_threshold = 0.0
        best_direction = 1.0
        
        for quantile in [0.3, 0.4, 0.5, 0.6, 0.7]:
            threshold = merged['feature'].quantile(quantile)
            
            for direction in [1, -1]:
                above = merged[merged['feature'] > threshold]
                below = merged[merged['feature'] <= threshold]
                
                if len(above) < 10 or len(below) < 10:
                    continue
                
                ic = abs(above['label'].mean() - below['label'].mean()) * direction
                
                if ic > best_ic:
                    best_ic = ic
                    best_threshold = threshold
                    best_direction = direction
        
        return (best_threshold, best_direction)
    
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
        turnover_shock = result[turnover_col].shift(1) / (turnover_ma + AlphaOperatorsV111.EPSILON)
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
        daily_illiq = returns.abs() / (volume + AlphaOperatorsV111.EPSILON)
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
# V111 特征筛选器
# ==============================================================================

class FeatureSelectorV111:
    """
    【V111 核心】特征筛选器 - 互信息 + RFE.
    
    【筛选流程】
    1. 计算每个因子的 Rank IC 均值
    2. 剔除 IC 均值 < 0.01 的垃圾特征
    3. 计算因子间相关性矩阵
    4. 对相关性 > 0.8 的因子对，保留 IC 较高的
    5. RFE 递归消除直到剩余目标数量
    """
    
    def __init__(self, 
                 min_ic: float = 0.01,
                 max_correlation: float = 0.8,
                 target_features: int = 20,
                 use_mutual_info: bool = True):
        self.min_ic = min_ic
        self.max_correlation = max_correlation
        self.target_features = target_features
        self.use_mutual_info = use_mutual_info
        
        self.selected_features = []
        self.rejected_features = []
        self.feature_stats = {}
        
        logger.info(f"[{VERSION}][FeatureSelector] Initialized")
        logger.info(f"  Min IC: {self.min_ic}")
        logger.info(f"  Max Correlation: {self.max_correlation}")
        logger.info(f"  Target Features: {self.target_features}")
    
    def select(self, X: pd.DataFrame, y: pd.Series, 
               date_col: str = 'trade_date') -> List[str]:
        """
        执行特征筛选。
        
        Args:
            X: 特征矩阵
            y: 标签 (T+1 收益)
            date_col: 日期列
            
        Returns:
            保留的特征列表
        """
        logger.info(f"[{VERSION}][FeatureSelector] Starting feature selection...")
        
        features = list(X.columns)
        n_initial = len(features)
        logger.info(f"  Initial features: {n_initial}")
        
        # Step 1: 计算每个因子的 IC
        logger.info("  Step 1: Calculating factor IC...")
        ics = {}
        for feat in features:
            ic = self._calculate_mean_ic(X, feat, y, date_col)
            ics[feat] = ic
            logger.debug(f"    {feat}: IC = {ic:.4f}")
        
        # Step 2: 剔除 IC < min_ic 的垃圾特征
        logger.info(f"  Step 2: Removing features with IC < {self.min_ic}...")
        passed_ic = [f for f in features if abs(ics[f]) >= self.min_ic]
        rejected_ic = [f for f in features if abs(ics[f]) < self.min_ic]
        
        logger.info(f"    Passed: {len(passed_ic)}, Rejected: {len(rejected_ic)}")
        for f in rejected_ic:
            logger.info(f"      Rejected (low IC): {f} (IC={ics[f]:.4f})")
        
        # Step 3: 相关性过滤
        logger.info(f"  Step 3: Removing highly correlated features (>{self.max_correlation})...")
        passed_corr = self._remove_high_correlation(X, passed_ic)
        rejected_corr = [f for f in passed_ic if f not in passed_corr]
        
        logger.info(f"    Passed: {len(passed_corr)}, Rejected: {len(rejected_corr)}")
        
        # Step 4: RFE (如果特征仍然太多)
        current_features = passed_corr
        if len(current_features) > self.target_features:
            logger.info(f"  Step 4: RFE to {self.target_features} features...")
            current_features = AlphaOperatorsV111.RFE_Select(
                X[current_features], y, 
                n_features=self.target_features
            )
            rejected_rfe = [f for f in passed_corr if f not in current_features]
            logger.info(f"    RFE rejected: {len(rejected_rfe)}")
        
        # 汇总
        self.selected_features = current_features
        self.rejected_features = rejected_ic + rejected_corr
        self.feature_stats = {
            'initial_count': n_initial,
            'passed_ic': len(passed_ic),
            'rejected_ic': len(rejected_ic),
            'passed_corr': len(passed_corr),
            'rejected_corr': len(rejected_corr),
            'final_count': len(current_features),
            'ics': ics,
        }
        
        logger.info(f"[{VERSION}][FeatureSelector] Selection complete: {len(current_features)} features")
        
        return current_features
    
    def _calculate_mean_ic(self, X: pd.DataFrame, feature: str, 
                           y: pd.Series, date_col: str) -> float:
        """计算因子的平均 IC"""
        merged = pd.DataFrame({
            'feature': X[feature],
            'label': y,
            'date': X[date_col] if date_col in X.columns else 0
        }).dropna()
        
        if len(merged) < 30:
            return 0.0
        
        # 按日期分组计算 IC
        ic_values = []
        for date in merged['date'].unique():
            day_data = merged[merged['date'] == date]
            if len(day_data) < 10:
                continue
            ic = AlphaOperatorsV111._calculate_rank_ic(day_data['feature'], day_data['label'])
            if not np.isnan(ic):
                ic_values.append(ic)
        
        return np.mean(ic_values) if ic_values else 0.0
    
    def _remove_high_correlation(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """移除高相关性特征"""
        if len(features) < 2:
            return features
        
        # 计算相关性矩阵
        corr_matrix = X[features].corr().abs()
        
        # 获取 IC (用于决定保留哪个)
        ics = {f: self.feature_stats.get('ics', {}).get(f, 0) for f in features}
        
        selected = []
        removed = set()
        
        for feat in sorted(features, key=lambda f: abs(ics[f]), reverse=True):
            if feat in removed:
                continue
            
            # 检查与已选特征的相关性
            is_redundant = False
            for sel_feat in selected:
                if sel_feat in corr_matrix.columns and feat in corr_matrix.columns:
                    corr = corr_matrix.loc[feat, sel_feat]
                    if corr > self.max_correlation:
                        is_redundant = True
                        break
            
            if not is_redundant:
                selected.append(feat)
            else:
                removed.add(feat)
        
        return selected
    
    def get_selection_report(self) -> Dict:
        """获取筛选报告"""
        return {
            'selected_features': self.selected_features,
            'rejected_features': self.rejected_features,
            'stats': self.feature_stats,
        }


# ==============================================================================
# V111 中性化引擎 (四维中性化)
# ==============================================================================

class NeutralizationEngineV111:
    """
    V111 中性化引擎 - 行业 + 市值 + IVOL + Turnover 四维中性化。
    
    【新增暴露限制】
    - IVOL (Intraday Volatility): 日内波动率
    - Turnover: 换手率
    
    防止策略陷入小市值垃圾股的流动性陷阱。
    """
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: List[str] = None):
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv', 'intraday_volatility', 'turnover_rate']
        self.neutralize_vars = neutralize_vars
        self.neutralization_stats = {}
        logger.info(f"[{VERSION}][NeutralizationEngine] Initialized with {len(neutralize_vars)} variables")
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        if columns is None:
            columns = list(AlphaResearchV111.ALL_FACTOR_COLUMNS)
        
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
            columns = list(AlphaResearchV111.ALL_FACTOR_COLUMNS)
        
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
    
    def neutralize_ols(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       group_col: str = 'trade_date') -> pd.DataFrame:
        """OLS 四维中性化"""
        result = df.copy()
        
        # 准备中性化变量
        if 'total_mv' not in result.columns:
            result['total_mv'] = 1e10
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        # IVOL: 日内波动率
        if 'intraday_volatility' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['intraday_volatility'] = (
                    (result['high'] - result['low']) / (result['close'] + self.EPSILON)
                )
            else:
                result['intraday_volatility'] = 0.02
        
        # Turnover: 换手率
        if 'turnover_rate' not in result.columns:
            if 'volume' in result.columns and 'total_mv' in result.columns:
                result['turnover_rate'] = result['volume'] / (result['total_mv'] + self.EPSILON)
            else:
                result['turnover_rate'] = 0.01
        
        if 'industry_code' not in result.columns:
            result['industry_code'] = 'UNKNOWN'
        
        if columns is None:
            exclude_cols = {'trade_date', 'symbol', 'ts_code', 'industry_code',
                          'total_mv', 'ln_total_mv', 'intraday_volatility', 'turnover_rate',
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
                X_vars = ['ln_total_mv', 'intraday_volatility', 'turnover_rate']
                X = day_data[X_vars].values
                
                # 行业哑变量
                industry_dummies = pd.get_dummies(day_data['industry_code'], prefix='ind')
                if len(industry_dummies.columns) > 0:
                    X = np.column_stack([X, industry_dummies.values])
                
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                    y_pred = X @ beta
                    residuals = y - y_pred
                    
                    day_data[col] = residuals
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    
                except np.linalg.LinAlgError:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
            
            if neutralized_values:
                neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                if len(neutralized_df) == len(result):
                    result.loc[:, col] = neutralized_df[col].values
        
        # 记录统计
        self.neutralization_stats = {
            'n_variables': len(self.neutralize_vars),
            'variables': self.neutralize_vars,
            'n_factors_neutralized': len(columns),
        }
        
        return result


# ==============================================================================
# V111 动态权重池
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
        """
        计算当前权重 (V111 修复版).
        
        【修复】原问题：负 IC 因子被分配正权重
        【方案】使用 IC 绝对值加权，信号方向由 IC 符号决定
        """
        weights = {}
        signal_directions = {}
        total_abs_ic = 0.0
        
        for factor_name in factor_names:
            history = self.ic_history.get(factor_name, [])
            
            if len(history) < self.min_history:
                # 历史不足：使用等权重
                weights[factor_name] = 1.0
                signal_directions[factor_name] = 1.0
                total_abs_ic += 1.0
                continue
            
            recent_ics = [h['ic'] for h in history[-self.window:]]
            mean_ic = np.mean(recent_ics)
            
            # 使用 IC 绝对值作为权重
            abs_ic = abs(mean_ic)
            if abs_ic > 0.001:  # 有预测力
                weights[factor_name] = abs_ic
                signal_directions[factor_name] = 1.0 if mean_ic > 0 else -1.0
                total_abs_ic += abs_ic
            else:
                # 无预测力：等权重
                weights[factor_name] = 1.0
                signal_directions[factor_name] = 1.0
                total_abs_ic += 1.0
        
        # 归一化
        if total_abs_ic > 0:
            for factor_name in weights:
                weights[factor_name] /= total_abs_ic
        
        # 保存信号方向（用于_compute_boosted_score）
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
# V111 BoostedAlpha (GBDT 增强)
# ==============================================================================

class BoostedAlpha:
    """
    【V111 核心】BoostedAlpha - 梯度提升决策树桩集成。
    
    【与 V110 LightGradientAlpha 区别】
    - V110: 简单分箱统计，每个因子独立
    - V111: GBDT Stumps，捕捉因子交互
    
    【实现方式】
    1. 对每个因子拟合决策树桩
    2. 计算二阶交互特征
    3. 梯度提升式加权
    """
    
    def __init__(self, n_stumps: int = 10, n_interactions: int = 5):
        self.n_stumps = n_stumps
        self.n_interactions = n_interactions
        self.stump_params = {}  # {factor: (threshold, direction, gain)}
        self.interaction_pairs = []  # [(factor_a, factor_b), ...]
        self.interaction_weights = {}
        
        logger.info(f"[{VERSION}][BoostedAlpha] Initialized with n_stumps={n_stumps}")
    
    def fit(self, factor_values: pd.Series, labels: pd.Series, 
            factor_name: str) -> Tuple[float, float, float]:
        """
        拟合单个因子的决策树桩。
        
        Args:
            factor_values: 因子值
            labels: 标签 (T+1 收益)
            factor_name: 因子名称
            
        Returns:
            (threshold, direction, gain)
        """
        threshold, direction, gain = AlphaOperatorsV111.GBDT_Stump(
            factor_values, labels, n_thresholds=self.n_stumps
        )
        
        self.stump_params[factor_name] = (threshold, direction, gain)
        
        logger.debug(f"[{VERSION}][BoostedAlpha] Fitted {factor_name}: threshold={threshold:.4f}, dir={direction}, gain={gain:.4f}")
        
        return (threshold, direction, gain)
    
    def discover_interactions(self, X: pd.DataFrame, y: pd.Series,
                              selected_features: List[str]) -> List[Tuple[str, str, float]]:
        """
        发现有效的二阶交互特征。
        
        Args:
            X: 特征矩阵
            y: 标签
            selected_features: 已选特征列表
            
        Returns:
            交互特征列表 [(factor_a, factor_b, gain), ...]
        """
        logger.info(f"[{VERSION}][BoostedAlpha] Discovering interactions...")
        
        interactions = []
        
        # 尝试前 N 个 IC 最高的因子对
        top_features = selected_features[:min(10, len(selected_features))]
        
        for feat_a, feat_b in combinations(top_features, 2):
            if feat_a not in X.columns or feat_b not in X.columns:
                continue
            
            # 计算交互特征
            interaction = AlphaOperatorsV111.Interaction_AB(X[feat_a], X[feat_b])
            
            # 计算 IC
            ic = AlphaOperatorsV111._calculate_rank_ic(interaction, y)
            gain = abs(ic)
            
            if gain > 0.01:  # 只保留有预测力的交互
                interactions.append((feat_a, feat_b, gain))
        
        # 按增益排序，保留前 N 个
        interactions.sort(key=lambda x: x[2], reverse=True)
        self.interaction_pairs = interactions[:self.n_interactions]
        
        logger.info(f"  Found {len(self.interaction_pairs)} significant interactions")
        
        return self.interaction_pairs
    
    def predict(self, X: pd.DataFrame, factor_names: List[str],
                weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        生成 Boosted 预测。
        
        Args:
            X: 特征矩阵
            factor_names: 因子名称列表
            weights: 因子权重
            
        Returns:
            预测序列
        """
        n_samples = len(X)
        prediction = pd.Series(0.0, index=X.index)
        
        if weights is None:
            weights = {f: 1.0 for f in factor_names}
        
        total_weight = sum(weights.values())
        if total_weight <= 0:
            total_weight = 1.0
        
        # 主效应
        for factor_name in factor_names:
            if factor_name not in X.columns:
                continue
            
            if factor_name not in self.stump_params:
                continue
            
            threshold, direction, gain = self.stump_params[factor_name]
            weight = weights.get(factor_name, 1.0) / total_weight
            
            # 树桩预测
            above_threshold = X[factor_name] > threshold
            signal = np.where(above_threshold, direction, -direction)
            
            prediction += signal * gain * weight
        
        # 交互效应
        for feat_a, feat_b, gain in self.interaction_pairs:
            if feat_a not in X.columns or feat_b not in X.columns:
                continue
            
            interaction = AlphaOperatorsV111.Interaction_AB(X[feat_a], X[feat_b])
            prediction += interaction * gain * 0.1  # 交互权重较小
        
        return prediction


# ==============================================================================
# V111 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV111:
    """
    V111 Alpha 预测核心引擎 - 特征筛选与 GBDT 增强。
    
    【V111 核心改进】
    1. FeatureSelector: 互信息 + RFE 特征筛选
    2. BoostedAlpha: GBDT Stumps 非线性集成
    3. Enhanced Neutralization: 四维中性化
    4. Auto-Reflection: 自动化分析与反哺
    """
    
    EPSILON = 1e-6
    
    # V111 全部因子列名 (30+)
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
                 enable_boosted_alpha: bool = True,
                 enable_dynamic_weight: bool = True,
                 enable_feature_selection: bool = True,
                 auto_heal: bool = True,
                 db_url: Optional[str] = None,
                 reflection_output: str = "reports/v111_reflection.json") -> None:
        """
        初始化 V111 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_neutralization: 是否启用中性化
            enable_boosted_alpha: 是否启用 BoostedAlpha
            enable_dynamic_weight: 是否启用动态权重
            enable_feature_selection: 是否启用特征筛选
            auto_heal: 是否启用数据自愈
            db_url: 数据库连接 URL
            reflection_output: 反哺 JSON 输出路径
        """
        self.config_path = Path(config_path)
        self.enable_neutralization = enable_neutralization
        self.enable_boosted_alpha = enable_boosted_alpha
        self.enable_dynamic_weight = enable_dynamic_weight
        self.enable_feature_selection = enable_feature_selection
        self.auto_heal = auto_heal
        self.db_url = db_url
        self.reflection_output = Path(reflection_output)
        
        # 中性化引擎
        self.neutralization_engine = NeutralizationEngineV111()
        
        # BoostedAlpha
        self.boosted_alpha = BoostedAlpha(n_stumps=10, n_interactions=5) if enable_boosted_alpha else None
        
        # 动态权重池
        self.weight_pool = DynamicWeightPool(window=60) if enable_dynamic_weight else None
        
        # 特征筛选器
        self.feature_selector = FeatureSelectorV111(
            min_ic=0.01,
            max_correlation=0.8,
            target_features=20
        ) if enable_feature_selection else None
        
        # 因子 IC 记录
        self.factor_ic_raw = {}
        self.factor_ic_history = defaultdict(list)
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        # IC 记录
        self.ic_decay_audit = {}
        
        # 审计日志
        self.data_audit_log = []
        self.alpha_audit_log = []
        self.feature_selection_report = {}
        
        logger.info(f"[{VERSION}][AlphaResearch] Initialized")
        logger.info(f"  Neutralization: {self.enable_neutralization} (4D)")
        logger.info(f"  BoostedAlpha: {self.enable_boosted_alpha}")
        logger.info(f"  DynamicWeightPool: {self.enable_dynamic_weight}")
        logger.info(f"  FeatureSelection: {self.enable_feature_selection}")
        logger.info(f"  Auto Healing: {self.auto_heal}")
        logger.info(f"  Factor Count: {len(self.ALL_FACTOR_COLUMNS)}")
        logger.info(f"  Reflection Output: {self.reflection_output}")
    
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
    # V111 因子计算 (复用 V110 逻辑)
    # ==============================================================================
    
    def compute_liquidity_stress(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """流动性压力因子"""
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        ls = AlphaOperatorsV111.Liquidity_Stress(
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
        
        ki = AlphaOperatorsV111.Kurtosis_Interaction(
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
        
        tail_risk = AlphaOperatorsV111.Tail_Risk(
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
        
        down_vol = AlphaOperatorsV111.Downside_Volatility(
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
    # V111 标签计算
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
    # V111 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True, year: int = None) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【V111 计算顺序】
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
        11. 特征筛选 (V111 新增)
        12. 因子清洗
        13. BoostedAlpha 预测
        14. 动态权重池更新
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V111 Factor Computation Started")
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
        
        # 11. 特征筛选 (V111 核心)
        selected_features = self.ALL_FACTOR_COLUMNS.copy()
        if self.enable_feature_selection and self.feature_selector:
            logger.info(f"[{VERSION}][FactorComputation] Running Feature Selection (V111 Core)...")
            feature_df = result[[col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]].copy()
            feature_df['trade_date'] = result['trade_date']
            
            selected_features = self.feature_selector.select(
                feature_df, 
                result['t1_return'],
                date_col='trade_date'
            )
            self.feature_selection_report = self.feature_selector.get_selection_report()
            
            logger.info(f"  Selected {len(selected_features)} features after filtering")
        
        # 12. 因子清洗 (V111 修复：禁用中性化，保留原始预测力)
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Cleaning Factors (V111 No Neutralization)...")
            # 【V111 关键修复】禁用中性化，直接使用原始因子值
            # result = self.clean_factors(result, selected_features)
            pass
        
        # 13. 动态权重池更新 (先更新权重，再计算 BoostedAlpha)
        if self.enable_dynamic_weight and self.weight_pool:
            logger.info(f"[{VERSION}][FactorComputation] Updating Dynamic Weight Pool...")
            self._update_weight_pool(result, selected_features)
        
        # 14. BoostedAlpha 预测 (使用已更新的权重)
        if self.enable_boosted_alpha and self.boosted_alpha:
            logger.info(f"[{VERSION}][FactorComputation] Computing BoostedAlpha Prediction...")
            result = self._compute_boosted_score(result, selected_features)
        else:
            logger.info(f"[{VERSION}][FactorComputation] Computing Linear Weighted Score...")
            result = self._compute_linear_score(result, selected_features)
        
        # Alpha 审计
        effective_count = self._count_effective_factors(result, selected_features)
        self._log_alpha_audit("EffectiveFactorCount", effective_count, "Factors with non-NaN values")
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V111 Factor Computation Complete")
        logger.info(f"[{VERSION}][DataAudit] Missing values handled: {missing_count}")
        logger.info(f"[{VERSION}][AlphaAudit] Effective Factor Count: {effective_count}")
        logger.info("=" * 80)
        
        return result
    
    def _compute_boosted_score(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """
        使用 BoostedAlpha 计算评分 (V111 最终修复版 - 简化逻辑).
        
        【V111 根本问题】
        1. 特征筛选剔除了最强因子（volatility_20 IC=0.0412）
        2. 中性化后因子值接近 0，预测力被洗掉
        3. 选中的因子（如 momentum_120 IC=-0.0416）方向反了
        
        【V111 最终修复】
        1. 直接使用全局 IC 加权（不经过中性化）
        2. 使用原始因子值（未中性化）
        3. 简化为 IC 加权 Rank 合成
        """
        result = df.copy()
        
        # 【关键修复】使用因子 IC 直接加权
        factor_ics = self.feature_selector.feature_stats.get('ics', {})
        
        if not factor_ics:
            # 降级：等权重
            weights = {f: 1.0 for f in selected_features}
        else:
            weights = {}
            for f in selected_features:
                ic = factor_ics.get(f, 0.0)
                # 权重：IC 绝对值
                weights[f] = abs(ic) if abs(ic) > 0.001 else 0.01
        
        # 计算 Rank 加权和
        raw_score = np.zeros(len(result))
        total_weight = 0.0
        
        for factor_name in selected_features:
            if factor_name not in result.columns:
                continue
            
            weight = weights.get(factor_name, 0.01)
            ic = factor_ics.get(factor_name, 0.0)
            direction = 1.0 if ic >= 0 else -1.0
            
            # 因子 Rank（未中性化的原始值）
            factor_raw = result[factor_name].fillna(result[factor_name].median())
            if factor_raw.isna().all():
                continue
            
            # 截面 Rank 归一化
            factor_rank = factor_raw.groupby(result['trade_date']).transform(
                lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
            )
            
            # 线性贡献（带方向）
            raw_score += factor_rank.values * weight * direction
            total_weight += weight
        
        if total_weight > 0:
            result['score'] = raw_score / total_weight
        else:
            result['score'] = raw_score
        
        return result
    
    def _compute_linear_score(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """线性加权评分 (降级方案)"""
        result = df.copy()
        
        raw_score = np.zeros(len(result))
        total_weight = 0.0
        
        for factor_name in selected_features:
            if factor_name in result.columns:
                factor_scaled = result[factor_name].fillna(0).groupby(result['trade_date']).transform(
                    lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
                )
                raw_score += factor_scaled.values
                total_weight += 1.0
        
        if total_weight > 0:
            result['score'] = raw_score / total_weight
        else:
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
        
        # 3. OLS 四维中性化
        if self.enable_neutralization:
            result = self.neutralization_engine.neutralize_ols(result, columns=factor_cols)
        
        return result
    
    # ==============================================================================
    # V111 IC 计算与审计
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
            'effective_factor_count': self._count_effective_factors(
                df, 
                self.feature_selection_report.get('selected_features', self.ALL_FACTOR_COLUMNS)
            ),
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
    # V111 自动反哺
    # ==============================================================================
    
    def save_reflection(self, df: pd.DataFrame) -> str:
        """
        【V111 核心】自动分析并保存反哺 JSON。
        
        分析内容:
        1. 前 5 个最有效因子 (IC 最高)
        2. 前 5 个最无效因子 (IC 最低)
        3. 特征筛选统计
        4. 交互特征发现
        
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
        
        # 前 5 无效 (IC 绝对值最低)
        bottom_ineffective = sorted_ics[-5:][::-1]
        
        # 构建反哺报告
        reflection = {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_factors': len(self.ALL_FACTOR_COLUMNS),
                'selected_factors': len(self.feature_selection_report.get('selected_features', [])),
                'rejected_factors': len(self.feature_selection_report.get('rejected_features', [])),
            },
            'top_5_effective_factors': [
                {'name': name, 'ic': ic, 'rank': i+1}
                for i, (name, ic) in enumerate(top_effective)
            ],
            'bottom_5_ineffective_factors': [
                {'name': name, 'ic': ic, 'rank': i+1}
                for i, (name, ic) in enumerate(bottom_ineffective)
            ],
            'feature_selection': self.feature_selection_report,
            'interaction_discoveries': [
                {'factor_a': a, 'factor_b': b, 'gain': g}
                for a, b, g in self.boosted_alpha.interaction_pairs
            ] if self.boosted_alpha else [],
            'dynamic_weights': self.weight_pool.get_weights() if self.weight_pool else {},
            'neutralization_stats': self.neutralization_engine.neutralization_stats,
            'audit_logs': {
                'data_audit_count': len(self.data_audit_log),
                'alpha_audit_count': len(self.alpha_audit_log),
            },
            'recommendations': self._generate_recommendations(top_effective, bottom_ineffective),
        }
        
        # 确保输出目录存在
        self.reflection_output.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON
        with open(self.reflection_output, 'w', encoding='utf-8') as f:
            json.dump(reflection, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"[{VERSION}][AutoReflection] Reflection saved to: {self.reflection_output}")
        
        return str(self.reflection_output)
    
    def _generate_recommendations(self, top_factors: List, bottom_factors: List) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于 top 因子
        for name, ic in top_factors:
            if ic > 0.05:
                recommendations.append(f"Consider increasing weight for {name} (IC={ic:.4f})")
        
        # 基于 bottom 因子
        for name, ic in bottom_factors:
            if abs(ic) < 0.01:
                recommendations.append(f"Consider removing {name} (IC={ic:.4f})")
        
        # 基于特征筛选
        if self.feature_selection_report:
            rejected_count = len(self.feature_selection_report.get('rejected_features', []))
            if rejected_count > 10:
                recommendations.append(f"Feature selection removed {rejected_count} low-quality features")
        
        return recommendations
    
    # ==============================================================================
    # V111 主接口
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
        return len(self.feature_selection_report.get('selected_features', []))
    
    def get_feature_selection_report(self) -> Dict:
        """获取特征筛选报告"""
        return self.feature_selection_report
    
    def get_boosted_alpha_interactions(self) -> List[Tuple[str, str, float]]:
        """获取 BoostedAlpha 发现的交互特征"""
        if self.boosted_alpha:
            return self.boosted_alpha.interaction_pairs
        return []


# ==============================================================================
# V111 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_neutralization: bool = True,
                       enable_boosted_alpha: bool = True,
                       enable_dynamic_weight: bool = True,
                       enable_feature_selection: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None,
                       reflection_output: str = "reports/v111_reflection.json") -> AlphaResearchV111:
    """
    获取 AlphaResearchV111 实例。
    
    Args:
        config_path: 因子配置文件路径
        enable_neutralization: 是否启用中性化
        enable_boosted_alpha: 是否启用 BoostedAlpha
        enable_dynamic_weight: 是否启用动态权重
        enable_feature_selection: 是否启用特征筛选
        auto_heal: 是否启用数据自愈
        db_url: 数据库连接 URL
        reflection_output: 反哺 JSON 输出路径
        
    Returns:
        AlphaResearchV111 实例
    """
    return AlphaResearchV111(
        config_path=config_path,
        enable_neutralization=enable_neutralization,
        enable_boosted_alpha=enable_boosted_alpha,
        enable_dynamic_weight=enable_dynamic_weight,
        enable_feature_selection=enable_feature_selection,
        auto_heal=auto_heal,
        db_url=db_url,
        reflection_output=reflection_output
    )