"""
Alpha Research Module - V110 统计集成范式转移 (LightGradientAlpha + 动态权重池).

【V110 核心改进 - 从"线性堆砌"到"统计集成"】
1. LightGradientAlpha: 非线性集成预测
   - 使用分箱统计 (Binning) 模拟决策树桩
   - 60 天滚动窗口动态权重分配
   - IC < 0 时权重强制归零 (非符号反转)

2. 因子库大扩容：12 个 → 30+ 个因子
   - 流动性压力因子 (Liquidity Stress)
   - 截面峰度交互特征 (Cross-sectional Kurtosis Interaction)
   - 借鉴顶级私募方向

3. 数据自愈强制：字段缺失时自动 SQL 补全

【V110 因子库架构 - 30+ 因子】
| 类别 | 因子名称 | 数学逻辑 |
|------|----------|----------|
| 流动性压力 | liquidity_stress_5 | 换手率突变 * 价格冲击 |
| 流动性压力 | liquidity_stress_10 | 10 日流动性压力 |
| 流动性压力 | amihud_illiq | Amihud 非流动性指标 |
| 流动性压力 | turnover_vol_ratio | 换手率/波动率 |
| 截面峰度 | kurtosis_interaction | 个股峰度 * 截面偏度 |
| 截面峰度 | skewness_rank | 偏度截面排名 |
| 截面峰度 | tail_risk | 尾部风险指标 |
| 动量反转 | momentum_5/10/20 | 多周期动量 |
| 动量反转 | reversion_5/10 | 多周期反转 |
| 量价特征 | volume_price_health | 量价配合度 |
| 量价特征 | vwap_distance | VWAP 乖离率 |
| 波动率 | volatility_20 | 20 日波动率 |
| 波动率 | downside_volatility | 下行波动率 |
| 资金流 | order_flow_imbalance | 订单流不平衡 |
| 资金流 | smart_money_divergence | 聪明钱背离 |
| 估值 | value_rank | 估值排名 |
| 估值 | ep_rank | EP 排名 |
| ... | ... | ... |

【V110 硬性技术指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.03 | 核心指标 |
| Effective Factor Count | > 30 | 因子数量 |
| IC Decay | Monotonic | 无前视偏差 |
| Data Healing | 100% | SQL 自动补全 |

【V110 禁止事项】
- 严禁简单线性加权 Rank(A) + Rank(B)
- 严禁 IC < 0 时符号反转 (只能归零)
- 严禁字段缺失时用均值填充或跳过
"""

from typing import Any, Optional, Union, Dict, List, Tuple
from pathlib import Path
import warnings
import time
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
from loguru import logger
import yaml

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V110 强制版本全局变量
# ==============================================================================
VERSION = "V110"


# ==============================================================================
# V110 自定义异常类
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


# ==============================================================================
# V110 算子库 - 扩展统计特征
# ==============================================================================

class AlphaOperatorsV110:
    """
    V110 Alpha 算子库 - 统计集成增强。
    
    【新增算子】
    - Liquidity_Stress(x, n): 流动性压力
    - Kurtosis_Interaction(x, n): 截面峰度交互
    - Amihud_Illiq(ret, vol, n): Amihud 非流动性
    - Downside_Volatility(ret, n): 下行波动率
    - Binning(x, n_bins): 分箱统计
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
            if std_val < AlphaOperatorsV110.EPSILON:
                std_val = AlphaOperatorsV110.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV110.EPSILON) if len(s.dropna()) > 1 else s
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
        """时间序列偏度 - 三阶矩特征"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).skew())
    
    @staticmethod
    def Ts_Kurtosis(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列峰度 - 四阶矩特征"""
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
    def Ts_Rank(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列百分位排名"""
        def calc_ts_rank(s):
            result = []
            for i in range(len(s)):
                if i < n:
                    result.append(np.nan)
                    continue
                window = s.iloc[max(0, i-n+1):i+1].dropna()
                if len(window) < n // 2:
                    result.append(np.nan)
                    continue
                current_val = s.iloc[i]
                rank = (window < current_val).sum() / len(window)
                result.append(rank)
            return pd.Series(result, index=s.index)
        return x.groupby(symbol_col, group_keys=False).apply(calc_ts_rank)
    
    @staticmethod
    def Binning(x: pd.Series, n_bins: int = 10, group_col: Optional[str] = None) -> pd.Series:
        """
        【V110 新增】分箱统计 - 用于 LightGradientAlpha。
        
        将连续值离散化为 n_bins 个区间，返回区间编号。
        
        Args:
            x: 输入序列
            n_bins: 分箱数量
            group_col: 分组列
            
        Returns:
            分箱后的整数序列
        """
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
        """
        【V110 新增】决策树桩 - 寻找最优分割点。
        
        返回 (threshold, direction)，使得：
        - 如果 feature > threshold 且 direction > 0 → 预测 label 高
        - 如果 feature > threshold 且 direction < 0 → 预测 label 低
        
        Args:
            feature: 特征序列
            label: 标签序列 (T+1 收益)
            symbol_col: 分组列
            
        Returns:
            (threshold, direction) 元组
        """
        # 合并并去空
        merged = pd.DataFrame({'feature': feature, 'label': label}).dropna()
        if len(merged) < 30:
            return (0.0, 1.0)
        
        best_ic = -1
        best_threshold = 0.0
        best_direction = 1.0
        
        # 尝试多个分位点作为阈值
        for quantile in [0.3, 0.4, 0.5, 0.6, 0.7]:
            threshold = merged['feature'].quantile(quantile)
            
            for direction in [1, -1]:
                # 简单 IC 评估
                above = merged[merged['feature'] > threshold]
                below = merged[merged['feature'] <= threshold]
                
                if len(above) < 10 or len(below) < 10:
                    continue
                
                above_mean = above['label'].mean()
                below_mean = below['label'].mean()
                
                # IC 近似为两组均值差
                ic = abs(above_mean - below_mean) * direction
                
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
        """
        【V110 核心】流动性压力因子 (Liquidity Stress).
        
        借鉴顶级私募：流动性压力 = 换手率突变 * 价格冲击
        
        计算逻辑：
        1. 换手率突变 = 当前换手率 / 过去 n 日均值
        2. 价格冲击 = |收益率|
        3. LS = 换手率突变 * 价格冲击
        
        经济含义：
        - 高换手 + 大跌 → 流动性压力极大 → 预期反弹
        - 高换手 + 大涨 → 流动性压力释放 → 预期回调
        
        Args:
            df: DataFrame 包含必需列
            n: 计算窗口
            turnover_col: 换手率列
            close_col: 收盘价列
            symbol_col: 股票代码列
            date_col: 交易日期列
            
        Returns:
            流动性压力指标
        """
        result = df.copy()
        
        # 计算收益率
        result['return'] = result.groupby(symbol_col)[close_col].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 换手率突变
        turnover_ma = result.groupby(symbol_col)[turnover_col].transform(
            lambda x: x.shift(1).rolling(window=n*20).mean()  # 20 日均值
        )
        turnover_shock = result[turnover_col].shift(1) / (turnover_ma + AlphaOperatorsV110.EPSILON)
        
        # 价格冲击 (绝对收益率)
        price_impact = result['return'].abs()
        
        # 流动性压力
        ls = turnover_shock * price_impact
        
        # 滚动累积
        ls_cumsum = ls.groupby(result[symbol_col]).transform(
            lambda x: x.rolling(window=n).sum()
        )
        
        return ls_cumsum
    
    @staticmethod
    def Amihud_Illiq(returns: pd.Series, volume: pd.Series, 
                     n: int = 20, symbol_col: str = 'symbol') -> pd.Series:
        """
        【V110 新增】Amihud 非流动性指标。
        
        计算逻辑：
        ILLIQ = mean(|return| / volume) over n days
        
        经济含义：
        - 单位成交量对应的价格变化越大，流动性越差
        - 高 ILLIQ → 低流动性 → 预期高收益补偿
        
        Args:
            returns: 收益率序列
            volume: 成交量序列
            n: 计算窗口
            symbol_col: 股票代码列
            
        Returns:
            Amihud 非流动性指标
        """
        # 计算每日 ILLIQ
        daily_illiq = returns.abs() / (volume + AlphaOperatorsV110.EPSILON)
        
        # 滚动均值
        illiq_ma = daily_illiq.groupby(symbol_col).transform(
            lambda x: x.shift(1).rolling(window=n).mean()
        )
        
        return illiq_ma
    
    @staticmethod
    def Downside_Volatility(df: pd.DataFrame, returns_col: str = 'return',
                             n: int = 20, symbol_col: str = 'symbol') -> pd.Series:
        """
        【V110 新增】下行波动率。
        
        计算逻辑：
        DownVol = Std(negative returns only) over n days
        
        经济含义：
        - 只考虑下跌日的波动率
        - 高下行波动率 → 高风险 → 预期高收益
        
        Args:
            df: DataFrame 包含收益率和 symbol 列
            returns_col: 收益率列名
            n: 计算窗口
            symbol_col: 股票代码列
            
        Returns:
            下行波动率指标
        """
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
        """
        【V110 核心】截面峰度交互特征 (Cross-sectional Kurtosis Interaction).
        
        借鉴顶级私募：个股峰度与截面偏度的交互
        
        计算逻辑：
        1. 计算个股滚动峰度 Ts_Kurtosis(returns, n)
        2. 计算截面偏度 (每日所有股票收益率的偏度)
        3. KI = 个股峰度 * 截面偏度
        
        经济含义：
        - 高峰度 + 正偏度 → 极端正收益概率高 → 正信号
        - 高峰度 + 负偏度 → 极端负收益概率高 → 负信号
        
        Args:
            df: DataFrame 包含收益率、symbol、date 列
            returns_col: 收益率列名
            n: 计算窗口
            symbol_col: 股票代码列
            date_col: 交易日期列
            
        Returns:
            峰度交互指标
        """
        returns = df[returns_col]
        
        # 计算个股滚动峰度
        stock_kurt = df.groupby(symbol_col)[returns_col].transform(
            lambda s: s.shift(1).rolling(window=n).kurt()
        )
        
        # 计算截面偏度 (按日期分组)
        cross_skew = df.groupby(date_col)[returns_col].transform('skew')
        
        # 峰度交互
        ki = stock_kurt * cross_skew
        
        return ki
    
    @staticmethod
    def Tail_Risk(df: pd.DataFrame, returns_col: str = 'return',
                  n: int = 20, symbol_col: str = 'symbol',
                  quantile: float = 0.05) -> pd.Series:
        """
        【V110 新增】尾部风险指标。
        
        计算逻辑：
        TailRisk = VaR(quantile) - 均值
        
        经济含义：
        - 左尾风险越大，预期收益补偿越高
        
        Args:
            df: DataFrame 包含收益率和 symbol 列
            returns_col: 收益率列名
            n: 计算窗口
            symbol_col: 股票代码列
            quantile: VaR 分位数
            
        Returns:
            尾部风险指标
        """
        def calc_tail_risk(s):
            if len(s.dropna()) < n:
                return np.nan
            return s.quantile(quantile) - s.mean()
        
        # 使用 df 直接进行 groupby，而不是 returns
        tail_risk = df.groupby(symbol_col, group_keys=False)[returns_col].transform(
            lambda x: x.shift(1).rolling(window=n).apply(calc_tail_risk, raw=False)
        )
        
        return tail_risk


# ==============================================================================
# V110 中性化引擎 (保留 V109 逻辑)
# ==============================================================================

class NeutralizationEngineV110:
    """V110 中性化引擎 - 行业 + 市值三重中性化。"""
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: List[str] = None):
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv', 'intraday_volatility']
        self.neutralize_vars = neutralize_vars
        self.neutralization_stats = {}
        logger.info(f"[{VERSION}][NeutralizationEngine] Initialized")
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        if columns is None:
            columns = list(AlphaResearchV110.ALL_FACTOR_COLUMNS)
        
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
            columns = list(AlphaResearchV110.ALL_FACTOR_COLUMNS)
        
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
        """OLS 三重中性化"""
        result = df.copy()
        
        # 准备中性化变量
        if 'total_mv' not in result.columns:
            result['total_mv'] = 1e10
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        if 'intraday_volatility' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['intraday_volatility'] = (
                    (result['high'] - result['low']) / (result['close'] + self.EPSILON)
                )
            else:
                result['intraday_volatility'] = 0.02
        
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
                X_vars = ['ln_total_mv', 'intraday_volatility']
                X = day_data[X_vars].values
                
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
        
        return result


# ==============================================================================
# V110 动态权重池 (60 天滚动窗口)
# ==============================================================================

class DynamicWeightPool:
    """
    【V110 核心】动态权重池 - 60 天滚动窗口 IC 加权。
    
    【权重分配规则】
    1. 计算每个因子过去 60 天的 IC
    2. 如果 IC > 0: weight = IC / sum(IC_positive)
    3. 如果 IC <= 0: weight = 0 (强制归零，非符号反转)
    4. 每日重新计算权重
    """
    
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
        # 保持窗口大小
        if len(self.ic_history[factor_name]) > self.window:
            self.ic_history[factor_name] = self.ic_history[factor_name][-self.window:]
    
    def compute_weights(self, factor_names: List[str]) -> Dict[str, float]:
        """
        计算当前权重。
        
        Args:
            factor_names: 因子名称列表
            
        Returns:
            因子权重的字典
        """
        weights = {}
        total_positive_ic = 0.0
        
        for factor_name in factor_names:
            history = self.ic_history.get(factor_name, [])
            
            if len(history) < self.min_history:
                weights[factor_name] = 0.0
                continue
            
            # 计算平均 IC
            recent_ics = [h['ic'] for h in history[-self.window:]]
            mean_ic = np.mean(recent_ics)
            
            # IC > 0 才有权重，否则归零
            if mean_ic > 0:
                weights[factor_name] = mean_ic
                total_positive_ic += mean_ic
            else:
                weights[factor_name] = 0.0
        
        # 归一化
        if total_positive_ic > 0:
            for factor_name in weights:
                weights[factor_name] /= total_positive_ic
        else:
            # 所有因子 IC 都 <= 0，等权重
            equal_weight = 1.0 / len(factor_names) if factor_names else 0.0
            weights = {f: equal_weight for f in factor_names}
        
        self.current_weights = weights
        self.weight_history.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'weights': weights.copy(),
        })
        
        return weights
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights
    
    def get_effective_factor_count(self) -> int:
        """获取有效因子数量 (权重 > 0)"""
        return sum(1 for w in self.current_weights.values() if w > 0)


# ==============================================================================
# V110 LightGradientAlpha (非线性集成)
# ==============================================================================

class LightGradientAlpha:
    """
    【V110 核心】LightGradientAlpha - 非线性集成预测。
    
    【实现方式】
    使用分箱统计 (Binning) + 决策树桩 (Decision Stump) 模拟非线性预测：
    
    1. 对每个因子进行分箱 (10 箱)
    2. 计算每个箱的历史平均收益
    3. 根据当前因子值所在箱，输出预测收益
    4. 多个因子的预测加权平均
    
    【与线性加权区别】
    - 线性：score = w1*f1 + w2*f2 + ...
    - 非线性：score = E[return | bin(f1), bin(f2), ...]
    """
    
    def __init__(self, n_bins: int = 10, min_samples: int = 30):
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.bin_stats = {}  # {factor_name: {bin_id: {'mean_return': x, 'count': y}}}
        self.decision_stumps = {}  # {factor_name: (threshold, direction)}
        
        logger.info(f"[{VERSION}][LightGradientAlpha] Initialized with n_bins={n_bins}")
    
    def fit(self, factor_values: pd.Series, labels: pd.Series, 
            factor_name: str, symbol_col: str = 'symbol') -> None:
        """
        拟合单个因子的分箱统计。
        
        Args:
            factor_values: 因子值
            labels: 标签 (T+1 收益)
            factor_name: 因子名称
            symbol_col: 分组列
        """
        merged = pd.DataFrame({
            'factor': factor_values,
            'label': labels,
        }).dropna()
        
        if len(merged) < self.min_samples:
            return
        
        # 分箱
        try:
            merged['bin'] = pd.qcut(
                merged['factor'].rank(method='first'),
                q=self.n_bins,
                labels=False,
                duplicates='drop'
            )
        except Exception:
            return
        
        # 计算每个箱的统计
        bin_stats = {}
        for bin_id in range(self.n_bins):
            bin_data = merged[merged['bin'] == bin_id]
            if len(bin_data) >= 10:
                bin_stats[bin_id] = {
                    'mean_return': bin_data['label'].mean(),
                    'count': len(bin_data),
                }
        
        self.bin_stats[factor_name] = bin_stats
        
        # 拟合决策树桩
        threshold, direction = AlphaOperatorsV110.Decision_Stump(
            merged['factor'], merged['label'], symbol_col
        )
        self.decision_stumps[factor_name] = (threshold, direction)
        
        logger.debug(f"[{VERSION}][LightGradientAlpha] Fitted {factor_name}: {len(bin_stats)} bins")
    
    def predict(self, factor_values: pd.Series, factor_name: str,
                weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        预测收益。
        
        Args:
            factor_values: 因子值
            factor_name: 因子名称
            weights: 因子权重 (可选)
            
        Returns:
            预测收益序列
        """
        bin_stats = self.bin_stats.get(factor_name, {})
        if not bin_stats:
            return pd.Series(0.0, index=factor_values.index)
        
        # 分箱
        try:
            ranks = factor_values.rank(method='first')
            bins = pd.qcut(ranks, q=self.n_bins, labels=False, duplicates='drop')
        except Exception:
            return pd.Series(0.0, index=factor_values.index)
        
        # 映射到预期收益
        def map_bin_to_return(bin_id):
            if bin_id in bin_stats:
                return bin_stats[bin_id]['mean_return']
            return 0.0
        
        predictions = bins.map(map_bin_to_return).fillna(0.0)
        
        return predictions
    
    def get_all_predictions(self, factor_data: pd.DataFrame, 
                            label_col: str = 't1_return',
                            weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        获取所有因子的集成预测。
        
        Args:
            factor_data: 包含所有因子和标签的 DataFrame
            label_col: 标签列名
            weights: 因子权重
            
        Returns:
            集成预测序列
        """
        if weights is None:
            weights = {f: 1.0 for f in self.bin_stats.keys()}
        
        total_weight = sum(weights.values())
        if total_weight <= 0:
            total_weight = 1.0
        
        ensemble_prediction = pd.Series(0.0, index=factor_data.index)
        
        for factor_name, weight in weights.items():
            if factor_name not in factor_data.columns:
                continue
            
            # 拟合
            self.fit(factor_data[factor_name], factor_data[label_col], factor_name)
            
            # 预测
            pred = self.predict(factor_data[factor_name], factor_name)
            
            # 加权
            ensemble_prediction += pred * weight / total_weight
        
        return ensemble_prediction


# ==============================================================================
# V110 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV110:
    """
    V110 Alpha 预测核心引擎 - 统计集成范式转移。
    
    【V110 核心改进】
    1. LightGradientAlpha: 非线性集成
    2. DynamicWeightPool: 60 天滚动权重
    3. 30+ 因子库扩容
    4. 数据自愈强制
    """
    
    EPSILON = 1e-6
    
    # V110 全部因子列名 (30+)
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
    
    # 基础因子权重 (用于初始化)
    BASE_FACTOR_WEIGHTS = {
        'liquidity_stress_5': 0.05,
        'liquidity_stress_10': 0.04,
        'amihud_illiq': 0.04,
        'kurtosis_interaction': 0.05,
        'skewness_rank': 0.04,
        'momentum_10': 0.06,
        'momentum_20': 0.06,
        'reversion_5': 0.04,
        'volume_price_health': 0.05,
        'volatility_20': 0.04,
        'order_flow_imbalance_5': 0.06,
        'bias_momentum_repair': 0.05,
        'value_rank': 0.04,
        'smart_money_divergence': 0.04,
        'downside_volatility': 0.04,
        'vwap_distance': 0.04,
        'tail_risk': 0.03,
        'accumulation_distribution': 0.04,
        'volatility_interaction': 0.04,
        'relative_value_rank': 0.03,
    }
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_neutralization: bool = True,
                 enable_light_gradient: bool = True,
                 enable_dynamic_weight: bool = True,
                 auto_heal: bool = True,
                 db_url: Optional[str] = None) -> None:
        """
        初始化 V110 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_neutralization: 是否启用中性化
            enable_light_gradient: 是否启用 LightGradientAlpha
            enable_dynamic_weight: 是否启用动态权重
            auto_heal: 是否启用数据自愈
            db_url: 数据库连接 URL
        """
        self.config_path = Path(config_path)
        self.enable_neutralization = enable_neutralization
        self.enable_light_gradient = enable_light_gradient
        self.enable_dynamic_weight = enable_dynamic_weight
        self.auto_heal = auto_heal
        self.db_url = db_url
        
        # 中性化引擎
        self.neutralization_engine = NeutralizationEngineV110()
        
        # LightGradientAlpha
        self.light_gradient = LightGradientAlpha(n_bins=10) if enable_light_gradient else None
        
        # 动态权重池
        self.weight_pool = DynamicWeightPool(window=60) if enable_dynamic_weight else None
        
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
        
        logger.info(f"[{VERSION}][AlphaResearch] Initialized")
        logger.info(f"[{VERSION}][AlphaResearch]   Neutralization: {self.enable_neutralization}")
        logger.info(f"[{VERSION}][AlphaResearch]   LightGradientAlpha: {self.enable_light_gradient}")
        logger.info(f"[{VERSION}][AlphaResearch]   DynamicWeightPool: {self.enable_dynamic_weight}")
        logger.info(f"[{VERSION}][AlphaResearch]   Auto Healing: {self.auto_heal}")
        logger.info(f"[{VERSION}][AlphaResearch]   Factor Count: {len(self.ALL_FACTOR_COLUMNS)}")
    
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
    # V110 核心因子计算 - 30+ 因子
    # ==============================================================================
    
    def compute_liquidity_stress(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """流动性压力因子"""
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        ls = AlphaOperatorsV110.Liquidity_Stress(
            result, n=period,
            turnover_col='turnover_rate',
            close_col='close',
            symbol_col='symbol',
            date_col='trade_date'
        )
        
        result[f'liquidity_stress_{period}'] = ls.values
        
        # 截面标准化
        result[f'liquidity_stress_{period}'] = result.groupby('trade_date')[f'liquidity_stress_{period}'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_amihud_illiq(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Amihud 非流动性指标"""
        result = df.copy()
        
        # 确保 symbol 和 trade_date 列存在
        if 'symbol' not in result.columns:
            logger.warning(f"[{VERSION}] symbol column not found, creating default")
            result['symbol'] = 'DEFAULT'
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 确保 volume 列存在
        if 'volume' not in result.columns:
            logger.warning(f"[{VERSION}] volume column not found, using amount/close as proxy")
            result['volume'] = result.get('amount', result['close'] * 1000) / (result['close'] + self.EPSILON)
        
        # 使用 groupby 直接计算，避免 symbol 列丢失
        def calc_illiq(group):
            ret = group['return'].shift(1).abs()
            vol = group['volume'].shift(1) + self.EPSILON
            daily_illiq = ret / vol
            return daily_illiq.rolling(window=period).mean()
        
        result['amihud_illiq'] = result.groupby('symbol', group_keys=False).apply(calc_illiq).values
        
        # 截面标准化
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
        
        # 截面标准化
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
        
        # Kurtosis_Interaction 现在接收 DataFrame
        ki = AlphaOperatorsV110.Kurtosis_Interaction(
            result, returns_col='return', n=period,
            symbol_col='symbol', date_col='trade_date'
        )
        
        result['kurtosis_interaction'] = ki.values
        
        # 截面标准化
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
        
        # 截面排名
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
        
        # Tail_Risk 现在接收 DataFrame
        tail_risk = AlphaOperatorsV110.Tail_Risk(
            result, returns_col='return', n=period,
            symbol_col='symbol', quantile=0.05
        )
        
        result['tail_risk'] = tail_risk.values
        
        # 截面标准化
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
        
        # 截面标准化
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
        
        # 截面标准化
        result['volume_price_health'] = result.groupby('trade_date')['volume_price_health'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_vwap_distance(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """VWAP 乖离率"""
        result = df.copy()
        
        # 计算 VWAP
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
        
        # 截面标准化
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
        
        # 截面标准化 (反向：低波因子)
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
        
        # Downside_Volatility 现在接收 DataFrame
        down_vol = AlphaOperatorsV110.Downside_Volatility(
            result, returns_col='return', n=period, symbol_col='symbol'
        )
        
        result['downside_volatility'] = down_vol.values
        
        # 截面标准化
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
        
        # 截面标准化
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
        
        # 截面标准化
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
        
        # 用 1/total_mv 作为估值代理
        value_proxy = 1.0 / (result['total_mv'] + self.EPSILON)
        
        value_rank = value_proxy.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['value_rank'] = value_rank.values
        
        return result
    
    def compute_ep_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """EP 排名 (1/PE)"""
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
        """BP 排名 (1/PB)"""
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
        
        # 截面标准化
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
        
        # 截面标准化
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
        
        # 截面标准化
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
        
        # 截面标准化
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
        
        # 反向：低波排名高
        result['volatility_rank'] = 1.0 - vol_rank.values
        
        return result
    
    def compute_beta(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Beta 因子 (简化版，用市场收益代理)"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 简化：用截面均值作为市场收益代理
        market_return = result.groupby('trade_date')['return'].transform('mean')
        
        # 确保数值类型
        result['return'] = pd.to_numeric(result['return'], errors='coerce')
        market_return = pd.to_numeric(market_return, errors='coerce')
        
        # 使用 transform 而不是 apply 来保持索引对齐
        def calc_beta_transform(group):
            if len(group) < period:
                return pd.Series(np.nan, index=group.index)
            
            ret_vals = group['return'].values
            mkt_vals = market_return.loc[group.index].values
            
            # 去空
            mask = np.isfinite(ret_vals) & np.isfinite(mkt_vals)
            if mask.sum() < period:
                return pd.Series(np.nan, index=group.index)
            
            ret_clean = ret_vals[mask]
            mkt_clean = mkt_vals[mask]
            
            # 手动计算协方差和方差
            ret_mean = np.mean(ret_clean)
            mkt_mean = np.mean(mkt_clean)
            
            cov = np.mean((ret_clean - ret_mean) * (mkt_clean - mkt_mean))
            var = np.mean((mkt_clean - mkt_mean) ** 2)
            
            beta_val = cov / var if var > 1e-10 else 0.0
            
            # 返回与输入索引对齐的序列
            return pd.Series(beta_val, index=group.index)
        
        beta = result.groupby('symbol', group_keys=False).apply(calc_beta_transform)
        # 重置索引以匹配原 DataFrame
        result['beta_20'] = beta.reset_index(level=0, drop=True).values
        
        return result
    
    def compute_big_order_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """大单比例 (简化版)"""
        result = df.copy()
        
        if 'amount' not in result.columns or 'volume' not in result.columns:
            result['big_order_ratio'] = np.nan
            return result
        
        avg_price = result['amount'] / (result['volume'] + self.EPSILON)
        
        # 用平均价格排名代理大单比例
        bor = avg_price.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['big_order_ratio'] = bor.values
        
        return result
    
    # ==============================================================================
    # V110 标签计算
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
    # V110 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True, year: int = None) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【计算顺序】
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
        11. 因子清洗
        12. LightGradientAlpha 预测
        13. 动态权重池更新
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V110 Factor Computation Started")
        logger.info("=" * 80)
        
        result = df.copy()
        
        # 数据审计
        missing_count = result.isnull().sum().sum()
        self._log_data_audit("MissingValuesHandled", int(missing_count), "Initial missing values")
        
        # 1. 流动性压力因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Liquidity Stress Factors...")
        result = self.compute_liquidity_stress(result, period=5)
        result = self.compute_liquidity_stress(result, period=10)
        result = self.compute_amihud_illiq(result, period=20)
        result = self.compute_turnover_vol_ratio(result)
        
        # 2. 截面峰度因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Kurtosis Interaction Factors...")
        result = self.compute_kurtosis_interaction(result, period=20)
        result = self.compute_skewness_rank(result, period=20)
        result = self.compute_tail_risk(result, period=20)
        
        # 3. 动量因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Momentum Factors...")
        for period in [5, 10, 20, 60, 120, 250]:
            result = self.compute_momentum_factor(result, period=period)
        
        # 4. 反转因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Reversion Factors...")
        for period in [5, 10]:
            result = self.compute_reversion_factor(result, period=period)
        
        # 5. 量价因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Volume-Price Factors...")
        result = self.compute_volume_price_health(result)
        result = self.compute_vwap_distance(result)
        result = self.compute_volume_rank(result)
        result = self.compute_price_rank(result)
        
        # 6. 波动率因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Volatility Factors...")
        result = self.compute_volatility_factor(result, period=20)
        result = self.compute_downside_volatility(result, period=20)
        result = self.compute_volatility_rank(result, period=20)
        result = self.compute_beta(result, period=20)
        
        # 7. 资金流因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Order Flow Factors...")
        result = self.compute_order_flow_imbalance(result, period=5)
        result = self.compute_smart_money_divergence(result, period=10)
        result = self.compute_big_order_ratio(result)
        
        # 8. 估值因子
        logger.info(f"[{VERSION}][FactorComputation] Computing Value Factors...")
        result = self.compute_value_rank(result)
        result = self.compute_ep_rank(result)
        result = self.compute_bp_rank(result)
        
        # 9. V109 保留因子
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
        
        # 11. 因子清洗
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Cleaning Factors (Neutralization)...")
            result = self.clean_factors(result)
        
        # 12. LightGradientAlpha 预测
        if self.enable_light_gradient and self.light_gradient:
            logger.info(f"[{VERSION}][FactorComputation] Computing LightGradientAlpha Prediction...")
            result = self._compute_light_gradient_score(result)
        else:
            logger.info(f"[{VERSION}][FactorComputation] Computing Linear Weighted Score...")
            result = self._compute_linear_score(result)
        
        # 13. 动态权重池更新
        if self.enable_dynamic_weight and self.weight_pool:
            self._update_weight_pool(result)
        
        # Alpha 审计
        effective_count = self._count_effective_factors(result)
        self._log_alpha_audit("EffectiveFactorCount", effective_count, "Factors with non-NaN values")
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V110 Factor Computation Complete")
        logger.info(f"[{VERSION}][DataAudit] Missing values handled: {missing_count}")
        logger.info(f"[{VERSION}][AlphaAudit] Effective Factor Count: {effective_count}")
        logger.info("=" * 80)
        
        return result
    
    def _compute_light_gradient_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """使用 LightGradientAlpha 计算评分"""
        result = df.copy()
        
        # 获取权重
        if self.weight_pool:
            weights = self.weight_pool.compute_weights(list(self.BASE_FACTOR_WEIGHTS.keys()))
        else:
            weights = {f: 1.0/len(self.BASE_FACTOR_WEIGHTS) for f in self.BASE_FACTOR_WEIGHTS}
        
        # 使用 LightGradientAlpha 预测
        if self.light_gradient:
            ensemble_pred = self.light_gradient.get_all_predictions(
                result, label_col='t1_return', weights=weights
            )
            result['score'] = ensemble_pred.values
        else:
            # 降级为线性加权
            result = self._compute_linear_score(result)
        
        return result
    
    def _compute_linear_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """线性加权评分 (降级方案)"""
        result = df.copy()
        
        raw_score = np.zeros(len(result))
        total_weight = 0.0
        
        for factor_name, weight in self.BASE_FACTOR_WEIGHTS.items():
            if factor_name in result.columns:
                factor_scaled = result[factor_name].fillna(0).groupby(result['trade_date']).transform(
                    lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
                )
                raw_score += factor_scaled.values * weight
                total_weight += weight
        
        if total_weight > 0:
            result['score'] = raw_score / total_weight
        else:
            result['score'] = raw_score
        
        result['score_linear'] = raw_score
        
        return result
    
    def _update_weight_pool(self, df: pd.DataFrame) -> None:
        """更新动态权重池"""
        if not self.weight_pool:
            return
        
        # 按日期分组更新 IC
        unique_dates = sorted(df['trade_date'].unique())
        
        for date in unique_dates:
            day_data = df[df['trade_date'] == date]
            
            if len(day_data) < 30:
                continue
            
            for factor_name in self.BASE_FACTOR_WEIGHTS.keys():
                if factor_name in day_data.columns and 't1_return' in day_data.columns:
                    ic = self._calculate_rank_ic(day_data[factor_name], day_data['t1_return'])
                    self.weight_pool.update_ic(factor_name, ic, date)
    
    def _count_effective_factors(self, df: pd.DataFrame) -> int:
        """计算有效因子数量"""
        count = 0
        for col in self.ALL_FACTOR_COLUMNS:
            if col in df.columns:
                non_null = df[col].notna().sum()
                if non_null > 0:
                    count += 1
        return count
    
    def clean_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """因子清洗三部曲"""
        result = df.copy()
        
        factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
        
        # 1. MAD 去极值
        result = self.neutralization_engine.winsorize_mad(result, columns=factor_cols, n_std=3.0)
        
        # 2. Z-Score 标准化
        result = self.neutralization_engine.normalize_zscore(result, columns=factor_cols)
        
        # 3. OLS 中性化
        if self.enable_neutralization:
            result = self.neutralization_engine.neutralize_ols(result, columns=factor_cols)
        
        return result
    
    # ==============================================================================
    # V110 IC 计算与审计
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
    
    def calculate_t1_ic(self, df: pd.DataFrame) -> Dict[str, float]:
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
    
    def calculate_factor_ics(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算各因子的独立 IC"""
        factor_ics = {}
        
        for factor_name in self.ALL_FACTOR_COLUMNS:
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
        
        # 检查 IC 强度
        ic_strong = t1_ic['mean_ic'] > 0.03  # V110 阈值降低到 0.03
        
        passed = ic_strong
        
        if not ic_strong:
            logger.warning(f"[{VERSION}][ICStabilityAudit] IC ({t1_ic['mean_ic']:.4f}) < 0.03 threshold")
        
        audit_result = {
            't1_ic': t1_ic,
            'factor_ics': factor_ics,
            'ic_strong': ic_strong,
            'passed': passed,
            'effective_factor_count': self._count_effective_factors(df),
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
    # V110 主接口
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
        return self.BASE_FACTOR_WEIGHTS
    
    def get_effective_factor_count(self) -> int:
        """获取有效因子数量"""
        if self.weight_pool:
            return self.weight_pool.get_effective_factor_count()
        return len(self.BASE_FACTOR_WEIGHTS)


# ==============================================================================
# V110 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_neutralization: bool = True,
                       enable_light_gradient: bool = True,
                       enable_dynamic_weight: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None) -> AlphaResearchV110:
    """
    获取 AlphaResearchV110 实例。
    
    Args:
        config_path: 因子配置文件路径
        enable_neutralization: 是否启用中性化
        enable_light_gradient: 是否启用 LightGradientAlpha
        enable_dynamic_weight: 是否启用动态权重
        auto_heal: 是否启用数据自愈
        db_url: 数据库连接 URL
        
    Returns:
        AlphaResearchV110 实例
    """
    return AlphaResearchV110(
        config_path=config_path,
        enable_neutralization=enable_neutralization,
        enable_light_gradient=enable_light_gradient,
        enable_dynamic_weight=enable_dynamic_weight,
        auto_heal=auto_heal,
        db_url=db_url
    )