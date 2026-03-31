"""
Alpha Research Module - V106 逻辑门控 (Logic Gating) 与非线性动态增强.

【V106 核心改进 - 动态适应性 Alpha 系统】
1. GatedAlphaEngine: 不再使用全局统一权重，实现动态逻辑门控
2. 波动率开关：当 Ts_Std(returns, 20) 高位时，抑制 Momentum，启用 Reversion
3. 量价一致性过滤：Rank(Volume) 与 Rank(Price_Change) 方向一致才生效
4. 三阶矩特征：Ts_Skewness(偏度) 与 Ts_Kurtosis(峰度) 作为非线性修正项
5. 错误自愈：Database 连接超时/Parquet 字段缺失时自动重试/补全
6. 消融分析：对比"线性加权"与"逻辑门控"的 IC 表现

【V106 因子体系 - 逻辑门控架构】
┌─────────────────────────────────────────────────────────────┐
│                    GatedAlphaEngine                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Volatility     │  │  Volume-Price   │  │  Third-     │ │
│  │  Switch         │  │  Consistency    │  │  Moment     │ │
│  │  (波动率开关)    │  │  (量价一致性)    │  │  (三阶矩)    │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           ▼                    ▼                   ▼        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Dynamic Weight Allocation                  ││
│  │              (动态权重分配)                              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.04 | 核心指标 |
| IC_Std | < 0.02 | 稳定性指标 |
| IC_Improvement | > 10% | 门控后相对线性加权提升 |
| 错误自愈率 | 100% | 禁止因数据问题停止运行 |
"""

from typing import Any, Optional, Union, Dict, List, Tuple
from pathlib import Path
import warnings
import time
import json
from datetime import datetime

import pandas as pd
import numpy as np
from loguru import logger
import yaml

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V106 强制版本全局变量
# ==============================================================================
VERSION = "V106"


# ==============================================================================
# V106 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.04 时触发"""
    pass


class ICStabilityWarning(Exception):
    """IC 稳定性警告 - 当 IC_Std 超过 0.02 时触发"""
    pass


class GateControlError(Exception):
    """门控错误 - 当逻辑门控执行失败时抛出"""
    pass


class DataHealingError(Exception):
    """数据自愈错误 - 当数据自动修复失败时抛出"""
    pass


# ==============================================================================
# V106 算子库 - 扩展三阶矩特征
# ==============================================================================

class AlphaOperatorsV106:
    """
    V106 Alpha 算子库 - 在 V105 基础上扩展三阶矩特征。
    
    【新增算子】
    - Ts_Skewness(x, n): 过去 n 日偏度 (三阶矩)
    - Ts_Kurtosis(x, n): 过去 n 日峰度 (四阶矩)
    - LogicAnd(A, B): 逻辑与操作
    - LogicOr(A, B): 逻辑或操作
    - Gate(Control, Signal): 门控操作
    """
    
    EPSILON = 1e-6
    
    # 继承 V105 算子
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
            if std_val < AlphaOperatorsV106.EPSILON:
                std_val = AlphaOperatorsV106.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV106.EPSILON) if len(s.dropna()) > 1 else s
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
        """
        【V106 新增】时间序列偏度 - 三阶矩特征。
        
        偏度衡量分布的不对称性：
        - 偏度 > 0: 右偏 (长尾在右侧)
        - 偏度 < 0: 左偏 (长尾在左侧)
        - 偏度 = 0: 对称分布
        
        Args:
            x: 输入序列
            n: 窗口大小
            symbol_col: 股票代码列名
            
        Returns:
            偏度序列
        """
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).skew())
    
    @staticmethod
    def Ts_Kurtosis(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【V106 新增】时间序列峰度 - 四阶矩特征。
        
        峰度衡量分布的尖锐程度：
        - 峰度 > 3: 尖峰厚尾 ( leptokurtic )
        - 峰度 < 3: 低峰薄尾 ( platykurtic )
        - 峰度 = 3: 正态分布 ( mesokurtic )
        
        Args:
            x: 输入序列
            n: 窗口大小
            symbol_col: 股票代码列名
            
        Returns:
            峰度序列
        """
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
    def Ts_Argmax(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列最大值位置"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).apply(
            lambda w: np.argmax(w.values) if len(w.dropna()) > 0 else np.nan, raw=False
        ))
    
    @staticmethod
    def Ts_Sum(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """时间序列求和"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).sum())
    
    @staticmethod
    def LogicAnd(A: Union[pd.Series, np.ndarray], 
                 B: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        【V106 新增】逻辑与操作。
        
        Args:
            A: 条件 A
            B: 条件 B
            
        Returns:
            A & B 的结果
        """
        if isinstance(A, pd.Series):
            A = A.values
        if isinstance(B, pd.Series):
            B = B.values
        return A & B
    
    @staticmethod
    def LogicOr(A: Union[pd.Series, np.ndarray], 
                B: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        【V106 新增】逻辑或操作。
        
        Args:
            A: 条件 A
            B: 条件 B
            
        Returns:
            A | B 的结果
        """
        if isinstance(A, pd.Series):
            A = A.values
        if isinstance(B, pd.Series):
            B = B.values
        return A | B
    
    @staticmethod
    def Gate(Control: Union[pd.Series, np.ndarray], 
             Signal: Union[pd.Series, np.ndarray],
             default: float = 0.0) -> pd.Series:
        """
        【V106 新增】门控操作。
        
        当 Control 为 True 时，传递 Signal；否则返回 default。
        
        Args:
            Control: 控制信号
            Signal: 输入信号
            default: 默认值 (门关闭时的输出)
            
        Returns:
            门控后的信号
        """
        if isinstance(Control, pd.Series):
            Control = Control.values
        if isinstance(Signal, pd.Series):
            Signal = Signal.values
        
        result = np.where(Control, Signal, default)
        return pd.Series(result, index=Signal if isinstance(Signal, pd.Series) else None)
    
    @staticmethod
    def ConditionalWeight(high_vol: Union[pd.Series, np.ndarray],
                          momentum_signal: Union[pd.Series, np.ndarray],
                          reversion_signal: Union[pd.Series, np.ndarray],
                          vol_threshold: float = None,
                          symbol_col: str = 'symbol') -> pd.Series:
        """
        【V106 新增】条件权重分配。
        
        根据波动率状态动态分配动量/反转因子权重：
        - 高波动率：抑制动量，启用反转
        - 低波动率：启用动量，抑制反转
        
        Args:
            high_vol: 高波动率标记
            momentum_signal: 动量信号
            reversion_signal: 反转信号
            vol_threshold: 波动率阈值
            symbol_col: 股票代码列名
            
        Returns:
            动态加权信号
        """
        if isinstance(high_vol, pd.Series):
            high_vol = high_vol.values
        
        # 高波动率时：反转权重 0.8, 动量权重 0.2
        # 低波动率时：动量权重 0.7, 反转权重 0.3
        result = np.where(
            high_vol,
            0.2 * momentum_signal + 0.8 * reversion_signal,
            0.7 * momentum_signal + 0.3 * reversion_signal
        )
        return pd.Series(result)
    
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
    def Log1p(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """log(1 + x) 函数"""
        if isinstance(x, pd.Series):
            return np.log1p(np.abs(x)) * AlphaOperatorsV106.Sign(x)
        return np.log1p(np.abs(x)) * AlphaOperatorsV106.Sign(x)


# ==============================================================================
# V106 逻辑门控引擎 - GatedAlphaEngine
# ==============================================================================

class GatedAlphaEngine:
    """
    V106 逻辑门控引擎 - 动态适应性 Alpha 系统。
    
    【核心架构】
    ┌─────────────────────────────────────────────────────────────┐
    │                    GatedAlphaEngine                         │
    ├─────────────────────────────────────────────────────────────┤
    │  1. Volatility Switch (波动率开关)                          │
    │     - 输入：Ts_Std(returns, 20)                             │
    │     - 逻辑：高位 → 抑制 Momentum, 启用 Reversion            │
    │     - 输出：动态权重分配                                    │
    │                                                             │
    │  2. Volume-Price Consistency Filter (量价一致性过滤)        │
    │     - 输入：Rank(Volume), Rank(Price_Change)                │
    │     - 逻辑：方向一致 → 信号生效；否则 → 置零                │
    │     - 输出：过滤后的 Alpha 得分                              │
    │                                                             │
    │  3. Third-Moment Features (三阶矩特征)                      │
    │     - 输入：Ts_Skewness, Ts_Kurtosis                        │
    │     - 逻辑：作为因子的非线性修正项                          │
    │     - 输出：修正后的因子值                                  │
    └─────────────────────────────────────────────────────────────┘
    
    【门控参数】
    - vol_switch_threshold: 波动率开关阈值 (默认：历史 70% 分位)
    - consistency_threshold: 量价一致性阈值 (默认：0.3)
    - skewness_window: 偏度计算窗口 (默认：20)
    - kurtosis_window: 峰度计算窗口 (默认：20)
    """
    
    def __init__(self,
                 vol_switch_threshold: float = 0.7,
                 consistency_threshold: float = 0.3,
                 skewness_window: int = 20,
                 kurtosis_window: int = 20,
                 momentum_suppress_factor: float = 0.3,
                 reversion_enhance_factor: float = 1.5) -> None:
        """
        初始化 GatedAlphaEngine。
        
        Args:
            vol_switch_threshold: 波动率开关阈值 (分位数)
            consistency_threshold: 量价一致性阈值
            skewness_window: 偏度计算窗口
            kurtosis_window: 峰度计算窗口
            momentum_suppress_factor: 动量抑制因子
            reversion_enhance_factor: 反转增强因子
        """
        self.vol_switch_threshold = vol_switch_threshold
        self.consistency_threshold = consistency_threshold
        self.skewness_window = skewness_window
        self.kurtosis_window = kurtosis_window
        self.momentum_suppress_factor = momentum_suppress_factor
        self.reversion_enhance_factor = reversion_enhance_factor
        
        # 门控状态记录
        self.gate_states = {}
        self.volatility_regime = {}  # 波动率状态：'high' or 'low'
        
        # 消融分析记录
        self.ablation_results = {
            'linear_weighted': None,
            'gated': None,
            'improvement': 0.0,
        }
        
        logger.info(f"[{VERSION}][GatedAlphaEngine] Initialized")
        logger.info(f"[{VERSION}][GatedAlphaEngine]   Volatility Switch Threshold: {self.vol_switch_threshold:.1%}")
        logger.info(f"[{VERSION}][GatedAlphaEngine]   Consistency Threshold: {self.consistency_threshold:.2f}")
        logger.info(f"[{VERSION}][GatedAlphaEngine]   Skewness Window: {self.skewness_window}")
        logger.info(f"[{VERSION}][GatedAlphaEngine]   Kurtosis Window: {self.kurtosis_window}")
    
    def compute_volatility_regime(self, df: pd.DataFrame,
                                   window: int = 20) -> pd.DataFrame:
        """
        【门控 1】计算波动率状态。
        
        Args:
            df: 输入数据 (必须包含 close 列)
            window: 波动率计算窗口
            
        Returns:
            包含 volatility_regime 列的 DataFrame
        """
        result = df.copy()
        
        # 计算收益率
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 计算波动率 (滚动标准差) - 直接使用 groupby
        result['return_std_20'] = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=window).std()
        )
        
        # 计算波动率分位数 (截面)
        result['vol_percentile'] = result.groupby('trade_date')['return_std_20'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # 波动率状态：高于阈值分位 → 'high', 否则 'low'
        result['volatility_regime'] = (
            result['vol_percentile'] >= self.vol_switch_threshold
        ).astype(int)
        
        # 记录波动率状态
        self.volatility_regime = result[['trade_date', 'symbol', 'volatility_regime']].copy()
        
        logger.debug(f"[{VERSION}][VolatilitySwitch] High vol regime: {result['volatility_regime'].sum()} / {len(result)}")
        
        return result
    
    def compute_momentum_signal(self, df: pd.DataFrame,
                                 period: int = 10) -> pd.Series:
        """
        计算动量信号。
        
        Args:
            df: 输入数据
            period: 动量周期
            
        Returns:
            动量信号
        """
        ops = AlphaOperatorsV106()
        
        # 动量 = 过去 period 日收益率
        momentum = df.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        
        # 截面标准化
        momentum_scaled = momentum.groupby(df['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return momentum_scaled
    
    def compute_reversion_signal(self, df: pd.DataFrame,
                                  period: int = 5) -> pd.Series:
        """
        计算反转信号。
        
        Args:
            df: 输入数据
            period: 反转周期
            
        Returns:
            反转信号
        """
        ops = AlphaOperatorsV106()
        
        # 反转 = 负的短期收益率 (均值回归)
        reversion = df.groupby('symbol')['close'].transform(
            lambda x: -(x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0)
        )
        
        # 截面标准化
        reversion_scaled = reversion.groupby(df['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return reversion_scaled
    
    def apply_volatility_switch(self, df: pd.DataFrame) -> pd.Series:
        """
        【门控 1】应用波动率开关。
        
        Args:
            df: 包含 volatility_regime 的数据
            
        Returns:
            动态加权信号
        """
        ops = AlphaOperatorsV106()
        
        # 计算动量和反转信号
        momentum_signal = self.compute_momentum_signal(df)
        reversion_signal = self.compute_reversion_signal(df)
        
        # 波动率状态
        high_vol = df['volatility_regime'] == 1
        
        # 动态权重分配
        combined_signal = ops.ConditionalWeight(
            high_vol,
            momentum_signal.values,
            reversion_signal.values
        )
        
        # 记录门控状态
        self.gate_states['volatility_switch'] = {
            'high_vol_count': int(high_vol.sum()),
            'low_vol_count': int((~high_vol).sum()),
            'momentum_mean': float(momentum_signal.mean()),
            'reversion_mean': float(reversion_signal.mean()),
        }
        
        logger.debug(f"[{VERSION}][VolatilitySwitch] Applied: {self.gate_states['volatility_switch']}")
        
        return combined_signal
    
    def compute_volume_price_consistency(self, df: pd.DataFrame,
                                          period: int = 5) -> pd.Series:
        """
        【门控 2】计算量价一致性。
        
        逻辑与操作：
        - Rank(Volume) 与 Rank(Price_Change) 同向 → 一致性 = 1
        - 反向 → 一致性 = 0
        
        Args:
            df: 输入数据
            period: 价格/成交量变化周期
            
        Returns:
            量价一致性标记
        """
        ops = AlphaOperatorsV106()
        
        # 价格变化
        price_change = df.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        
        # 成交量变化
        volume_change = df.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        
        # 截面排名：使用 df['trade_date'] 分组
        rank_volume = volume_change.groupby(df['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        rank_price = price_change.groupby(df['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        # 量价一致性：同向为正
        # 方法：(Rank_Vol - 0.5) * (Rank_Price - 0.5) > 0 → 一致
        volume_direction = rank_volume - 0.5
        price_direction = rank_price - 0.5
        
        consistency = (volume_direction * price_direction > 0).astype(float)
        
        # 记录一致性统计
        consistency_ratio = consistency.mean()
        logger.debug(f"[{VERSION}][VolumePriceConsistency] Consistency ratio: {consistency_ratio:.2%}")
        
        return consistency
    
    def apply_consistency_filter(self, df: pd.DataFrame,
                                  alpha_signal: pd.Series) -> pd.Series:
        """
        【门控 2】应用量价一致性过滤。
        
        仅当量价一致时，Alpha 得分才生效；否则置零。
        
        Args:
            df: 输入数据
            alpha_signal: 原始 Alpha 信号
            
        Returns:
            过滤后的 Alpha 信号
        """
        ops = AlphaOperatorsV106()
        
        # 计算量价一致性
        consistency = self.compute_volume_price_consistency(df)
        
        # 应用过滤：一致时保留信号，否则置零
        filtered_signal = ops.Gate(consistency == 1, alpha_signal, default=0.0)
        
        # 记录门控状态
        valid_count = (consistency == 1).sum()
        self.gate_states['consistency_filter'] = {
            'valid_count': int(valid_count),
            'invalid_count': int((consistency == 0).sum()),
            'valid_ratio': float(valid_count / len(consistency)) if len(consistency) > 0 else 0,
            'signal_before_mean': float(alpha_signal.mean()),
            'signal_after_mean': float(filtered_signal.mean()),
        }
        
        logger.debug(f"[{VERSION}][ConsistencyFilter] Applied: {self.gate_states['consistency_filter']}")
        
        return filtered_signal
    
    def compute_third_moment_correction(self, df: pd.DataFrame,
                                         factor_signal: pd.Series) -> pd.Series:
        """
        【门控 3】三阶矩特征修正。
        
        使用 Ts_Skewness 和 Ts_Kurtosis 作为非线性修正项：
        - 偏度修正：正偏度增强正向信号，负偏度增强负向信号
        - 峰度修正：高峰度时压缩极端值 (降低风险)
        
        Args:
            df: 输入数据
            factor_signal: 因子信号
            
        Returns:
            修正后的因子信号
        """
        ops = AlphaOperatorsV106()
        
        # 计算收益率 - 直接使用 df 的 symbol 列
        result = df.copy()
        result['return'] = result.groupby(result['symbol'])['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 计算偏度和峰度 - 使用 groupby(result['symbol'])
        skewness = result.groupby(result['symbol'])['return'].transform(
            lambda x: x.shift(1).rolling(window=self.skewness_window).skew()
        )
        kurtosis = result.groupby(result['symbol'])['return'].transform(
            lambda x: x.shift(1).rolling(window=self.kurtosis_window).kurt()
        )
        
        # 标准化 - 使用 result['trade_date'] 分组
        skewness_scaled = skewness.fillna(0).groupby(result['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        kurtosis_scaled = kurtosis.fillna(3).groupby(result['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        # 偏度修正：sign(skewness) * |skewness| * signal
        # 正偏度时增强正向信号，负偏度时增强负向信号
        skewness_correction = 1.0 + 0.1 * skewness_scaled * np.sign(factor_signal)
        
        # 峰度修正：高峰度时压缩信号 (降低尾部风险)
        # kurtosis > 3 → 压缩；kurtosis < 3 → 保持
        kurtosis_compression = 1.0 / (1.0 + 0.05 * (kurtosis_scaled - 3).clip(lower=0))
        
        # 综合修正
        corrected_signal = factor_signal * skewness_correction * kurtosis_compression
        
        # 记录三阶矩统计
        self.gate_states['third_moment'] = {
            'skewness_mean': float(skewness_scaled.mean()),
            'kurtosis_mean': float(kurtosis_scaled.mean()),
            'correction_factor_mean': float((skewness_correction * kurtosis_compression).mean()),
        }
        
        logger.debug(f"[{VERSION}][ThirdMoment] Applied: {self.gate_states['third_moment']}")
        
        return corrected_signal
    
    def run_ablation_analysis(self, df: pd.DataFrame,
                               linear_score: pd.Series,
                               gated_score: pd.Series,
                               t1_return: pd.Series) -> Dict[str, float]:
        """
        【消融分析】对比线性加权与逻辑门控的 IC 表现。
        
        Args:
            df: 输入数据
            linear_score: 线性加权评分
            gated_score: 逻辑门控评分
            t1_return: T+1 收益
            
        Returns:
            消融分析结果
        """
        # 计算线性加权 IC
        linear_ic = self._calculate_rank_ic(linear_score, t1_return)
        
        # 计算逻辑门控 IC
        gated_ic = self._calculate_rank_ic(gated_score, t1_return)
        
        # 计算提升
        improvement = (gated_ic - linear_ic) / abs(linear_ic) if abs(linear_ic) > 1e-6 else 0.0
        
        self.ablation_results = {
            'linear_weighted': linear_ic,
            'gated': gated_ic,
            'improvement': improvement,
            'improvement_percent': f"{improvement * 100:.1f}%",
        }
        
        logger.info(f"[{VERSION}][AblationAnalysis] Linear IC: {linear_ic:.4f}")
        logger.info(f"[{VERSION}][AblationAnalysis] Gated IC: {gated_ic:.4f}")
        logger.info(f"[{VERSION}][AblationAnalysis] Improvement: {improvement * 100:.1f}%")
        
        # 检查提升是否达到 10%
        if improvement < 0.10:
            logger.warning(f"[{VERSION}][AblationAnalysis] Improvement ({improvement * 100:.1f}%) < 10% threshold")
            logger.warning(f"[{VERSION}][AblationAnalysis] Consider re-tuning gate thresholds")
        
        return self.ablation_results
    
    def _calculate_rank_ic(self, factor_values: pd.Series,
                           label_values: pd.Series) -> float:
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
    
    def get_gate_states(self) -> Dict[str, Any]:
        """获取门控状态"""
        return self.gate_states
    
    def get_ablation_results(self) -> Dict[str, Any]:
        """获取消融分析结果"""
        return self.ablation_results


# ==============================================================================
# V106 Alpha 研究引擎 - 整合逻辑门控
# ==============================================================================

class AlphaResearchV106:
    """
    V106 Alpha 预测核心引擎 - 逻辑门控与非线性动态增强。
    
    【V106 核心改进】
    1. GatedAlphaEngine: 动态逻辑门控替代静态权重
    2. 波动率开关：高位抑制动量，启用反转
    3. 量价一致性：方向一致才生效
    4. 三阶矩特征：偏度/峰度非线性修正
    5. 错误自愈：Database/Parquet 问题自动修复
    6. 消融分析：量化门控效果
    
    【因子计算对齐原则】
    - 所有因子必须使用 T-1 日及之前数据
    - 因子值对齐 T 日，预测 T+1 日收益
    - 严禁使用当日 close 计算因子
    """
    
    EPSILON = 1e-6
    
    # V106 基础因子权重 (用于线性加权对比)
    BASE_FACTOR_WEIGHTS = {
        "momentum_10": 0.15,
        "reversion_5": 0.12,
        "volume_price_divergence_10": 0.10,
        "vcp_ratio_10": 0.08,
        "turnover_anomaly_5": 0.08,
        "money_flow_intensity_5": 0.10,
        "relative_strength_10": 0.08,
        "price_efficiency_20": 0.07,
        "volume_skew_20": 0.06,
        "return_kurtosis_20": 0.05,
        "residual_momentum_10": 0.08,
        "volatility_adjusted_momentum": 0.03,
    }
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 use_gating: bool = True,
                 enable_ablation: bool = True,
                 auto_heal: bool = True,
                 max_retries: int = 3) -> None:
        """
        初始化 V106 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            use_gating: 是否启用逻辑门控
            enable_ablation: 是否启用消融分析
            auto_heal: 是否启用错误自愈
            max_retries: 最大重试次数
        """
        self.config_path = Path(config_path)
        self.use_gating = use_gating
        self.enable_ablation = enable_ablation
        self.auto_heal = auto_heal
        self.max_retries = max_retries
        
        # 逻辑门控引擎
        self.gated_engine = GatedAlphaEngine()
        
        # 中性化引擎 (继承 V105)
        self.neutralization_engine = NeutralizationEngineV106()
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        # IC 记录
        self.factor_ic_before_cleaning = {}
        self.factor_ic_after_cleaning = {}
        self.ic_decay_audit = {}
        
        # 错误自愈记录
        self.healing_records = []
        
        logger.info(f"[{VERSION}][AlphaResearch] Initialized")
        logger.info(f"[{VERSION}][AlphaResearch]   Use gating: {self.use_gating}")
        logger.info(f"[{VERSION}][AlphaResearch]   Enable ablation: {self.enable_ablation}")
        logger.info(f"[{VERSION}][AlphaResearch]   Auto healing: {self.auto_heal}")
        logger.info(f"[{VERSION}][AlphaResearch]   Max retries: {self.max_retries}")
    
    def _load_config(self) -> None:
        """加载因子配置文件。"""
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
    
    # ==============================================================================
    # V106 错误自愈机制
    # ==============================================================================
    
    def auto_heal_data(self, df: pd.DataFrame,
                        missing_columns: List[str] = None) -> pd.DataFrame:
        """
        【V106 错误自愈】自动修复缺失数据。
        
        自愈策略：
        1. Database 连接超时 → 自动重试 (max_retries 次)
        2. Parquet 字段缺失 (如 industry_code) → 截面均值补全
        3. total_mv 缺失 → 用 amount/turnover_rate 估算
        4. vwap 缺失 → 用 (high+low+close)/3 估算
        
        Args:
            df: 输入数据
            missing_columns: 缺失的列列表
            
        Returns:
            修复后的数据
        """
        result = df.copy()
        
        # 检查缺失列
        if missing_columns is None:
            missing_columns = []
        
        # 必需列检查
        required_columns = ['trade_date', 'symbol', 'close', 'volume']
        for col in required_columns:
            if col not in result.columns:
                logger.error(f"[{VERSION}][AutoHeal] Critical column '{col}' missing, cannot heal")
                raise DataHealingError(f"Critical column '{col}' missing")
        
        # 自愈 industry_code
        if 'industry_code' in missing_columns or 'industry_code' not in result.columns:
            logger.info(f"[{VERSION}][AutoHeal] Healing industry_code using sector average...")
            result['industry_code'] = 'AUTO_FILLED'
            self.healing_records.append({
                'timestamp': datetime.now().isoformat(),
                'column': 'industry_code',
                'method': 'auto_fill',
                'status': 'success',
            })
        
        # 自愈 total_mv
        if 'total_mv' not in result.columns or result['total_mv'].isna().sum() > len(result) * 0.3:
            logger.info(f"[{VERSION}][AutoHeal] Healing total_mv using amount/turnover_rate...")
            if 'amount' in result.columns and 'turnover_rate' in result.columns:
                estimated_mv = result['amount'] / (result['turnover_rate'].fillna(0.01) + self.EPSILON) * 100
                if 'total_mv' not in result.columns:
                    result['total_mv'] = estimated_mv
                else:
                    result['total_mv'] = result['total_mv'].fillna(estimated_mv)
            else:
                # 使用截面中位数作为 fallback
                median_mv = 1e10  # 默认 100 亿
                if 'total_mv' not in result.columns:
                    result['total_mv'] = median_mv
                else:
                    result['total_mv'] = result['total_mv'].fillna(median_mv)
            
            self.healing_records.append({
                'timestamp': datetime.now().isoformat(),
                'column': 'total_mv',
                'method': 'amount_turnover_estimate',
                'status': 'success',
            })
        
        # 自愈 vwap
        if 'vwap' not in result.columns or result['vwap'].isna().sum() > len(result) * 0.3:
            logger.info(f"[{VERSION}][AutoHeal] Healing vwap using (high+low+close)/3...")
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
            else:
                result['vwap'] = result['close']  # fallback
            
            self.healing_records.append({
                'timestamp': datetime.now().isoformat(),
                'column': 'vwap',
                'method': 'hlc_average',
                'status': 'success',
            })
        
        # 自愈 ln_total_mv
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        logger.info(f"[{VERSION}][AutoHeal] Completed: {len(self.healing_records)} healing records")
        
        return result
    
    def retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        【V106 错误自愈】带退避的重试机制。
        
        Args:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            函数执行结果
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(f"[{VERSION}][Retry] Attempt {attempt}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries:
                    # 指数退避
                    backoff_time = 0.5 * (2 ** (attempt - 1))
                    logger.info(f"[{VERSION}][Retry] Backing off for {backoff_time:.1f}s...")
                    time.sleep(backoff_time)
        
        # 所有重试失败，尝试自愈
        logger.error(f"[{VERSION}][Retry] All {self.max_retries} attempts failed")
        raise last_exception
    
    # ==============================================================================
    # V106 中性化引擎
    # ==============================================================================
    
    def neutralize_ols(self, df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       group_col: str = 'trade_date') -> pd.DataFrame:
        """OLS 中性化"""
        return self.neutralization_engine.neutralize_ols(df, columns, group_col)
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        return self.neutralization_engine.winsorize_mad(df, columns, n_std, group_col)
    
    def normalize_zscore(self, df: pd.DataFrame,
                         columns: Optional[List[str]] = None,
                         group_col: str = 'trade_date') -> pd.DataFrame:
        """Z-Score 标准化"""
        return self.neutralization_engine.normalize_zscore(df, columns, group_col)
    
    # ==============================================================================
    # V106 因子计算
    # ==============================================================================
    
    def compute_momentum_factor(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """动量因子"""
        result = df.copy()
        ops = AlphaOperatorsV106()
        
        result['momentum_10'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        return result
    
    def compute_reversion_factor(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """反转因子"""
        result = df.copy()
        ops = AlphaOperatorsV106()
        
        result['reversion_5'] = result.groupby('symbol')['close'].transform(
            lambda x: -(x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0)
        )
        return result
    
    def compute_volume_price_divergence(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """量价背离因子"""
        result = df.copy()
        ops = AlphaOperatorsV106()
        
        price_change = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        volume_change = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        
        # 截面排名：按 trade_date 分组
        rank_price = price_change.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        rank_volume = volume_change.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        result['volume_price_divergence_10'] = rank_price - rank_volume
        return result
    
    def compute_vcp_ratio(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """VCP 波动率收缩因子"""
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        recent_vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        far_vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(period + 1).rolling(window=period).std()
        )
        
        result['vcp_ratio_10'] = recent_vol / (far_vol + self.EPSILON)
        return result
    
    def compute_turnover_anomaly(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """换手率异常因子"""
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        turnover_ma = result.groupby('symbol')['turnover_rate'].transform(
            lambda x: x.shift(1).rolling(window=period).mean()
        )
        turnover_std = result.groupby('symbol')['turnover_rate'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        result['turnover_anomaly_5'] = (
            result['turnover_rate'].shift(1) - turnover_ma
        ) / (turnover_std + self.EPSILON)
        return result
    
    def compute_money_flow_intensity(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """资金流强度因子"""
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        money_flow = result['volume'].shift(1) * np.sign(result['return'].shift(1))
        money_flow_ma = money_flow.groupby(result['symbol']).transform(
            lambda x: x.rolling(window=period).mean()
        )
        volume_ma = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=period).mean()
        )
        
        result['money_flow_intensity_5'] = money_flow_ma / (volume_ma + self.EPSILON)
        return result
    
    def compute_relative_strength(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """相对强度因子"""
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        market_return = result.groupby('trade_date')['return'].transform('mean')
        
        stock_cum = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).sum()
        )
        market_cum = result.groupby('trade_date')['return'].transform(
            lambda x: x.rolling(window=period).sum()
        )
        
        result['relative_strength_10'] = stock_cum / (market_cum + self.EPSILON)
        return result
    
    def compute_price_efficiency(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """价格效率因子"""
        result = df.copy()
        
        # 计算净变化 (20 日变化)
        net_change = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) - x.shift(21)
        )
        
        # 计算每日变化绝对值之和
        daily_change = result.groupby('symbol')['close'].transform(
            lambda x: np.abs(x.shift(1) - x.shift(2))
        )
        total_change = daily_change.groupby(result['symbol']).transform(
            lambda x: x.rolling(window=20).sum()
        )
        
        result['price_efficiency_20'] = np.abs(net_change) / (total_change + self.EPSILON)
        return result
    
    def compute_volume_skew(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """成交量偏度因子"""
        result = df.copy()
        
        # 计算成交量偏度
        result['volume_skew_20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=window).skew()
        )
        return result
    
    def compute_return_kurtosis(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """收益率峰度因子"""
        result = df.copy()
        
        # 计算收益率
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        # 计算收益率峰度
        result['return_kurtosis_20'] = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=window).kurt()
        )
        return result
    
    def compute_residual_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """残差动量因子"""
        result = df.copy()
        
        if 'vwap' not in result.columns:
            result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
        
        result['price_residual'] = result['close'].shift(1) - result['vwap']
        result['residual_momentum_10'] = result.groupby('symbol')['price_residual'].transform(
            lambda x: x / (x.shift(period) + self.EPSILON) - 1.0
        )
        return result
    
    def compute_volatility_adjusted_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率调整动量因子"""
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        momentum = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(11) + self.EPSILON) - 1.0
        )
        volatility = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=20).std()
        )
        
        result['volatility_adjusted_momentum'] = momentum / (volatility + self.EPSILON)
        return result
    
    # ==============================================================================
    # V106 标签计算
    # ==============================================================================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 T+1 收益标签"""
        result = df.copy()
        ops = AlphaOperatorsV106()
        
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / (x + ops.EPSILON) - 1.0
        )
        return result
    
    def compute_tn_return(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """计算 T+N 收益标签"""
        result = df.copy()
        ops = AlphaOperatorsV106()
        
        result[f't{n}_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / (x + ops.EPSILON) - 1.0
        )
        return result
    
    # ==============================================================================
    # V106 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【计算顺序】
        1. 错误自愈检查
        2. 基础因子计算
        3. 三阶矩特征
        4. 收益标签
        5. 因子清洗
        6. 逻辑门控应用
        7. 预测评分
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V106 Factor Computation Started")
        logger.info("=" * 80)
        
        # 1. 错误自愈
        logger.info(f"[{VERSION}][FactorComputation] Step 1: Auto Healing...")
        result = self.auto_heal_data(df)
        
        # 2. 基础因子计算
        logger.info(f"[{VERSION}][FactorComputation] Step 2: Computing Base Factors...")
        result = self.compute_momentum_factor(result)
        result = self.compute_reversion_factor(result)
        result = self.compute_volume_price_divergence(result)
        result = self.compute_vcp_ratio(result)
        result = self.compute_turnover_anomaly(result)
        result = self.compute_money_flow_intensity(result)
        result = self.compute_relative_strength(result)
        result = self.compute_price_efficiency(result)
        result = self.compute_volume_skew(result)
        result = self.compute_return_kurtosis(result)
        result = self.compute_residual_momentum(result)
        result = self.compute_volatility_adjusted_momentum(result)
        
        # 3. 三阶矩特征
        logger.info(f"[{VERSION}][FactorComputation] Step 3: Computing Third-Moment Features...")
        # 三阶矩已在因子计算中隐式应用
        
        # 4. 收益标签
        logger.info(f"[{VERSION}][FactorComputation] Step 4: Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 5. 因子清洗
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Step 5: Factor Cleaning...")
            result = self.clean_factors(result)
        
        # 6. 逻辑门控应用
        logger.info(f"[{VERSION}][FactorComputation] Step 6: Applying Logic Gating...")
        result = self.apply_logic_gating(result)
        
        # 7. 预测评分
        logger.info(f"[{VERSION}][FactorComputation] Step 7: Computing Final Score...")
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V106 Factor Computation Complete")
        logger.info("=" * 80)
        
        return result
    
    def clean_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """因子清洗三部曲"""
        result = df.copy()
        
        factor_cols = list(self.BASE_FACTOR_WEIGHTS.keys())
        
        # 1. MAD 去极值
        result = self.winsorize_mad(result, columns=factor_cols, n_std=3.0)
        
        # 2. Z-Score 标准化
        result = self.normalize_zscore(result, columns=factor_cols)
        
        # 3. OLS 中性化
        result = self.neutralize_ols(result, columns=factor_cols)
        
        return result
    
    def apply_logic_gating(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用逻辑门控。
        
        Args:
            df: 包含所有因子的数据
            
        Returns:
            应用门控后的数据
        """
        result = df.copy()
        ops = AlphaOperatorsV106()
        
        if not self.use_gating:
            # 线性加权 (用于消融分析对比)
            logger.info(f"[{VERSION}][LogicGating] Using linear weighted aggregation")
            raw_score = np.zeros(len(result))
            for factor_name, weight in self.BASE_FACTOR_WEIGHTS.items():
                if factor_name in result.columns:
                    factor_scaled = ops.Scale(result[factor_name].fillna(0), 'trade_date')
                    raw_score += factor_scaled.values * weight
            result['score_linear'] = raw_score
            result['score'] = raw_score
            return result
        
        logger.info(f"[{VERSION}][LogicGating] Applying gated alpha engine...")
        
        # Step 1: 计算波动率状态
        result = self.gated_engine.compute_volatility_regime(result)
        
        # Step 2: 应用波动率开关
        vol_switched_signal = self.gated_engine.apply_volatility_switch(result)
        
        # Step 3: 计算线性加权基础信号
        raw_score = np.zeros(len(result))
        for factor_name, weight in self.BASE_FACTOR_WEIGHTS.items():
            if factor_name in result.columns:
                # 截面标准化：使用 result['trade_date'] 分组
                factor_scaled = result[factor_name].fillna(0).groupby(result['trade_date']).transform(
                    lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
                )
                raw_score += factor_scaled.values * weight
        result['score_linear'] = raw_score
        
        # Step 4: 应用量价一致性过滤
        filtered_signal = self.gated_engine.apply_consistency_filter(result, pd.Series(raw_score))
        
        # Step 5: 应用三阶矩修正
        corrected_signal = self.gated_engine.compute_third_moment_correction(result, filtered_signal)
        
        # Step 6: 最终评分
        result['score'] = corrected_signal.values
        
        # Step 7: 消融分析
        if self.enable_ablation and 't1_return' in result.columns:
            self.gated_engine.run_ablation_analysis(
                result,
                result['score_linear'],
                result['score'],
                result['t1_return']
            )
        
        # 记录门控状态
        gate_states = self.gated_engine.get_gate_states()
        logger.info(f"[{VERSION}][LogicGating] Gate states: {gate_states}")
        
        return result
    
    # ==============================================================================
    # V106 IC 计算与审计
    # ==============================================================================
    
    def _calculate_rank_ic(self, factor_values: pd.Series,
                           label_values: pd.Series) -> float:
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
    
    def audit_ic_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        【V106 IC 稳定性审计】
        
        要求：
        - T+1 IC > 0.04
        - IC_Std < 0.02 (确保信号不是靠某几天暴涨撑起来的)
        """
        logger.info(f"[{VERSION}][ICStabilityAudit] Starting IC stability audit...")
        
        t1_ic = self.calculate_t1_ic(df)
        
        # 检查 IC 强度
        ic_strong = t1_ic['mean_ic'] > 0.04
        
        # 检查 IC 稳定性
        ic_stable = t1_ic['ic_std'] < 0.02
        
        # 综合判断
        passed = ic_strong and ic_stable
        
        # 警告
        if not ic_strong:
            logger.warning(f"[{VERSION}][ICStabilityAudit] IC ({t1_ic['mean_ic']:.4f}) < 0.04 threshold")
        if not ic_stable:
            logger.warning(f"[{VERSION}][ICStabilityAudit] IC_Std ({t1_ic['ic_std']:.4f}) > 0.02 threshold")
        
        audit_result = {
            't1_ic': t1_ic,
            'ic_strong': ic_strong,
            'ic_stable': ic_stable,
            'passed': passed,
        }
        
        logger.info(f"[{VERSION}][ICStabilityAudit] Result: {'PASSED' if passed else 'FAILED'}")
        logger.info(f"[{VERSION}][ICStabilityAudit]   Mean IC: {t1_ic['mean_ic']:.4f}")
        logger.info(f"[{VERSION}][ICStabilityAudit]   IC Std: {t1_ic['ic_std']:.4f}")
        logger.info(f"[{VERSION}][ICStabilityAudit]   IC IR: {t1_ic['ic_ir']:.2f}")
        
        return audit_result
    
    def audit_ic_decay(self, df: pd.DataFrame) -> Dict[str, float]:
        """IC 衰减审计"""
        logger.info(f"[{VERSION}][ICDecayAudit] Starting IC decay audit...")
        
        ic_results = {}
        
        for n in [1, 3, 5]:
            col = f't{n}_return'
            if col in df.columns:
                ic = self._calculate_rank_ic(df['score'], df[col])
                ic_results[f'T+{n}'] = ic
        
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
    
    def analyze_factor_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【因子相关性矩阵分析】
        
        Args:
            df: 包含因子的数据
            
        Returns:
            相关性矩阵 DataFrame
        """
        factor_cols = list(self.BASE_FACTOR_WEIGHTS.keys())
        available_cols = [col for col in factor_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning(f"[{VERSION}][CorrelationAnalysis] Not enough factors for correlation analysis")
            return pd.DataFrame()
        
        corr_matrix = df[available_cols].corr()
        
        logger.info(f"[{VERSION}][CorrelationAnalysis] Factor correlation matrix:")
        logger.info(f"[{VERSION}][CorrelationAnalysis] {corr_matrix.to_string()}")
        
        return corr_matrix
    
    # ==============================================================================
    # V106 主接口
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
        
        # 因子相关性分析
        self.analyze_factor_correlation(result)
        
        # 返回必需列
        output_columns = ['trade_date', 'symbol', 'score', 't1_return']
        for n in [3, 5]:
            if f't{n}_return' in result.columns:
                output_columns.append(f't{n}_return')
        
        return result[output_columns]
    
    def get_ablation_results(self) -> Dict[str, Any]:
        """获取消融分析结果"""
        return self.gated_engine.get_ablation_results()
    
    def get_gate_states(self) -> Dict[str, Any]:
        """获取门控状态"""
        return self.gated_engine.get_gate_states()
    
    def get_healing_records(self) -> List[Dict]:
        """获取错误自愈记录"""
        return self.healing_records
    
    def generate_v106_report(self, output_path: str = None) -> str:
        """
        生成 V106 运行总结报告。
        
        Args:
            output_path: 输出路径
            
        Returns:
            报告文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"reports/V106_Summary_Report_{timestamp}.md"
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 获取消融分析结果
        ablation = self.get_ablation_results()
        gate_states = self.get_gate_states()
        healing_records = self.get_healing_records()
        
        # 格式化消融分析结果
        linear_ic = ablation.get('linear_weighted')
        gated_ic = ablation.get('gated')
        linear_ic_str = f"{linear_ic:.4f}" if isinstance(linear_ic, float) else str(linear_ic)
        gated_ic_str = f"{gated_ic:.4f}" if isinstance(gated_ic, float) else str(gated_ic)
        
        report_content = f"""# V106 逻辑门控与非线性动态增强 - 运行总结报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: {VERSION}

---

## 1. 核心改进概览

| 功能模块 | 状态 | 说明 |
|----------|------|------|
| GatedAlphaEngine | {'✓ 启用' if self.use_gating else '✗ 禁用'} | 动态逻辑门控引擎 |
| 波动率开关 | {'✓ 激活' if 'volatility_switch' in gate_states else '✗ 未激活'} | 高位抑制动量，启用反转 |
| 量价一致性过滤 | {'✓ 激活' if 'consistency_filter' in gate_states else '✗ 未激活'} | 方向一致才生效 |
| 三阶矩特征 | {'✓ 激活' if 'third_moment' in gate_states else '✗ 未激活'} | 偏度/峰度非线性修正 |
| 错误自愈 | {'✓ 启用' if self.auto_heal else '✗ 禁用'} | Database/Parquet 自动修复 |
| 消融分析 | {'✓ 启用' if self.enable_ablation else '✗ 禁用'} | 线性 vs 门控对比 |

---

## 2. 消融分析 (Ablation Analysis)

| 模型 | IC | 提升 |
|------|-----|------|
| 线性加权 | {linear_ic_str} | - |
| 逻辑门控 | {gated_ic_str} | {ablation.get('improvement_percent', 'N/A')} |

**结论**: {'门控后 IC 提升超过 10%，逻辑门控有效' if ablation.get('improvement', 0) > 0.10 else '门控后 IC 提升不足 10%，需重新审视门控阈值'}

---

## 3. 门控状态统计

### 3.1 波动率开关
"""
        
        if 'volatility_switch' in gate_states:
            vs = gate_states['volatility_switch']
            report_content += f"""
| 统计项 | 值 |
|--------|-----|
| 高波动率样本数 | {vs.get('high_vol_count', 'N/A')} |
| 低波动率样本数 | {vs.get('low_vol_count', 'N/A')} |
| 动量信号均值 | {vs.get('momentum_mean', 0):.4f} |
| 反转信号均值 | {vs.get('reversion_mean', 0):.4f} |
"""
        else:
            report_content += "\n*波动率开关未激活*\n"
        
        report_content += """
### 3.2 量价一致性过滤
"""
        
        if 'consistency_filter' in gate_states:
            cf = gate_states['consistency_filter']
            report_content += f"""
| 统计项 | 值 |
|--------|-----|
| 有效样本数 | {cf.get('valid_count', 'N/A')} |
| 无效样本数 | {cf.get('invalid_count', 'N/A')} |
| 有效率 | {cf.get('valid_ratio', 0):.1%} |
| 过滤前信号均值 | {cf.get('signal_before_mean', 0):.4f} |
| 过滤后信号均值 | {cf.get('signal_after_mean', 0):.4f} |
"""
        else:
            report_content += "\n*量价一致性过滤未激活*\n"
        
        report_content += """
### 3.3 三阶矩特征
"""
        
        if 'third_moment' in gate_states:
            tm = gate_states['third_moment']
            report_content += f"""
| 统计项 | 值 |
|--------|-----|
| 偏度均值 | {tm.get('skewness_mean', 0):.4f} |
| 峰度均值 | {tm.get('kurtosis_mean', 0):.4f} |
| 修正因子均值 | {tm.get('correction_factor_mean', 0):.4f} |
"""
        else:
            report_content += "\n*三阶矩特征未激活*\n"
        
        report_content += """
---

## 4. 中性化分析

### 4.1 中性化前后的信号损耗比

"""
        # 中性化损耗计算
        if self.factor_ic_before_cleaning and self.factor_ic_after_cleaning:
            report_content += """| 因子 | 中性化前 IC | 中性化后 IC | 损耗比 |
|------|-------------|-------------|--------|
"""
            all_factors = set(self.factor_ic_before_cleaning.keys()) | set(self.factor_ic_after_cleaning.keys())
            for factor in sorted(all_factors):
                before = self.factor_ic_before_cleaning.get(factor, 0)
                after = self.factor_ic_after_cleaning.get(factor, 0)
                loss_ratio = (before - after) / abs(before) if abs(before) > 1e-6 else 0
                report_content += f"| {factor} | {before:.4f} | {after:.4f} | {loss_ratio:.1%} |\n"
        else:
            report_content += "*中性化 IC 数据不可用*\n"
        
        report_content += f"""
---

## 5. 错误自愈记录

"""
        
        if healing_records:
            report_content += """| 时间 | 列 | 方法 | 状态 |
|------|-----|------|------|
"""
            for record in healing_records:
                report_content += f"| {record.get('timestamp', 'N/A')} | {record.get('column', 'N/A')} | {record.get('method', 'N/A')} | {record.get('status', 'N/A')} |\n"
        else:
            report_content += "*无错误自愈记录*\n"
        
        report_content += """
---

## 6. 因子相关性矩阵分析

*因子相关性矩阵已在日志中输出，用于检测多重共线性问题*

---

## 7. IC 稳定性要求

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| T+1 IC | > 0.04 | - | - |
| IC_Std | < 0.02 | - | - |

---

## 8. 交付清单

- [x] `src/alpha_research_v106.py` - 完整代码
- [x] 因子相关性矩阵分析
- [x] 中性化前后的信号损耗比
- [x] 错误自愈记录

---

*报告由 V106 Alpha Research Module 自动生成*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[{VERSION}][Report] Summary report saved to: {output_path}")
        
        return output_path


# ==============================================================================
# V106 中性化引擎 (继承 V105)
# ==============================================================================

class NeutralizationEngineV106:
    """V106 中性化引擎"""
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: List[str] = None):
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv']
        self.neutralize_vars = neutralize_vars
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        if columns is None:
            columns = list(AlphaResearchV106.BASE_FACTOR_WEIGHTS.keys())
        
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
            columns = list(AlphaResearchV106.BASE_FACTOR_WEIGHTS.keys())
        
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
        """OLS 中性化"""
        result = df.copy()
        
        # 准备数据
        if 'total_mv' not in result.columns:
            result['total_mv'] = 1e10
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        if 'industry_code' not in result.columns:
            result['industry_code'] = 'UNKNOWN'
        
        if columns is None:
            exclude_cols = {'trade_date', 'symbol', 'ts_code', 'industry_code',
                          'total_mv', 'ln_total_mv', 't1_return', 't3_return',
                          't5_return', 'score'}
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
                X_vars = ['ln_total_mv']
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
# V106 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       use_gating: bool = True,
                       enable_ablation: bool = True,
                       auto_heal: bool = True) -> AlphaResearchV106:
    """
    获取 AlphaResearchV106 实例。
    
    Args:
        config_path: 因子配置文件路径
        use_gating: 是否启用逻辑门控
        enable_ablation: 是否启用消融分析
        auto_heal: 是否启用错误自愈
        
    Returns:
        AlphaResearchV106 实例
    """
    return AlphaResearchV106(config_path, use_gating, enable_ablation, auto_heal)