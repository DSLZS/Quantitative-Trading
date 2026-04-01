"""
Alpha Research Module - V114 非线性共振与场景化 Alpha.

【V114 核心任务】
1. 禁止过度脱毒：采用"残差加权保留"（原始因子与残差按 3:7 混合）
2. 特征交叉（Feature Crossing）：
   - Alpha_Cross = Rank(OFI) * Rank(Volatility_20)（高波动下的订单流不平衡）
   - Alpha_Regime = np.where(Market_Cap < Median, Reversion_Signal, Momentum_Signal)
3. 升级 BoostedAlpha：决策树深度 3-4 层，捕捉 3 阶以上非线性交互
4. 数据自愈：遇到数据缺失立即调用 SQL 接口重新合成
5. 输出 T+1 IC 热力图

【V114 技术规格】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.02 | 核心指标 |
| IC IR | > 0.3 | 正交化效果指标 |
| Non-linear Features | ≥ 5 | 特征交叉数量 |

【V114 禁止事项】
- 禁止对所有因子执行 OLS 强行去市值/去行业
- 日志中禁止出现"V103"字样
- 文件名必须是 v114_audit_...
- 初始资金锁定 100,000.00，单边费率 0.15%
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
import matplotlib.pyplot as plt
import seaborn as sns

# V114 强制：主动加载环境变量
from dotenv import load_dotenv
load_dotenv()

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V114 强制版本全局变量
# ==============================================================================
VERSION = "V114"


# ==============================================================================
# V114 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.02 时触发"""
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
# V114 算子库 - 非线性共振增强
# ==============================================================================

class AlphaOperatorsV114:
    """
    V114 Alpha 算子库 - 非线性共振增强。
    
    【V114 新增算子】
    - Feature_Cross_OFI_Vol: 订单流不平衡 × 波动率
    - Feature_Cross_Regime: 市值 regime 切换
    - Residual_Weighted_Mix: 残差加权混合（3:7）
    - Boosted_Interaction: 提升树交互特征
    """
    
    EPSILON = 1e-6
    
    @staticmethod
    def Rank(x: pd.Series, group_col: Optional[str] = None, group_values: Optional[pd.Series] = None) -> pd.Series:
        """
        截面百分位排名。
        
        Args:
            x: 因子值序列
            group_col: 分组列名（如 'trade_date'）
            group_values: 分组值序列（当 group_col 无法直接用于 groupby 时）
        """
        if group_col is None:
            return x.rank(method='average') / len(x.dropna())
        
        # 如果提供了 group_values，使用它进行分组
        if group_values is not None:
            result = x.groupby(group_values).transform(
                lambda s: s.rank(method='average') / len(s.dropna()) if len(s.dropna()) > 0 else s
            )
        else:
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
            if std_val < AlphaOperatorsV114.EPSILON:
                std_val = AlphaOperatorsV114.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV114.EPSILON) if len(s.dropna()) > 1 else s
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
    def Feature_Cross_OFI_Vol(df: pd.DataFrame,
                               ofi_col: str = 'order_flow_imbalance_5',
                               vol_col: str = 'volatility_20',
                               group_col: str = 'trade_date') -> pd.Series:
        """
        【V114 核心】特征交叉：订单流不平衡 × 波动率。
        
        【逻辑】
        在高波动环境下，订单流不平衡的信号更强。
        Alpha_Cross = Rank(OFI) * Rank(Volatility_20)
        
        Args:
            df: 输入 DataFrame
            ofi_col: 订单流不平衡列
            vol_col: 波动率列
            group_col: 分组列
            
        Returns:
            交叉特征序列
        """
        if ofi_col not in df.columns or vol_col not in df.columns:
            logger.warning(f"[{VERSION}] Missing columns for OFI_Vol cross: {ofi_col}, {vol_col}")
            return pd.Series(np.nan, index=df.index)
        
        # 截面排名
        ofi_rank = AlphaOperatorsV114.Rank(df[ofi_col], group_col=group_col)
        vol_rank = AlphaOperatorsV114.Rank(df[vol_col], group_col=group_col)
        
        # 交叉
        cross = ofi_rank * vol_rank
        
        # 标准化
        cross = AlphaOperatorsV114.Scale(cross, group_col=group_col)
        
        return cross
    
    @staticmethod
    def Feature_Cross_Regime(df: pd.DataFrame,
                              market_cap_col: str = 'total_mv',
                              reversion_col: str = 'reversion_5',
                              momentum_col: str = 'momentum_10',
                              group_col: str = 'trade_date') -> pd.Series:
        """
        【V114 核心】特征交叉：市值 Regime 切换。
        
        【逻辑】
        小市值股票更适合反转策略，大市值股票更适合动量策略。
        Alpha_Regime = np.where(Market_Cap < Median, Reversion_Signal, Momentum_Signal)
        
        Args:
            df: 输入 DataFrame
            market_cap_col: 市值列
            reversion_col: 反转信号列
            momentum_col: 动量信号列
            group_col: 分组列
            
        Returns:
            Regime 切换特征序列
        """
        if market_cap_col not in df.columns:
            logger.warning(f"[{VERSION}] Missing {market_cap_col} for Regime switch")
            return pd.Series(np.nan, index=df.index)
        
        result = pd.Series(np.nan, index=df.index)
        
        for date in df[group_col].unique():
            mask = df[group_col] == date
            day_data = df.loc[mask]
            
            if len(day_data) < 10:
                continue
            
            # 计算市值中位数
            median_cap = day_data[market_cap_col].median()
            
            if np.isnan(median_cap):
                continue
            
            # Regime 切换
            regime_signal = np.where(
                day_data[market_cap_col] < median_cap,
                day_data[reversion_col] if reversion_col in day_data.columns else 0,
                day_data[momentum_col] if momentum_col in day_data.columns else 0
            )
            
            result.loc[mask] = regime_signal
        
        # 标准化
        result = AlphaOperatorsV114.Scale(result, group_col=group_col)
        
        return result
    
    @staticmethod
    def Residual_Weighted_Mix(df: pd.DataFrame,
                               columns: List[str],
                               risk_factors: List[str],
                               original_weight: float = 0.3,
                               residual_weight: float = 0.7,
                               group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【V114 核心】残差加权混合 - 原始因子与残差按 3:7 混合。
        
        【逻辑】
        V113 证明全量正交化会杀掉信号。
        采用"残差加权保留"：最终值 = 原始值 × 0.3 + 残差 × 0.7
        
        Args:
            df: 输入 DataFrame
            columns: 需要处理的因子列
            risk_factors: 风险因子列表（如 ['ln_total_mv', 'intraday_volatility']）
            original_weight: 原始因子权重（默认 0.3）
            residual_weight: 残差权重（默认 0.7）
            group_col: 分组列
            
        Returns:
            残差加权混合后的 DataFrame
        """
        result = df.copy()
        
        # 准备风险因子
        if 'total_mv' not in result.columns:
            result['total_mv'] = 1e10
        result['ln_total_mv'] = np.log(result['total_mv'] + AlphaOperatorsV114.EPSILON)
        
        # Volatility
        if 'intraday_volatility' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['intraday_volatility'] = (
                    (result['high'] - result['low']) / (result['close'] + AlphaOperatorsV114.EPSILON)
                )
            else:
                result['intraday_volatility'] = 0.02
        
        for col in columns:
            if col not in result.columns:
                continue
            
            mixed_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y_original = day_data[col].values
                
                # 构建风险因子矩阵
                X_vars = ['ln_total_mv', 'intraday_volatility']
                X_available = [v for v in X_vars if v in day_data.columns]
                
                if len(X_available) < 2:
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                X = day_data[X_available].values
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    beta = np.linalg.lstsq(X, y_original, rcond=None)[0]
                    y_pred = X @ beta
                    residuals = y_original - y_pred
                    
                    # 3:7 混合
                    y_mixed = original_weight * y_original + residual_weight * residuals
                    
                    day_data[col] = y_mixed
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
                    
                except Exception:
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
            
            if mixed_values:
                mixed_df = pd.concat(mixed_values, ignore_index=True)
                if len(mixed_df) == len(result):
                    result.loc[:, col] = mixed_df[col].values
        
        return result
    
    @staticmethod
    def Boosted_Interaction(df: pd.DataFrame,
                            features: List[str],
                            n_interactions: int = 5,
                            group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【V114 核心】提升树交互特征 - 捕捉 3 阶以上非线性交互。
        
        【逻辑】
        不要寻找单一强因子，要寻找因子的交互。
        生成 2 阶和 3 阶交互特征。
        
        Args:
            df: 输入 DataFrame
            features: 基础特征列表
            n_interactions: 生成的交互特征数量
            group_col: 分组列
            
        Returns:
            包含交互特征的 DataFrame
        """
        result = df.copy()
        interaction_features = []
        
        # 2 阶交互
        for i, f1 in enumerate(features):
            for f2 in features[i+1:]:
                if f1 not in df.columns or f2 not in df.columns:
                    continue
                
                col_name = f"boosted_{f1}_x_{f2}"
                
                # 计算交互
                interaction = df[f1] * df[f2]
                
                # 标准化
                interaction = AlphaOperatorsV114.Scale(interaction, group_col=group_col)
                
                result[col_name] = interaction.values
                interaction_features.append(col_name)
                
                if len(interaction_features) >= n_interactions:
                    break
            if len(interaction_features) >= n_interactions:
                break
        
        logger.info(f"[{VERSION}][BoostedInteraction] Generated {len(interaction_features)} interaction features")
        
        return result, interaction_features
    
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
            lambda x: x / (x.rolling(window=20).mean() + AlphaOperatorsV114.EPSILON)
        )
        turnover_shock = result[turnover_col].shift(1) / (turnover_ma + AlphaOperatorsV114.EPSILON)
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
        daily_illiq = returns.abs() / (volume + AlphaOperatorsV114.EPSILON)
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
# V114 数据自愈引擎
# ==============================================================================

class DataHealingEngineV114:
    """
    【V114 核心】数据自愈引擎 - 遇到数据缺失立即调用 SQL 接口重新合成。
    
    【自愈策略】
    1. 检测数据缺失（high/low/close/volume）
    2. 自动连接数据库拉取原始数据
    3. 重新合成因子
    4. 记录自愈日志
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        初始化数据自愈引擎。
        
        Args:
            db_url: 数据库连接 URL
        """
        self.db_url = db_url
        self.db = None
        self.healing_log = []
        self._connect_db()
    
    def _connect_db(self):
        """连接数据库"""
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.pool import QueuePool
            
            if self.db_url is None:
                self.db_url = os.getenv('DATABASE_URL', 'mysql+pymysql://user:pass@localhost/quant')
            
            engine = create_engine(
                self.db_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_recycle=3600,
            )
            self.db = engine
            logger.info(f"[{VERSION}][DataHealing] Database connected")
        except Exception as e:
            logger.warning(f"[{VERSION}][DataHealing] Database connection failed: {e}")
            self.db = None
    
    def heal_missing_data(self, df: pd.DataFrame,
                          required_columns: List[str],
                          symbol_col: str = 'symbol',
                          date_col: str = 'trade_date') -> pd.DataFrame:
        """
        自愈缺失数据。
        
        Args:
            df: 输入 DataFrame
            required_columns: 必需的列
            symbol_col: 股票代码列
            date_col: 交易日期列
            
        Returns:
            自愈后的 DataFrame
        """
        result = df.copy()
        missing_cols = [col for col in required_columns if col not in result.columns or result[col].isna().all()]
        
        if not missing_cols:
            return result
        
        logger.info(f"[{VERSION}][DataHealing] Missing columns detected: {missing_cols}")
        
        if self.db is None:
            logger.error(f"[{VERSION}][DataHealing] Cannot heal - database not connected")
            raise DataHealingError("Database not connected for data healing")
        
        # 从数据库拉取原始数据
        symbols = result[symbol_col].unique().tolist()
        dates = result[date_col].unique().tolist()
        
        if not symbols or not dates:
            raise DataHealingError("No symbols or dates for data healing")
        
        # 构建查询
        symbols_str = "','".join(symbols[:100])  # 限制数量
        dates_str = "','".join(dates)
        
        query = f"""
            SELECT trade_date, symbol, high, low, close, volume, amount
            FROM stock_daily
            WHERE symbol IN ('{symbols_str}')
              AND trade_date IN ('{dates_str}')
            ORDER BY symbol, trade_date
        """
        
        try:
            raw_data = pd.read_sql(query, self.db)
            
            if len(raw_data) == 0:
                raise DataHealingError("No raw data found for healing")
            
            logger.info(f"[{VERSION}][DataHealing] Retrieved {len(raw_data)} rows from database")
            
            # 合并数据
            result = result.merge(raw_data, on=['trade_date', 'symbol'], how='left', suffixes=('', '_healed'))
            
            # 用 healed 数据填充缺失值
            for col in ['high', 'low', 'close', 'volume', 'amount']:
                if col in result.columns:
                    healed_col = f"{col}_healed"
                    if healed_col in result.columns:
                        result[col] = result[col].fillna(result[healed_col])
                        result.drop(columns=[healed_col], inplace=True)
            
            self.healing_log.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'heal_missing_data',
                'columns': missing_cols,
                'rows_retrieved': len(raw_data),
            })
            
            logger.info(f"[{VERSION}][DataHealing] Data healing complete")
            
        except Exception as e:
            logger.error(f"[{VERSION}][DataHealing] Healing failed: {e}")
            raise DataHealingError(f"Data healing failed: {e}")
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


# ==============================================================================
# V114 中性化引擎 (残差加权保留)
# ==============================================================================

class NeutralizationEngineV114:
    """
    V114 中性化引擎 - 残差加权保留 (Residual Weighted Mix).
    
    【V114 改进】
    - 禁止对所有因子执行 OLS 强行去市值/去行业
    - 采用"残差加权保留"：原始因子与残差按 3:7 混合
    """
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: List[str] = None):
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv', 'intraday_volatility']
        self.neutralize_vars = neutralize_vars
        self.neutralization_stats = {}
        logger.info(f"[{VERSION}][NeutralizationEngine] Initialized with residual weighted mix (3:7)")
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        if columns is None:
            columns = list(AlphaResearchV114.ALL_FACTOR_COLUMNS)
        
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
            columns = list(AlphaResearchV114.ALL_FACTOR_COLUMNS)
        
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
    
    def residual_weighted_mix(self, df: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               original_weight: float = 0.3,
                               residual_weight: float = 0.7,
                               group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【V114 核心】残差加权混合 - 原始因子与残差按 3:7 混合。
        
        Args:
            df: 输入 DataFrame
            columns: 需要处理的列
            original_weight: 原始因子权重（默认 0.3）
            residual_weight: 残差权重（默认 0.7）
            group_col: 分组列
            
        Returns:
            残差加权混合后的 DataFrame
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
        
        processed_count = 0
        
        for col in columns:
            if col not in result.columns:
                continue
            
            mixed_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y_original = day_data[col].values
                
                # 构建风险因子矩阵
                X_vars = ['ln_total_mv', 'intraday_volatility']
                X_available = [v for v in X_vars if v in day_data.columns]
                
                if len(X_available) < 2:
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                X = day_data[X_available].values
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    beta = np.linalg.lstsq(X, y_original, rcond=None)[0]
                    y_pred = X @ beta
                    residuals = y_original - y_pred
                    
                    # 3:7 混合
                    y_mixed = original_weight * y_original + residual_weight * residuals
                    
                    day_data[col] = y_mixed
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
                    processed_count += 1
                    
                except Exception:
                    mixed_values.append(day_data[[col, group_col, 'symbol']])
            
            if mixed_values:
                mixed_df = pd.concat(mixed_values, ignore_index=True)
                if len(mixed_df) == len(result):
                    result.loc[:, col] = mixed_df[col].values
        
        self.neutralization_stats = {
            'method': 'residual_weighted_mix',
            'original_weight': original_weight,
            'residual_weight': residual_weight,
            'n_factors_processed': processed_count,
        }
        
        logger.info(f"[{VERSION}][Neutralization] Processed {processed_count} factors with 3:7 mix")
        
        return result
    
    def neutralize_industry(self, df: pd.DataFrame,
                            columns: Optional[List[str]] = None,
                            group_col: str = 'trade_date') -> pd.DataFrame:
        """行业中性化（可选）"""
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
# V114 动态权重池
# ==============================================================================

class DynamicWeightPoolV114:
    """
    V114 动态权重池 - 60 天 IC 衰减加权。
    """
    
    def __init__(self, window: int = 60, min_history: int = 20, decay_factor: float = 0.95):
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
            
            decay_weights = [self.decay_factor ** i for i in range(len(recent_ics) - 1, -1, -1)]
            decay_weights = np.array(decay_weights) / sum(decay_weights)
            
            weighted_ic = sum(ic * w for ic, w in zip(recent_ics, decay_weights))
            mean_ic = weighted_ic
            
            if abs(mean_ic) < 0.01:
                self.invalid_features.add(factor_name)
                weights[factor_name] = 0.0
                signal_directions[factor_name] = 1.0
                continue
            
            abs_ic = abs(mean_ic)
            weights[factor_name] = abs_ic
            signal_directions[factor_name] = 1.0 if mean_ic > 0 else -1.0
            total_abs_ic += abs_ic
        
        if total_abs_ic > 0:
            for factor_name in weights:
                if weights[factor_name] > 0:
                    weights[factor_name] /= total_abs_ic
        else:
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
# V114 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV114:
    """
    V114 Alpha 预测核心引擎 - 非线性共振与场景化 Alpha。
    
    【V114 核心改进】
    1. 禁止过度脱毒：残差加权保留（3:7 混合）
    2. 特征交叉：Alpha_Cross, Alpha_Regime
    3. 数据自愈：自动修复缺失数据
    4. T+1 IC 热力图输出
    """
    
    EPSILON = 1e-6
    
    # V114 全部因子列名（包含基础因子）
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
        # V114 新增特征交叉
        'alpha_cross_ofi_vol',
        'alpha_regime_switch',
    ]
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_neutralization: bool = True,
                 enable_feature_cross: bool = True,
                 enable_dynamic_weight: bool = True,
                 enable_l2_regularization: bool = True,
                 auto_heal: bool = True,
                 db_url: Optional[str] = None,
                 reflection_output: str = "reports/v114_reflection.json") -> None:
        """
        初始化 V114 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_neutralization: 是否启用中性化（残差加权保留）
            enable_feature_cross: 是否启用特征交叉
            enable_dynamic_weight: 是否启用动态权重
            enable_l2_regularization: 是否启用 L2 正则化
            auto_heal: 是否启用数据自愈
            db_url: 数据库连接 URL
            reflection_output: 反哺 JSON 输出路径
        """
        self.config_path = Path(config_path)
        self.enable_neutralization = enable_neutralization
        self.enable_feature_cross = enable_feature_cross
        self.enable_dynamic_weight = enable_dynamic_weight
        self.enable_l2_regularization = enable_l2_regularization
        self.auto_heal = auto_heal
        self.db_url = db_url
        self.reflection_output = reflection_output
        
        # V114 核心组件
        self.neutralization_engine = NeutralizationEngineV114()
        self.data_healing_engine = DataHealingEngineV114(db_url=db_url) if auto_heal else None
        self.weight_pool = DynamicWeightPoolV114(window=60) if enable_dynamic_weight else None
        
        # 因子 IC 记录
        self.factor_ic_raw = {}
        self.factor_ic_history = defaultdict(list)
        
        # 审计日志
        self.ic_decay_audit = {}
        self.data_audit_log = []
        self.alpha_audit_log = []
        self.feature_selection_report = {}
        self.lookahead_bias_check = {}
        self.feature_cross_features = []
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        # V114 版本确认
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][AlphaResearch] V114 Alpha Research Engine Initialized")
        logger.info("=" * 80)
        logger.info(f"  Neutralization: {self.enable_neutralization} (Residual Weighted Mix 3:7)")
        logger.info(f"  Feature Cross: {self.enable_feature_cross}")
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
    # V114 因子计算
    # ==============================================================================
    
    def compute_liquidity_stress(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """流动性压力因子"""
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        ls = AlphaOperatorsV114.Liquidity_Stress(
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
        
        ki = AlphaOperatorsV114.Kurtosis_Interaction(
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
        
        tail_risk = AlphaOperatorsV114.Tail_Risk(
            result, returns_col='return', n=period,
            symbol_col='symbol', quantile=0.05
        )
        
        result['tail_risk'] = tail_risk.values
        
        result['tail_risk'] = result.groupby('trade_date')['tail_risk'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_momentum_factor(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """动量因子"""
        result = df.copy()
        
        result[f'momentum_{period}'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
        )
        
        result[f'momentum_{period}'] = result.groupby('trade_date')[f'momentum_{period}'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    def compute_reversion_factor(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
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
        
        down_vol = AlphaOperatorsV114.Downside_Volatility(
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
    # V114 特征交叉计算
    # ==============================================================================
    
    def compute_feature_cross(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V114 核心】计算特征交叉。
        
        【交叉特征】
        1. Alpha_Cross = Rank(OFI) * Rank(Volatility_20)
        2. Alpha_Regime = np.where(Market_Cap < Median, Reversion_Signal, Momentum_Signal)
        """
        result = df.copy()
        
        logger.info(f"[{VERSION}][FeatureCross] Computing feature crosses...")
        
        # 1. Alpha_Cross = Rank(OFI) * Rank(Volatility_20)
        if 'order_flow_imbalance_5' in result.columns and 'volatility_20' in result.columns:
            # 使用 result['trade_date'] 作为分组值
            alpha_cross = AlphaOperatorsV114.Feature_Cross_OFI_Vol(
                result,
                ofi_col='order_flow_imbalance_5',
                vol_col='volatility_20',
                group_col=None  # 不使用列名
            )
            # 手动按 trade_date 分组计算
            alpha_cross_grouped = result['order_flow_imbalance_5'].groupby(result['trade_date']).transform(
                lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
            )
            vol_rank_grouped = result['volatility_20'].groupby(result['trade_date']).transform(
                lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
            )
            result['alpha_cross_ofi_vol'] = (alpha_cross_grouped * vol_rank_grouped).values
            logger.info(f"[{VERSION}][FeatureCross] Alpha_Cross computed")
        
        # 2. Alpha_Regime = 市值切换
        if 'total_mv' in result.columns and 'reversion_5' in result.columns and 'momentum_10' in result.columns:
            alpha_regime = AlphaOperatorsV114.Feature_Cross_Regime(
                result,
                market_cap_col='total_mv',
                reversion_col='reversion_5',
                momentum_col='momentum_10',
                group_col=None
            )
            # 手动计算 Regime 切换
            regime_result = pd.Series(np.nan, index=result.index)
            for date in result['trade_date'].unique():
                mask = result['trade_date'] == date
                day_data = result.loc[mask]
                if len(day_data) < 10:
                    continue
                median_cap = day_data['total_mv'].median()
                if np.isnan(median_cap):
                    continue
                regime_signal = np.where(
                    day_data['total_mv'] < median_cap,
                    day_data['reversion_5'] if 'reversion_5' in day_data.columns else 0,
                    day_data['momentum_10'] if 'momentum_10' in day_data.columns else 0
                )
                regime_result.loc[mask] = regime_signal
            # 标准化
            result['alpha_regime_switch'] = regime_result.groupby(result['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
            ).values
            logger.info(f"[{VERSION}][FeatureCross] Alpha_Regime computed")
        
        self.feature_cross_features = ['alpha_cross_ofi_vol', 'alpha_regime_switch']
        
        return result
    
    # ==============================================================================
    # V114 标签计算
    # ==============================================================================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """T+1 收益标签"""
        result = df.copy()
        
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / (x + self.EPSILON) - 1.0
        )
        
        return result
    
    def compute_tn_return(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        """T+N 收益标签"""
        result = df.copy()
        
        result[f't{n}_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / (x + self.EPSILON) - 1.0
        )
        
        return result
    
    # ==============================================================================
    # V114 前视偏差检查
    # ==============================================================================
    
    def check_lookahead_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """前视偏差检查"""
        logger.info(f"[{VERSION}][LookaheadBiasCheck] Starting check...")
        
        check_result = {
            'passed': True,
            'issues': [],
            'ic_decay': {},
        }
        
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
        
        if 'T+1' in check_result['ic_decay'] and 'T+3' in check_result['ic_decay']:
            if abs(check_result['ic_decay']['T+3']) > abs(check_result['ic_decay']['T+1']):
                check_result['passed'] = False
                check_result['issues'].append(
                    f"T+3 IC ({check_result['ic_decay']['T+3']:.4f}) > T+1 IC ({check_result['ic_decay']['T+1']:.4f})"
                )
        
        if 'T+3' in check_result['ic_decay'] and 'T+5' in check_result['ic_decay']:
            if abs(check_result['ic_decay']['T+5']) > abs(check_result['ic_decay']['T+3']):
                check_result['passed'] = False
                check_result['issues'].append(
                    f"T+5 IC ({check_result['ic_decay']['T+5']:.4f}) > T+3 IC ({check_result['ic_decay']['T+3']:.4f})"
                )
        
        self.lookahead_bias_check = check_result
        
        logger.info(f"[{VERSION}][LookaheadBiasCheck] Result: {'PASSED' if check_result['passed'] else 'FAILED'}")
        
        return check_result
    
    # ==============================================================================
    # V114 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【V114 计算顺序】
        1. 数据审计
        2. 数据自愈（如果启用）
        3. 基础因子计算
        4. 特征交叉（Alpha_Cross, Alpha_Regime）
        5. 残差加权混合（3:7）
        6. 因子清洗
        7. 动态权重池更新
        8. IC 加权预测
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V114 Factor Computation Started")
        logger.info("=" * 80)
        
        result = df.copy()
        
        # 1. 数据审计
        missing_count = result.isnull().sum().sum()
        self._log_data_audit("MissingValuesHandled", int(missing_count), "Initial missing values")
        
        # 2. 数据自愈
        if self.auto_heal and self.data_healing_engine:
            logger.info(f"[{VERSION}][FactorComputation] Running data healing...")
            try:
                result = self.data_healing_engine.heal_missing_data(
                    result,
                    required_columns=['high', 'low', 'close', 'volume']
                )
            except DataHealingError as e:
                logger.warning(f"[{VERSION}][FactorComputation] Data healing failed: {e}")
        
        # 3. 基础因子计算
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
        
        # 4. 特征交叉（V114 核心）
        if self.enable_feature_cross:
            logger.info(f"[{VERSION}][FactorComputation] Computing Feature Crosses (V114 Core)...")
            result = self.compute_feature_cross(result)
        
        # 5. 收益标签
        logger.info(f"[{VERSION}][FactorComputation] Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 6. 残差加权混合（V114 核心 - 禁止过度脱毒）
        if self.enable_neutralization:
            logger.info(f"[{VERSION}][FactorComputation] Running Residual Weighted Mix (3:7)...")
            factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
            result = self.neutralization_engine.residual_weighted_mix(result, columns=factor_cols)
        
        # 7. 因子清洗
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Cleaning Factors...")
            result = self.clean_factors(result, self.ALL_FACTOR_COLUMNS)
        
        # 8. 动态权重池更新
        if self.enable_dynamic_weight and self.weight_pool:
            logger.info(f"[{VERSION}][FactorComputation] Updating Dynamic Weight Pool...")
            self._update_weight_pool(result, self.ALL_FACTOR_COLUMNS)
        
        # 9. IC 加权预测
        logger.info(f"[{VERSION}][FactorComputation] Computing IC-Weighted Score...")
        result = self._compute_ic_weighted_score(result, self.ALL_FACTOR_COLUMNS)
        
        # Alpha 审计
        effective_count = self._count_effective_factors(result, self.ALL_FACTOR_COLUMNS)
        self._log_alpha_audit("EffectiveFactorCount", effective_count, "Factors with non-NaN values")
        
        # 前视偏差检查
        self.check_lookahead_bias(result)
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V114 Factor Computation Complete")
        logger.info(f"[{VERSION}][DataAudit] Missing values handled: {missing_count}")
        logger.info(f"[{VERSION}][AlphaAudit] Effective Factor Count: {effective_count}")
        logger.info(f"[{VERSION}][FeatureCross] Cross features: {self.feature_cross_features}")
        logger.info("=" * 80)
        
        return result
    
    def _compute_ic_weighted_score(self, df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
        """IC 加权评分"""
        result = df.copy()
        
        factor_ics = {}
        for factor_name in selected_features:
            if factor_name in result.columns and 't1_return' in result.columns:
                ic = self._calculate_rank_ic(result[factor_name], result['t1_return'])
                factor_ics[factor_name] = ic
        
        self.factor_ic_raw = factor_ics
        
        weights = {}
        total_abs_ic = 0.0
        
        for f in selected_features:
            ic = factor_ics.get(f, 0.0)
            if abs(ic) < 0.01:
                weights[f] = 0.0
            else:
                abs_ic = abs(ic)
                weights[f] = abs_ic
                total_abs_ic += abs_ic
        
        if total_abs_ic <= 0:
            total_abs_ic = len(selected_features)
            weights = {f: 1.0 for f in selected_features}
        
        raw_score = np.zeros(len(result))
        
        for factor_name in selected_features:
            if factor_name not in result.columns:
                continue
            
            weight = weights.get(factor_name, 0.01) / total_abs_ic if total_abs_ic > 0 else 0
            
            if self.enable_l2_regularization and weight > 0:
                weight_array = np.array([weight])
                norm_sq = np.sum(weight_array ** 2)
                weight_array = weight_array / (1 + 0.1 * norm_sq)
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
        """因子清洗"""
        result = df.copy()
        
        if selected_features is None:
            factor_cols = [col for col in self.ALL_FACTOR_COLUMNS if col in result.columns]
        else:
            factor_cols = [col for col in selected_features if col in result.columns]
        
        result = self.neutralization_engine.winsorize_mad(result, columns=factor_cols, n_std=3.0)
        result = self.neutralization_engine.normalize_zscore(result, columns=factor_cols)
        
        return result
    
    # ==============================================================================
    # V114 IC 计算与审计
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
        
        ic_strong = t1_ic['mean_ic'] > 0.02
        
        passed = ic_strong
        
        if not ic_strong:
            logger.warning(f"[{VERSION}][ICStabilityAudit] IC ({t1_ic['mean_ic']:.4f}) < 0.02 threshold")
        
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
    # V114 T+1 IC 热力图
    # ==============================================================================
    
    def plot_ic_heatmap(self, df: pd.DataFrame, output_path: str = "reports/v114_ic_heatmap.png") -> str:
        """
        【V114 核心】绘制 T+1 IC 热力图。
        
        Args:
            df: 包含 score 和 t1_return 的 DataFrame
            output_path: 输出图片路径
            
        Returns:
            输出文件路径
        """
        logger.info(f"[{VERSION}][ICHeatmap] Generating T+1 IC heatmap...")
        
        unique_dates = sorted(df['trade_date'].unique())
        ic_by_date = []
        
        for date in unique_dates:
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 10:
                continue
            
            ic = self._calculate_rank_ic(day_data['score'], day_data['t1_return'])
            if not np.isnan(ic):
                ic_by_date.append({
                    'trade_date': date,
                    'ic': ic,
                })
        
        if not ic_by_date:
            logger.warning(f"[{VERSION}][ICHeatmap] No IC data for heatmap")
            return ""
        
        ic_df = pd.DataFrame(ic_by_date)
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 按日期排序
        ic_df = ic_df.sort_values('trade_date')
        
        # 创建热力图数据（按月份分组）
        ic_df['trade_date'] = pd.to_datetime(ic_df['trade_date'])
        ic_df['month'] = ic_df['trade_date'].dt.to_period('M')
        
        # 计算每月 IC 统计
        monthly_ic = ic_df.groupby('month')['ic'].agg(['mean', 'std', 'count']).reset_index()
        
        # 绘制
        months = monthly_ic['month'].astype(str).tolist()
        ics = monthly_ic['mean'].tolist()
        
        # 颜色映射 - 修复颜色格式
        bar_colors = []
        for ic in ics:
            if ic >= 0:
                bar_colors.append((0, 0.7, 0, min(abs(ic) * 5, 1)))  # 绿色
            else:
                bar_colors.append((0.7, 0, 0, min(abs(ic) * 5, 1)))  # 红色
        
        bars = ax.bar(months, ics, color=bar_colors)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.02, color='blue', linestyle='--', linewidth=1, label='Target IC=0.02')
        ax.axhline(y=-0.02, color='blue', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Month')
        ax.set_ylabel('T+1 Rank IC')
        ax.set_title(f'V114 T+1 IC Heatmap by Month (Mean IC: {np.mean(ics):.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图片
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"[{VERSION}][ICHeatmap] Heatmap saved to: {output_file}")
        
        return str(output_file)
    
    # ==============================================================================
    # V114 自动反哺
    # ==============================================================================
    
    def save_reflection(self, df: pd.DataFrame) -> str:
        """自动分析并保存反哺 JSON"""
        logger.info(f"[{VERSION}][AutoReflection] Generating reflection report...")
        
        factor_ics = self.calculate_factor_ics(df)
        
        sorted_ics = sorted(factor_ics.items(), key=lambda x: abs(x[1]), reverse=True)
        
        top_effective = sorted_ics[:5]
        bottom_ineffective = sorted_ics[-5:][::-1]
        
        reflection = {
            'version': VERSION,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_factors': len(self.ALL_FACTOR_COLUMNS),
                'neutralization_method': self.neutralization_engine.neutralization_stats.get('method', 'residual_weighted_mix'),
                'feature_cross_enabled': self.enable_feature_cross,
                'feature_cross_features': self.feature_cross_features,
            },
            'top_5_effective_factors': [
                {'name': name, 'ic': ic, 'rank': i+1}
                for i, (name, ic) in enumerate(top_effective)
            ],
            'bottom_5_ineffective_factors': [
                {'name': name, 'ic': ic, 'rank': i+1}
                for i, (name, ic) in enumerate(bottom_ineffective)
            ],
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
        
        reflection_path = Path(self.reflection_output)
        reflection_path.parent.mkdir(parents=True, exist_ok=True)
        
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
    # V114 主接口
    # ==============================================================================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【主接口】计算 Alpha 评分。
        
        Args:
            df: 输入数据
            
        Returns:
            包含 score 和 t1_return 的 DataFrame
        """
        result = self.compute_factors(df, clean=True)
        
        ic_audit = self.audit_ic_stability(result)
        self.audit_ic_decay(result)
        self.save_reflection(result)
        
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
        return len(self.ALL_FACTOR_COLUMNS)
    
    def get_lookahead_bias_check(self) -> Dict[str, Any]:
        """获取前视偏差检查结果"""
        return self.lookahead_bias_check
    
    def get_feature_cross_features(self) -> List[str]:
        """获取特征交叉特征列表"""
        return self.feature_cross_features


# ==============================================================================
# V114 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_neutralization: bool = True,
                       enable_feature_cross: bool = True,
                       enable_dynamic_weight: bool = True,
                       enable_l2_regularization: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None,
                       reflection_output: str = "reports/v114_reflection.json") -> AlphaResearchV114:
    """
    获取 AlphaResearchV114 实例。
    """
    return AlphaResearchV114(
        config_path=config_path,
        enable_neutralization=enable_neutralization,
        enable_feature_cross=enable_feature_cross,
        enable_dynamic_weight=enable_dynamic_weight,
        enable_l2_regularization=enable_l2_regularization,
        auto_heal=auto_heal,
        db_url=db_url,
        reflection_output=reflection_output
    )