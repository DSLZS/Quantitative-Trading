"""
Alpha Research Module - V109 核心 Alpha 突破 (订单流不平衡 + 波动率截面交互).

【V109 核心改进 - 深度逻辑重构，禁止符号修补】
1. 订单流不平衡 (Order Flow Imbalance, OFI):
   - 基于 amount/volume 与价格变化的非线性关系
   - 捕捉"聪明钱"的流入流出方向
   
2. 波动率截面交互 (Cross-Sectional Volatility Interaction):
   - 个股波动率相对截面波动率的位置
   - 低波动率股票在高波动率环境中表现更好
   
3. 乖离率动量修复 (Bias Momentum Repair):
   - 价格偏离 20 日均线后的均值回归动能
   - 捕捉"超跌反弹"和"超买回调"

4. 非线性特征构建:
   - Ts_Std(returns, 20) * Volume_Zscore: 放量突破后的缩量回调
   - 乖离率 * 动量：偏离均线后的回归动能

【V109 禁止事项 - 红线】
- 严禁使用任何自动翻转符号逻辑
- 严禁为了凑 IC 而修改因子方向
- 如果因子 IC 为负，必须修改其背后的数学逻辑

【V109 因子库 - 深度逻辑重构】
| 因子名称 | 数学逻辑 | 预期 IC 方向 |
|----------|----------|-------------|
| order_flow_imbalance_5 | (amount/volume) 与 price_change 的相关性 | + |
| volatility_interaction_20 | 个股 vol / 截面 vol 的倒数 | + |
| bias_momentum_repair_10 | (price - MA20)/MA20 * recent_return | + |
| volume_volatility_product | Ts_Std(return,20) * Volume_Zscore | + |
| smart_money_divergence | 大单净流入与价格背离 | + |
| liquidity_shock_5 | 换手率突变与价格反应 | + |
| accumulation_distribution_20 | 量价配合的累积分布 | + |
| relative_value_rank | 估值与动量的交互 | + |

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标 |
| Top Factor IC | > 0.04 | 核心因子独立战斗力 |
| IC Decay | Monotonic | 无前视偏差 |
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
# V109 强制版本全局变量
# ==============================================================================
VERSION = "V109"


# ==============================================================================
# V109 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.05 时触发"""
    pass


class FactorLogicError(Exception):
    """因子逻辑错误 - 当因子数学逻辑有问题时抛出"""
    pass


class DataHealingError(Exception):
    """数据自愈错误 - 当数据自动修复失败时抛出"""
    pass


# ==============================================================================
# V109 算子库 - 扩展订单流和波动率特征
# ==============================================================================

class AlphaOperatorsV109:
    """
    V109 Alpha 算子库 - 深度逻辑重构。
    
    【新增算子】
    - Order_Flow_Imbalance(x, n): 订单流不平衡
    - Volatility_Interaction(x, n): 波动率截面交互
    - Bias_Momentum_Repair(x, n): 乖离率动量修复
    - Volume_Volatility_Product(vol, ret, n): 波动率 - 成交量乘积
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
            if std_val < AlphaOperatorsV109.EPSILON:
                std_val = AlphaOperatorsV109.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV109.EPSILON) if len(s.dropna()) > 1 else s
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
    def Order_Flow_Imbalance(df: pd.DataFrame, n: int = 5,
                              amount_col: str = 'amount',
                              volume_col: str = 'volume',
                              close_col: str = 'close',
                              symbol_col: str = 'symbol') -> pd.Series:
        """
        【V109 核心】订单流不平衡 (Order Flow Imbalance).
        
        计算逻辑：
        1. 计算单位成交量的金额 (amount/volume) = 平均成交价格
        2. 计算价格变化方向
        3. OFI = 平均成交价格变化 * 成交量变化 的累积和
        
        经济含义：
        - 当大单买入时，平均成交价格上升，成交量放大 → OFI 为正
        - 当大单卖出时，平均成交价格下降，成交量放大 → OFI 为负
        
        Args:
            df: DataFrame 包含所有必需列
            n: 计算窗口
            amount_col: 成交金额列名
            volume_col: 成交量列名
            close_col: 收盘价列名
            symbol_col: 股票代码列名
            
        Returns:
            订单流不平衡指标
        """
        # 计算平均成交价格 (amount/volume)
        avg_price = df[amount_col] / (df[volume_col] + AlphaOperatorsV109.EPSILON)
        
        # 按 symbol 分组计算
        def calc_ofi(group):
            # 价格变化
            price_change = group[close_col].shift(1) - group[close_col].shift(2)
            # 成交量变化
            volume_change = group[volume_col].shift(1) / (group[volume_col].shift(2) + AlphaOperatorsV109.EPSILON) - 1
            # 订单流不平衡
            ofi_daily = price_change * volume_change
            # 累积 n 日
            ofi_cumsum = ofi_daily.rolling(window=n).sum()
            return ofi_cumsum
        
        ofi_cumsum = df.groupby(symbol_col, group_keys=False).apply(calc_ofi)
        
        return ofi_cumsum
    
    @staticmethod
    def Volatility_Interaction(returns: pd.Series, n: int = 20,
                                symbol_col: str = 'symbol',
                                date_col: str = 'trade_date') -> pd.Series:
        """
        【V109 核心】波动率截面交互 (Cross-Sectional Volatility Interaction).
        
        计算逻辑：
        1. 计算个股滚动波动率 Ts_Std(returns, n)
        2. 计算截面平均波动率 (每日所有股票波动率的均值)
        3. 计算相对波动率 = 个股波动率 / 截面波动率
        4. 取倒数：低相对波动率 → 高分数
        
        经济含义：
        - 低波动率股票在高波动率环境中更受青睐
        - "避险"效应：市场波动大时，资金流向低波动股票
        
        Args:
            returns: 收益率序列
            n: 计算窗口
            symbol_col: 股票代码列名
            date_col: 交易日期列名
            
        Returns:
            波动率交互指标 (低波因子)
        """
        # 计算个股滚动波动率
        stock_vol = returns.groupby(symbol_col).transform(
            lambda s: s.shift(1).rolling(window=n).std()
        )
        
        # 按日期分组计算截面平均波动率
        def cross_sectional_mean(s):
            """计算截面均值"""
            return s.transform('mean')
        
        # 需要将波动率与日期对齐
        vol_with_date = pd.DataFrame({
            'trade_date': returns.index.get_level_values(date_col) if date_col in returns.index.names else returns.index,
            'stock_vol': stock_vol.values
        }) if isinstance(returns.index, pd.MultiIndex) else pd.DataFrame({
            'trade_date': returns.index,
            'stock_vol': stock_vol.values
        })
        
        # 计算截面均值
        cross_mean_vol = vol_with_date.groupby('trade_date')['stock_vol'].transform('mean')
        
        # 计算相对波动率并取倒数
        relative_vol = stock_vol / (cross_mean_vol.values + AlphaOperatorsV109.EPSILON)
        inv_relative_vol = 1.0 / (relative_vol + AlphaOperatorsV109.EPSILON)
        
        return inv_relative_vol
    
    @staticmethod
    def Bias_Momentum_Repair(close: pd.Series, ma_window: int = 20,
                              return_window: int = 5,
                              symbol_col: str = 'symbol') -> pd.Series:
        """
        【V109 核心】乖离率动量修复 (Bias Momentum Repair).
        
        计算逻辑：
        1. 计算乖离率 = (close - MA20) / MA20
        2. 计算近期动量 = 过去 return_window 日的累计收益
        3. BMR = -乖离率 * 近期动量
        
        经济含义：
        - 当价格低于均线 (乖离率负) 且近期下跌 (动量负) → 超跌，预期反弹 → 正信号
        - 当价格高于均线 (乖离率正) 且近期上涨 (动量正) → 超买，预期回调 → 负信号
        
        Args:
            close: 收盘价
            ma_window: 均线窗口
            return_window: 动量窗口
            symbol_col: 股票代码列名
            
        Returns:
            乖离率动量修复指标
        """
        # 计算 MA20
        ma20 = close.groupby(symbol_col).transform(
            lambda s: s.shift(1).rolling(window=ma_window).mean()
        )
        
        # 计算乖离率
        bias = (close.shift(1) - ma20) / (ma20 + AlphaOperatorsV109.EPSILON)
        
        # 计算近期动量 (累计收益)
        momentum = close.groupby(symbol_col).transform(
            lambda s: s.shift(1) / s.shift(return_window + 1) - 1
        )
        
        # BMR = -乖离率 * 动量
        # 负号确保：超跌 + 负动量 → 正信号
        bmr = -bias * momentum
        
        return bmr
    
    @staticmethod
    def Volume_Volatility_Product(volume: pd.Series, returns: pd.Series,
                                   n: int = 20, symbol_col: str = 'symbol') -> pd.Series:
        """
        【V109 核心】波动率 - 成交量乘积特征.
        
        计算逻辑：
        1. 计算滚动波动率 Ts_Std(returns, n)
        2. 计算成交量 Z 分数
        3. 乘积 = 波动率 * 成交量 Z 分数
        
        经济含义：
        - 高波动 + 放量 → 可能是突破信号
        - 高波动 + 缩量 → 可能是回调信号
        
        Args:
            volume: 成交量
            returns: 收益率
            n: 计算窗口
            symbol_col: 股票代码列名
            
        Returns:
            波动率 - 成交量乘积指标
        """
        # 计算滚动波动率
        rolling_vol = returns.groupby(symbol_col).transform(
            lambda s: s.shift(1).rolling(window=n).std()
        )
        
        # 计算成交量 Z 分数
        volume_mean = volume.groupby(symbol_col).transform(
            lambda s: s.shift(1).rolling(window=n).mean()
        )
        volume_std = volume.groupby(symbol_col).transform(
            lambda s: s.shift(1).rolling(window=n).std()
        )
        volume_zscore = (volume.shift(1) - volume_mean) / (volume_std + AlphaOperatorsV109.EPSILON)
        
        # 乘积
        product = rolling_vol * volume_zscore
        
        return product


# ==============================================================================
# V109 中性化引擎
# ==============================================================================

class NeutralizationEngineV109:
    """
    V109 中性化引擎 - 行业 + 市值三重中性化。
    """
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: List[str] = None):
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv', 'intraday_volatility']
        self.neutralize_vars = neutralize_vars
        
        # 中性化记录
        self.neutralization_stats = {}
        
        logger.info(f"[{VERSION}][NeutralizationEngine] Initialized")
        logger.info(f"[{VERSION}][NeutralizationEngine]   Neutralization variables: {self.neutralize_vars}")
    
    def winsorize_mad(self, df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      n_std: float = 3.0,
                      group_col: str = 'trade_date') -> pd.DataFrame:
        """MAD 去极值"""
        if columns is None:
            columns = list(AlphaResearchV109.BASE_FACTOR_WEIGHTS.keys())
        
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
            columns = list(AlphaResearchV109.BASE_FACTOR_WEIGHTS.keys())
        
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
        
        # 计算日内波动率
        if 'intraday_volatility' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['intraday_volatility'] = (
                    (result['high'] - result['low']) / (result['close'] + self.EPSILON)
                )
            else:
                result['intraday_volatility'] = 0.02
        
        # 行业代码
        if 'industry_code' not in result.columns:
            result['industry_code'] = 'UNKNOWN'
        
        if columns is None:
            exclude_cols = {'trade_date', 'symbol', 'ts_code', 'industry_code',
                          'total_mv', 'ln_total_mv', 'intraday_volatility',
                          't1_return', 't3_return', 't5_return', 'score'}
            columns = [col for col in result.columns
                      if col not in exclude_cols and pd.api.types.is_numeric_dtype(result[col])]
        
        neutralization_impact = {}
        
        for col in columns:
            if col not in result.columns:
                continue
            
            neutralized_values = []
            original_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    original_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y = day_data[col].values
                X_vars = ['ln_total_mv', 'intraday_volatility']
                X = day_data[X_vars].values
                
                # 行业虚拟变量
                industry_dummies = pd.get_dummies(day_data['industry_code'], prefix='ind')
                if len(industry_dummies.columns) > 0:
                    X = np.column_stack([X, industry_dummies.values])
                
                # 添加截距项
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    original_values.append(day_data[[col, group_col, 'symbol']].copy())
                    
                    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                    y_pred = X @ beta
                    residuals = y - y_pred
                    
                    day_data[col] = residuals
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    
                except np.linalg.LinAlgError:
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    original_values.append(day_data[[col, group_col, 'symbol']])
            
            if neutralized_values and original_values:
                neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                original_df = pd.concat(original_values, ignore_index=True)
                
                if len(neutralized_df) == len(result):
                    corr_before_after = np.corrcoef(
                        neutralized_df[col].fillna(0),
                        original_df[col].fillna(0)
                    )[0, 1]
                    
                    neutralization_impact[col] = {
                        'corr_before_after': float(corr_before_after) if not np.isnan(corr_before_after) else 0.0,
                        'variance_reduction': 1.0 - (neutralized_df[col].var() / (original_df[col].var() + self.EPSILON))
                    }
                    
                    result.loc[:, col] = neutralized_df[col].values
        
        self.neutralization_stats = neutralization_impact
        
        logger.info(f"[{VERSION}][Neutralization] Completed triple neutralization")
        
        return result


# ==============================================================================
# V109 数据环境自修复 (Auto-Env-Healer)
# ==============================================================================

class AutoEnvHealer:
    """
    【V109 核心】数据环境自修复引擎。
    
    【自修复流程】
    1. 检测 DATABASE_URL 环境变量
    2. 如果缺失，自动查找 .env 或 config/db_config.json
    3. 如果数据库连接失败，加载 data/parquet/ 下所有可用年份数据拼接
    4. 数据列缺失时，尝试从 alternative source 拼接
    """
    
    def __init__(self):
        self.db_url = None
        self.parquet_data = None
        self.healing_log = []
        
    def detect_database_url(self) -> Optional[str]:
        """检测 DATABASE_URL"""
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            logger.info(f"[{VERSION}][AutoEnvHealer] DATABASE_URL found in environment")
            self.db_url = db_url
            return db_url
        
        env_file = Path(".env")
        if env_file.exists():
            logger.info(f"[{VERSION}][AutoEnvHealer] Loading .env file...")
            try:
                with open(env_file, 'r') as f:
                    env_content = f.read()
                
                mysql_config = {}
                for line in env_content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key.startswith('MYSQL_'):
                            mysql_config[key] = value
                
                if mysql_config:
                    host = mysql_config.get('MYSQL_HOST', 'localhost')
                    port = mysql_config.get('MYSQL_PORT', '3306')
                    user = mysql_config.get('MYSQL_USER', 'root')
                    password = mysql_config.get('MYSQL_PASSWORD', '')
                    database = mysql_config.get('MYSQL_DATABASE', '')
                    
                    db_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
                    logger.info(f"[{VERSION}][AutoEnvHealer] DATABASE_URL constructed from .env")
                    self.db_url = db_url
                    self.healing_log.append({
                        'action': 'load_from_env_file',
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    })
                    return db_url
                    
            except Exception as e:
                logger.warning(f"[{VERSION}][AutoEnvHealer] Failed to parse .env: {e}")
                self.healing_log.append({
                    'action': 'parse_env_file',
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        config_file = Path("config/db_config.json")
        if config_file.exists():
            logger.info(f"[{VERSION}][AutoEnvHealer] Loading config/db_config.json...")
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                db_url = config.get('DATABASE_URL')
                if db_url:
                    logger.info(f"[{VERSION}][AutoEnvHealer] DATABASE_URL found in config/db_config.json")
                    self.db_url = db_url
                    self.healing_log.append({
                        'action': 'load_from_config_json',
                        'status': 'success',
                        'timestamp': datetime.now().isoformat()
                    })
                    return db_url
                    
            except Exception as e:
                logger.warning(f"[{VERSION}][AutoEnvHealer] Failed to parse config/db_config.json: {e}")
                self.healing_log.append({
                    'action': 'parse_config_json',
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        logger.warning(f"[{VERSION}][AutoEnvHealer] DATABASE_URL not found, will use Parquet fallback")
        self.healing_log.append({
            'action': 'database_url_detection',
            'status': 'not_found',
            'timestamp': datetime.now().isoformat()
        })
        return None
    
    def test_database_connection(self) -> bool:
        """测试数据库连接"""
        if not self.db_url:
            return False
        
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(self.db_url)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info(f"[{VERSION}][AutoEnvHealer] Database connection successful")
            self.healing_log.append({
                'action': 'database_connection_test',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            logger.warning(f"[{VERSION}][AutoEnvHealer] Database connection failed: {e}")
            self.healing_log.append({
                'action': 'database_connection_test',
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def load_parquet_data(self, parquet_dir: str = "data/parquet") -> Optional[pd.DataFrame]:
        """加载 data/parquet/ 下的所有可用年份数据"""
        parquet_path = Path(parquet_dir)
        if not parquet_path.exists():
            logger.warning(f"[{VERSION}][AutoEnvHealer] Parquet directory not found: {parquet_dir}")
            return None
        
        parquet_files = list(parquet_path.glob("*.parquet"))
        
        if not parquet_files:
            logger.warning(f"[{VERSION}][AutoEnvHealer] No Parquet files found in {parquet_dir}")
            return None
        
        logger.info(f"[{VERSION}][AutoEnvHealer] Found {len(parquet_files)} Parquet file(s)")
        
        all_dfs = []
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                logger.info(f"[{VERSION}][AutoEnvHealer] Loaded {len(df)} rows from {pf.name}")
                all_dfs.append(df)
                self.healing_log.append({
                    'action': 'load_parquet',
                    'file': pf.name,
                    'rows': len(df),
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"[{VERSION}][AutoEnvHealer] Failed to load {pf.name}: {e}")
                self.healing_log.append({
                    'action': 'load_parquet',
                    'file': pf.name,
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        if not all_dfs:
            return None
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"[{VERSION}][AutoEnvHealer] Combined {len(combined_df)} total rows")
        
        self.parquet_data = combined_df
        return combined_df
    
    def heal_column(self, df: pd.DataFrame, column: str, 
                    alternative_sources: List[str] = None) -> pd.DataFrame:
        """
        【V109 增强】自愈缺失列。
        
        如果某列缺失，尝试从其他列构造：
        - total_mv 缺失：用 amount * 252 / turnover_rate 估算
        - industry_code 缺失：用 symbol 前缀推断
        - vwap 缺失：用 (high + low + close) / 3 估算
        
        Args:
            df: 输入数据
            column: 需要自愈的列
            alternative_sources: 可选的替代数据源列表
            
        Returns:
            修复后的 DataFrame
        """
        if column in df.columns:
            return df
        
        logger.info(f"[{VERSION}][AutoEnvHealer] Attempting to heal missing column: {column}")
        
        result = df.copy()
        
        # total_mv 自愈逻辑
        if column == 'total_mv' and 'amount' in df.columns and 'turnover_rate' in df.columns:
            result['total_mv'] = (
                result['amount'] * 252 / (result['turnover_rate'] + self.EPSILON)
            )
            logger.info(f"[{VERSION}][AutoEnvHealer] Healed total_mv from amount/turnover_rate")
            self.healing_log.append({
                'action': 'heal_column',
                'column': 'total_mv',
                'method': 'amount * 252 / turnover_rate',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return result
        
        # vwap 自愈逻辑
        if column == 'vwap' and all(col in df.columns for col in ['high', 'low', 'close']):
            result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
            logger.info(f"[{VERSION}][AutoEnvHealer] Healed vwap from (high+low+close)/3")
            self.healing_log.append({
                'action': 'heal_column',
                'column': 'vwap',
                'method': '(high + low + close) / 3',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return result
        
        # industry_code 自愈逻辑
        if column == 'industry_code' and 'symbol' in df.columns:
            result['industry_code'] = result['symbol'].apply(
                lambda x: x[:6] if isinstance(x, str) and len(x) >= 6 else 'UNKNOWN'
            )
            logger.info(f"[{VERSION}][AutoEnvHealer] Healed industry_code from symbol prefix")
            self.healing_log.append({
                'action': 'heal_column',
                'column': 'industry_code',
                'method': 'symbol prefix',
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
            return result
        
        logger.warning(f"[{VERSION}][AutoEnvHealer] Cannot heal column: {column}")
        self.healing_log.append({
            'action': 'heal_column',
            'column': column,
            'status': 'failed',
            'timestamp': datetime.now().isoformat()
        })
        return result
    
    def heal(self, year: int = None) -> Tuple[Optional[pd.DataFrame], str]:
        """执行完整的数据自愈流程"""
        logger.info("=" * 70)
        logger.info(f"[{VERSION}][AutoEnvHealer] Starting data healing...")
        logger.info("=" * 70)
        
        db_url = self.detect_database_url()
        
        if db_url and self.test_database_connection():
            logger.info(f"[{VERSION}][AutoEnvHealer] Using database connection")
            return None, "database"
        
        logger.info(f"[{VERSION}][AutoEnvHealer] Database unavailable, loading Parquet data...")
        parquet_data = self.load_parquet_data()
        
        if parquet_data is not None:
            if year is not None and 'trade_date' in parquet_data.columns:
                parquet_data['trade_date'] = pd.to_datetime(parquet_data['trade_date'])
                parquet_data = parquet_data[parquet_data['trade_date'].dt.year == year]
                logger.info(f"[{VERSION}][AutoEnvHealer] Filtered to year {year}: {len(parquet_data)} rows")
            
            return parquet_data, "parquet"
        
        logger.error(f"[{VERSION}][AutoEnvHealer] No data source available!")
        return None, "none"
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


# ==============================================================================
# V109 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV109:
    """
    V109 Alpha 预测核心引擎 - 深度逻辑重构，禁止符号修补。
    
    【V109 核心改进】
    1. Order Flow Imbalance: 订单流不平衡
    2. Volatility Interaction: 波动率截面交互
    3. Bias Momentum Repair: 乖离率动量修复
    4. Volume Volatility Product: 波动率 - 成交量乘积
    
    【禁止事项】
    - 严禁使用任何自动翻转符号逻辑
    - 如果因子 IC 为负，必须修改其数学逻辑
    """
    
    EPSILON = 1e-6
    
    # V109 基础因子权重 - 深度逻辑重构后的因子
    BASE_FACTOR_WEIGHTS = {
        "order_flow_imbalance_5": 0.15,      # V109 新增：订单流不平衡
        "volatility_interaction_20": 0.12,   # V109 新增：波动率截面交互
        "bias_momentum_repair_10": 0.12,     # V109 新增：乖离率动量修复
        "volume_volatility_product": 0.10,   # V109 新增：波动率 - 成交量乘积
        "smart_money_divergence": 0.10,      # V109 新增：聪明钱背离
        "liquidity_shock_5": 0.08,           # V109 新增：流动性冲击
        "accumulation_distribution_20": 0.08, # V109 新增：累积分布
        "relative_value_rank": 0.07,         # V109 新增：相对价值排名
        "momentum_10": 0.06,                 # 保留：动量因子
        "reversion_5": 0.06,                 # 保留：反转因子
        "volume_price_health": 0.06,         # 保留：量价健康度
    }
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_neutralization: bool = True,
                 auto_heal: bool = True,
                 max_retries: int = 3,
                 db_url: Optional[str] = None) -> None:
        """
        初始化 V109 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_neutralization: 是否启用中性化
            auto_heal: 是否启用数据自愈
            max_retries: 最大重试次数
            db_url: 数据库连接 URL
        """
        self.config_path = Path(config_path)
        self.enable_neutralization = enable_neutralization
        self.auto_heal = auto_heal
        self.max_retries = max_retries
        self.db_url = db_url
        
        # 中性化引擎
        self.neutralization_engine = NeutralizationEngineV109()
        
        # 数据自愈引擎
        self.env_healer = AutoEnvHealer() if auto_heal else None
        
        # 因子 IC 记录 (禁止符号翻转，只记录原始 IC)
        self.factor_ic_raw = {}
        self.factor_ablation_results = {}  # 因子消融实验结果
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        # IC 记录
        self.ic_decay_audit = {}
        
        logger.info(f"[{VERSION}][AlphaResearch] Initialized")
        logger.info(f"[{VERSION}][AlphaResearch]   Neutralization: {self.enable_neutralization}")
        logger.info(f"[{VERSION}][AlphaResearch]   Auto Healing: {self.auto_heal}")
        logger.info(f"[{VERSION}][AlphaResearch]   Factor Count: {len(self.BASE_FACTOR_WEIGHTS)}")
    
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
    
    # ==============================================================================
    # V109 数据环境自修复
    # ==============================================================================
    
    def auto_heal_data(self, df: pd.DataFrame, year: int = None) -> Tuple[pd.DataFrame, str]:
        """自动修复缺失数据"""
        if df is not None and len(df) > 0:
            logger.info(f"[{VERSION}][AutoHeal] Using input data: {len(df)} rows")
            return df, "original"
        
        if not self.auto_heal or self.env_healer is None:
            return df, "original"
        
        parquet_data, source = self.env_healer.heal(year)
        
        if source == "parquet" and parquet_data is not None:
            logger.info(f"[{VERSION}][AutoHeal] Using Parquet data: {len(parquet_data)} rows")
            return parquet_data, "parquet"
        
        return df, "original"
    
    # ==============================================================================
    # V109 核心因子计算 - 深度逻辑重构
    # ==============================================================================
    
    def compute_order_flow_imbalance(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """
        【V109 核心】订单流不平衡因子 (Order Flow Imbalance).
        
        数学逻辑：
        OFI = Σ(平均成交价格变化 * 成交量变化)
        
        经济含义：
        - 大单买入 → 平均成交价格上升 + 成交量放大 → OFI 为正
        - 大单卖出 → 平均成交价格下降 + 成交量放大 → OFI 为负
        
        预期 IC 方向：正 (OFI 高 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        # 确保数据按 symbol 和 trade_date 排序
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 检查必需列
        required_cols = ['amount', 'volume', 'close']
        missing_cols = [col for col in required_cols if col not in result.columns]
        if missing_cols:
            logger.error(f"[{VERSION}][OFI] Missing required columns: {missing_cols}")
            result['order_flow_imbalance_5'] = np.nan
            return result
        
        # 计算订单流不平衡 - 使用 DataFrame 接口
        def calc_ofi(group):
            # 价格变化
            price_change = group['close'].shift(1) - group['close'].shift(2)
            # 成交量变化
            volume_change = group['volume'].shift(1) / (group['volume'].shift(2) + ops.EPSILON) - 1
            # 订单流不平衡
            ofi_daily = price_change * volume_change
            # 累积 n 日
            ofi_cumsum = ofi_daily.rolling(window=period).sum()
            return ofi_cumsum
        
        ofi = result.groupby('symbol', group_keys=False).apply(calc_ofi)
        result['order_flow_imbalance_5'] = ofi.values
        
        # 截面标准化
        result['order_flow_imbalance_5'] = result.groupby('trade_date')['order_flow_imbalance_5'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][OFI] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    def compute_volatility_interaction(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        【V109 核心】波动率截面交互因子 (Volatility Interaction).
        
        数学逻辑：
        VI = 1 / (个股波动率 / 截面平均波动率)
        
        经济含义：
        - 低相对波动率股票 → 高 VI 分数
        - "避险"效应：市场波动大时，资金流向低波动股票
        
        预期 IC 方向：正 (VI 高 → 低波股票 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算收益率
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 计算个股波动率
        stock_vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        # 计算截面平均波动率
        cross_mean_vol = stock_vol.groupby(result['trade_date']).transform('mean')
        
        # 计算相对波动率并取倒数
        relative_vol = stock_vol / (cross_mean_vol + ops.EPSILON)
        vi = 1.0 / (relative_vol + ops.EPSILON)
        
        result['volatility_interaction_20'] = vi.values
        
        # 截面标准化
        result['volatility_interaction_20'] = result.groupby('trade_date')['volatility_interaction_20'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][VI] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    def compute_bias_momentum_repair(self, df: pd.DataFrame, ma_window: int = 20,
                                      return_window: int = 5) -> pd.DataFrame:
        """
        【V109 核心】乖离率动量修复因子 (Bias Momentum Repair).
        
        数学逻辑：
        BMR = -乖离率 * 近期动量
        乖离率 = (close - MA20) / MA20
        近期动量 = 过去 5 日累计收益
        
        经济含义：
        - 超跌 (乖离率负) + 负动量 → 预期反弹 → 正信号
        - 超买 (乖离率正) + 正动量 → 预期回调 → 负信号
        
        预期 IC 方向：正 (BMR 高 → 预期反弹 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算 MA20
        ma20 = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1).rolling(window=ma_window).mean()
        )
        
        # 计算乖离率
        bias = (result['close'].shift(1) - ma20) / (ma20 + ops.EPSILON)
        
        # 计算近期动量
        momentum = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / x.shift(return_window + 1) - 1
        )
        
        # BMR = -乖离率 * 动量
        bmr = -bias * momentum
        
        result['bias_momentum_repair_10'] = bmr.values
        
        # 截面标准化
        result['bias_momentum_repair_10'] = result.groupby('trade_date')['bias_momentum_repair_10'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][BMR] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    def compute_volume_volatility_product(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        【V109 核心】波动率 - 成交量乘积因子.
        
        数学逻辑：
        VVP = Ts_Std(returns, 20) * Volume_Zscore
        
        经济含义：
        - 高波动 + 放量 → 可能是突破信号
        - 高波动 + 缩量 → 可能是回调信号
        
        预期 IC 方向：正 (VVP 高 → 放量突破 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算收益率
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # 计算滚动波动率
        rolling_vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        # 计算成交量 Z 分数
        volume_mean = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=period).mean()
        )
        volume_std = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        volume_zscore = (result['volume'].shift(1) - volume_mean) / (volume_std + ops.EPSILON)
        
        # 乘积
        vvp = rolling_vol * volume_zscore
        
        result['volume_volatility_product'] = vvp.values
        
        # 截面标准化
        result['volume_volatility_product'] = result.groupby('trade_date')['volume_volatility_product'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][VVP] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    def compute_smart_money_divergence(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """
        【V109 新增】聪明钱背离因子.
        
        数学逻辑：
        1. 计算大单净流入 = (amount - volume * avg_price) 的累积
        2. 计算价格变化
        3. SMD = 大单净流入与价格变化的背离
        
        经济含义：
        - 大单流入但价格不涨 → 背离，预期补涨 → 正信号
        - 大单流出但价格不跌 → 背离，预期补跌 → 负信号
        
        预期 IC 方向：正 (SMD 高 → 背离程度大 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算平均价格
        avg_price = result['amount'] / (result['volume'] + ops.EPSILON)
        
        # 计算大单净流入 (简化：用 amount/volume 作为代理)
        big_order_flow = result['amount'] / (result['volume'] + ops.EPSILON)
        big_order_cumsum = big_order_flow.groupby(result['symbol']).transform(
            lambda x: x.rolling(window=period).sum()
        )
        
        # 计算价格变化
        price_change = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) - x.shift(period + 1)
        )
        
        # 标准化
        big_order_rank = big_order_cumsum.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        price_rank = price_change.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        # 背离 = 大单排名 - 价格排名
        smd = big_order_rank - price_rank
        
        result['smart_money_divergence'] = smd.values
        
        # 截面标准化
        result['smart_money_divergence'] = result.groupby('trade_date')['smart_money_divergence'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][SMD] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    def compute_liquidity_shock(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """
        【V109 新增】流动性冲击因子.
        
        数学逻辑：
        1. 计算换手率突变 = 当前换手率 / 过去均值
        2. 计算价格反应
        3. LS = 换手率突变 * 价格反应
        
        经济含义：
        - 换手率突增 + 价格上涨 → 资金抢筹 → 正信号
        - 换手率突增 + 价格下跌 → 资金出逃 → 负信号
        
        预期 IC 方向：正 (LS 高 → 资金抢筹 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 检查 turnover_rate 是否存在
        if 'turnover_rate' not in result.columns:
            # 用 volume 代理
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + ops.EPSILON)
            )
        
        # 计算换手率突变
        turnover_ma = result.groupby('symbol')['turnover_rate'].transform(
            lambda x: x.shift(1).rolling(window=20).mean()
        )
        turnover_shock = result['turnover_rate'].shift(1) / (turnover_ma + ops.EPSILON)
        
        # 计算价格反应
        price_return = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / x.shift(2) - 1
        )
        
        # 流动性冲击
        ls = turnover_shock * price_return
        
        result['liquidity_shock_5'] = ls.values
        
        # 截面标准化
        result['liquidity_shock_5'] = result.groupby('trade_date')['liquidity_shock_5'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][LS] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    def compute_accumulation_distribution(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        【V109 新增】累积分布因子 (Accumulation/Distribution).
        
        数学逻辑：
        1. 计算 CLV = (close - low) - (high - close) / (high - low)
        2. 计算 ADL = CLV * volume 的累积
        3. AD = ADL 的滚动和
        
        经济含义：
        - 收盘价接近高点 + 放量 → 累积 → 正信号
        - 收盘价接近低点 + 放量 → 派发 → 负信号
        
        预期 IC 方向：正 (AD 高 → 累积 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 检查必需列
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in result.columns:
                logger.error(f"[{VERSION}][AD] Missing required column: {col}")
                result['accumulation_distribution_20'] = np.nan
                return result
        
        # 计算 CLV (Close Location Value)
        high_low_range = result['high'] - result['low'] + ops.EPSILON
        clv = ((result['close'] - result['low']) - (result['high'] - result['close'])) / high_low_range
        
        # 计算 ADL
        adl = clv * result['volume']
        
        # 滚动累积
        ad = adl.groupby(result['symbol']).transform(
            lambda x: x.rolling(window=period).sum()
        )
        
        result['accumulation_distribution_20'] = ad.values
        
        # 截面标准化
        result['accumulation_distribution_20'] = result.groupby('trade_date')['accumulation_distribution_20'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][AD] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    def compute_relative_value_rank(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        【V109 新增】相对价值排名因子.
        
        数学逻辑：
        1. 计算估值代理 = 1 / 波动率 (低波 = 低估值)
        2. 计算动量排名
        3. RVR = 估值排名 * 动量排名
        
        经济含义：
        - 低估值 + 正动量 → 价值 + 成长 → 正信号
        - 高估值 + 负动量 → 双杀 → 负信号
        
        预期 IC 方向：正 (RVR 高 → 低估值 + 正动量 → 预期收益高)
        """
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算收益率和波动率
        result['return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        rolling_vol = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=period).std()
        )
        
        # 估值代理 (低波 = 低估值)
        value_proxy = 1.0 / (rolling_vol + ops.EPSILON)
        
        # 动量
        momentum = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / x.shift(period + 1) - 1
        )
        
        # 截面排名
        value_rank = value_proxy.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        momentum_rank = momentum.groupby(result['trade_date']).transform(
            lambda x: x.rank(method='average') / len(x.dropna()) if len(x.dropna()) > 0 else x
        )
        
        # 相对价值排名
        rvr = value_rank * momentum_rank
        
        result['relative_value_rank'] = rvr.values
        
        # 截面标准化
        result['relative_value_rank'] = result.groupby('trade_date')['relative_value_rank'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][RVR] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    # ==============================================================================
    # V109 基础因子计算 (保留)
    # ==============================================================================
    
    def compute_momentum_factor(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """动量因子"""
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result['momentum_10'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        return result
    
    def compute_reversion_factor(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """反转因子"""
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result['reversion_5'] = result.groupby('symbol')['close'].transform(
            lambda x: -(x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0)
        )
        return result
    
    def compute_volume_price_health(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """量价健康度因子"""
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        # 价格趋势
        price_trend = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / x.shift(period + 1) - 1
        )
        
        # 成交量趋势
        volume_trend = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1) / x.shift(period + 1) - 1
        )
        
        # 量价配合：价升量增 → 健康
        vph = price_trend * volume_trend
        
        result['volume_price_health'] = vph.values
        
        # 截面标准化
        result['volume_price_health'] = result.groupby('trade_date')['volume_price_health'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        return result
    
    # ==============================================================================
    # V109 标签计算
    # ==============================================================================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 T+1 收益标签"""
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / (x + ops.EPSILON) - 1.0
        )
        return result
    
    def compute_tn_return(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """计算 T+N 收益标签"""
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        result[f't{n}_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / (x + ops.EPSILON) - 1.0
        )
        return result
    
    # ==============================================================================
    # V109 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True, year: int = None) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【计算顺序】
        1. 数据自愈 (Auto-Env-Healer)
        2. V109 核心因子计算 (订单流、波动率交互、乖离率修复)
        3. 基础因子计算
        4. 收益标签
        5. 因子清洗 (中性化)
        6. 预测评分
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V109 Factor Computation Started")
        logger.info("=" * 80)
        
        # 1. 数据自愈
        logger.info(f"[{VERSION}][FactorComputation] Step 1: Auto healing...")
        result, data_source = self.auto_heal_data(df, year)
        logger.info(f"[{VERSION}][FactorComputation]   Data source: {data_source}")
        
        # 2. V109 核心因子计算
        logger.info(f"[{VERSION}][FactorComputation] Step 2: Computing V109 Core Factors...")
        result = self.compute_order_flow_imbalance(result, period=5)
        result = self.compute_volatility_interaction(result, period=20)
        result = self.compute_bias_momentum_repair(result, ma_window=20, return_window=5)
        result = self.compute_volume_volatility_product(result, period=20)
        result = self.compute_smart_money_divergence(result, period=10)
        result = self.compute_liquidity_shock(result, period=5)
        result = self.compute_accumulation_distribution(result, period=20)
        result = self.compute_relative_value_rank(result, period=20)
        
        # 3. 基础因子计算
        logger.info(f"[{VERSION}][FactorComputation] Step 3: Computing Base Factors...")
        result = self.compute_momentum_factor(result)
        result = self.compute_reversion_factor(result)
        result = self.compute_volume_price_health(result)
        
        # 4. 收益标签
        logger.info(f"[{VERSION}][FactorComputation] Step 4: Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 5. 因子清洗
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Step 5: Factor Cleaning (Neutralization)...")
            result = self.clean_factors(result)
        
        # 6. 预测评分
        logger.info(f"[{VERSION}][FactorComputation] Step 6: Computing Final Score...")
        result = self._compute_final_score(result)
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V109 Factor Computation Complete")
        logger.info("=" * 80)
        
        return result
    
    def _compute_final_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算最终评分"""
        result = df.copy()
        ops = AlphaOperatorsV109()
        
        # 线性加权
        raw_score = np.zeros(len(result))
        
        for factor_name, weight in self.BASE_FACTOR_WEIGHTS.items():
            if factor_name in result.columns:
                # 截面标准化
                factor_scaled = result[factor_name].fillna(0).groupby(result['trade_date']).transform(
                    lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
                )
                raw_score += factor_scaled.values * weight
        
        result['score'] = raw_score
        result['score_linear'] = raw_score
        
        logger.info(f"[{VERSION}][Score] Final score computed, mean={np.mean(raw_score):.4f}, std={np.std(raw_score):.4f}")
        
        return result
    
    def clean_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """因子清洗三部曲"""
        result = df.copy()
        
        factor_cols = list(self.BASE_FACTOR_WEIGHTS.keys())
        
        # 1. MAD 去极值
        result = self.neutralization_engine.winsorize_mad(result, columns=factor_cols, n_std=3.0)
        
        # 2. Z-Score 标准化
        result = self.neutralization_engine.normalize_zscore(result, columns=factor_cols)
        
        # 3. OLS 中性化
        if self.enable_neutralization:
            result = self.neutralization_engine.neutralize_ols(result, columns=factor_cols)
        
        return result
    
    # ==============================================================================
    # V109 IC 计算与审计
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
        """
        计算各因子的独立 IC (用于消融实验).
        
        Returns:
            因子 IC 字典
        """
        factor_ics = {}
        
        for factor_name in self.BASE_FACTOR_WEIGHTS.keys():
            if factor_name in df.columns and 't1_return' in df.columns:
                # 按日期分组计算 IC
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
    
    def run_ablation_experiment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        【V109 核心】因子消融实验。
        
        依次去掉每个因子，观察 IC 变化：
        - 如果去掉某因子后 IC 提升 → 该因子是负贡献
        - 如果去掉某因子后 IC 下降 → 该因子是正贡献
        
        Args:
            df: 包含所有因子的数据
            
        Returns:
            消融实验结果
        """
        logger.info(f"[{VERSION}][AblationExperiment] Starting factor ablation experiment...")
        
        # 计算完整因子库的 IC
        full_score_ic = self._calculate_rank_ic(df['score'], df['t1_return'])
        logger.info(f"[{VERSION}][AblationExperiment] Full model IC: {full_score_ic:.4f}")
        
        ablation_results = {}
        
        for factor_name in self.BASE_FACTOR_WEIGHTS.keys():
            if factor_name not in df.columns:
                continue
            
            # 去掉该因子后重新计算评分
            reduced_score = np.zeros(len(df))
            
            for other_factor, weight in self.BASE_FACTOR_WEIGHTS.items():
                if other_factor != factor_name and other_factor in df.columns:
                    factor_scaled = df[other_factor].fillna(0).groupby(df['trade_date']).transform(
                        lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x.dropna()) > 1 else x
                    )
                    reduced_score += factor_scaled.values * weight
            
            # 转换为 Series 以匹配 _calculate_rank_ic 的输入类型
            reduced_score_series = pd.Series(reduced_score, index=df.index)
            
            # 计算去掉该因子后的 IC
            reduced_ic = self._calculate_rank_ic(reduced_score_series, df['t1_return'])
            ic_change = reduced_ic - full_score_ic
            
            ablation_results[factor_name] = {
                'full_ic': full_score_ic,
                'reduced_ic': reduced_ic,
                'ic_change': ic_change,
                'contribution': 'negative' if ic_change > 0 else 'positive',
            }
            
            logger.info(f"[{VERSION}][AblationExperiment]   {factor_name}: IC change = {ic_change:+.4f} ({ablation_results[factor_name]['contribution']} contribution)")
        
        self.factor_ablation_results = ablation_results
        return ablation_results
    
    def audit_ic_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """IC 稳定性审计"""
        logger.info(f"[{VERSION}][ICStabilityAudit] Starting IC stability audit...")
        
        t1_ic = self.calculate_t1_ic(df)
        
        # 计算因子 IC
        factor_ics = self.calculate_factor_ics(df)
        
        # 运行消融实验
        ablation_results = self.run_ablation_experiment(df)
        
        # 检查 IC 强度
        ic_strong = t1_ic['mean_ic'] > 0.05
        
        # 综合判断
        passed = ic_strong
        
        if not ic_strong:
            logger.warning(f"[{VERSION}][ICStabilityAudit] IC ({t1_ic['mean_ic']:.4f}) < 0.05 threshold")
        
        audit_result = {
            't1_ic': t1_ic,
            'factor_ics': factor_ics,
            'ablation_results': ablation_results,
            'ic_strong': ic_strong,
            'passed': passed,
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
    # V109 主接口
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
    
    def get_ablation_results(self) -> Dict[str, Any]:
        """获取消融实验结果"""
        return self.factor_ablation_results
    
    def get_neutralization_stats(self) -> Dict[str, Any]:
        """获取中性化统计"""
        return self.neutralization_engine.neutralization_stats
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        if self.env_healer:
            return self.env_healer.get_healing_log()
        return []


# ==============================================================================
# V109 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_neutralization: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None) -> AlphaResearchV109:
    """
    获取 AlphaResearchV109 实例。
    
    Args:
        config_path: 因子配置文件路径
        enable_neutralization: 是否启用中性化
        auto_heal: 是否启用数据自愈
        db_url: 数据库连接 URL
        
    Returns:
        AlphaResearchV109 实例
    """
    return AlphaResearchV109(
        config_path=config_path,
        enable_neutralization=enable_neutralization,
        auto_heal=auto_heal,
        db_url=db_url
    )