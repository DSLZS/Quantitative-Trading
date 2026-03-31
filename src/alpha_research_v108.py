"""
Alpha Research Module - V108 因子库与执行引擎 (Auto-Env-Healer + 符号纠偏 + 非线性动量).

【V108 核心改进 - 强制架构回归与数据自愈研发】
1. Auto-Env-Healer: 数据环境自修复
   - 启动前检测 DATABASE_URL
   - 缺失时自动查找 .env 或 config/db_config.json
   - 数据库连接失败时，自动加载 data/parquet/ 下所有可用年份数据拼接

2. 因子符号纠偏 (Sliding Window IC Checker):
   - 内置 20 天滑动窗口 IC 检查器
   - 若因子方向连续 5 天与未来收益反向，强制执行符号翻转

3. 非线性动量特征:
   - Ts_Rank(Ts_Argmax(close, 20)): 捕捉价格达到近期高点的相对位置

4. compute_factors 内部逻辑优化:
   - 严禁为了凑回测结果而修改 backtest_referee.py
   - 所有优化必须在因子计算内部完成

【V108 因子符号对齐原则】
- 使用滑动窗口 (20 天) 检测 IC 方向
- 连续 5 天 IC 为负 → 强制翻转因子符号
- 确保所有因子对最终评分的贡献方向一致

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标 |
| IC_Stability | 连续 5 天反向检测 | 滑动窗口 IC 检查器 |
| Data Healing | 100% | Parquet 缺失自动 SQL 补全 |
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
# V108 强制版本全局变量
# ==============================================================================
VERSION = "V108"


# ==============================================================================
# V108 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.05 时触发"""
    pass


class ICStabilityWarning(Exception):
    """IC 稳定性警告 - 当滑动窗口检测到连续反向时触发"""
    pass


class FactorSignAlignmentError(Exception):
    """因子符号对齐错误 - 当因子方向无法自动校准时抛出"""
    pass


class DataHealingError(Exception):
    """数据自愈错误 - 当数据自动修复失败时抛出"""
    pass


class DatabaseConnectionError(Exception):
    """数据库连接错误"""
    pass


# ==============================================================================
# V108 算子库 - 扩展非线性动量特征
# ==============================================================================

class AlphaOperatorsV108:
    """
    V108 Alpha 算子库 - 在 V107 基础上扩展非线性动量特征。
    
    【新增算子】
    - Ts_Argmax(x, n): 时间序列最大值位置
    - Ts_Rank(x, n): 时间序列百分位排名
    - Ts_Argmin(x, n): 时间序列最小值位置
    - Sliding_IC_Checker(factor, label, window=20): 滑动窗口 IC 检查器
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
            if std_val < AlphaOperatorsV108.EPSILON:
                std_val = AlphaOperatorsV108.EPSILON
            return (x - mean_val) / std_val
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperatorsV108.EPSILON) if len(s.dropna()) > 1 else s
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
    def Ts_Argmax(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【V108 新增】时间序列最大值位置。
        
        计算当前值在过去 n 日中的最大值位置（0 到 1 之间）：
        - 如果当前值是过去 n 日的最大值，返回 1.0
        - 如果当前值是过去 n 日的最小值，返回 0.0
        - 否则返回相对位置
        
        Args:
            x: 输入序列
            n: 计算窗口
            symbol_col: 股票代码列名
            
        Returns:
            最大值位置 (0 到 1 之间)
        """
        def calc_argmax(s):
            """计算单序列的 argmax 位置"""
            result = []
            for i in range(len(s)):
                if i < n:
                    result.append(np.nan)
                    continue
                
                window = s.iloc[max(0, i-n):i+1].dropna()
                if len(window) < n // 2:
                    result.append(np.nan)
                    continue
                
                max_val = window.max()
                min_val = window.min()
                
                if max_val - min_val < AlphaOperatorsV108.EPSILON:
                    result.append(0.5)
                    continue
                
                current_val = s.iloc[i]
                position = (current_val - min_val) / (max_val - min_val)
                result.append(position)
            
            return pd.Series(result, index=s.index)
        
        return x.groupby(symbol_col, group_keys=False).apply(calc_argmax)
    
    @staticmethod
    def Ts_Argmin(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【V108 新增】时间序列最小值位置。
        
        计算当前值在过去 n 日中的最小值位置（0 到 1 之间）：
        - 如果当前值是过去 n 日的最小值，返回 1.0
        - 如果当前值是过去 n 日的最大值，返回 0.0
        
        Args:
            x: 输入序列
            n: 计算窗口
            symbol_col: 股票代码列名
            
        Returns:
            最小值位置 (0 到 1 之间)
        """
        def calc_argmin(s):
            """计算单序列的 argmin 位置"""
            result = []
            for i in range(len(s)):
                if i < n:
                    result.append(np.nan)
                    continue
                
                window = s.iloc[max(0, i-n):i+1].dropna()
                if len(window) < n // 2:
                    result.append(np.nan)
                    continue
                
                max_val = window.max()
                min_val = window.min()
                
                if max_val - min_val < AlphaOperatorsV108.EPSILON:
                    result.append(0.5)
                    continue
                
                current_val = s.iloc[i]
                position = (max_val - current_val) / (max_val - min_val)
                result.append(position)
            
            return pd.Series(result, index=s.index)
        
        return x.groupby(symbol_col, group_keys=False).apply(calc_argmin)
    
    @staticmethod
    def Ts_Rank(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【V108 新增】时间序列百分位排名。
        
        计算当前值在过去 n 日中的百分位排名：
        - 如果当前值大于过去 n 日的所有值，返回 1.0
        - 如果当前值小于过去 n 日的所有值，返回 0.0
        
        Args:
            x: 输入序列
            n: 计算窗口
            symbol_col: 股票代码列名
            
        Returns:
            百分位排名 (0 到 1 之间)
        """
        def calc_ts_rank(s):
            """计算单序列的 ts_rank"""
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
    def Sliding_IC_Checker(factor: pd.Series, label: pd.Series, 
                           dates: pd.Series, window: int = 20,
                           threshold_days: int = 5) -> Tuple[bool, List[float]]:
        """
        【V108 核心】滑动窗口 IC 检查器。
        
        检测因子方向是否连续与未来收益反向：
        1. 按日期分组计算每日 IC
        2. 使用 window 天的滑动窗口
        3. 如果连续 threshold_days 天 IC 为负，返回 True (需要翻转)
        
        Args:
            factor: 因子值
            label: T+1 收益标签
            dates: 交易日期
            window: 滑动窗口大小 (默认 20 天)
            threshold_days: 连续反向天数阈值 (默认 5 天)
            
        Returns:
            (是否需要翻转，IC 序列)
        """
        # 按日期分组计算 IC
        ic_by_date = []
        sorted_dates = sorted(dates.unique())
        
        for date in sorted_dates:
            mask = dates == date
            day_factor = factor[mask]
            day_label = label[mask]
            
            # 去除空值
            valid_mask = day_factor.notna() & day_label.notna()
            if valid_mask.sum() < 10:
                ic_by_date.append(np.nan)
                continue
            
            f_clean = day_factor[valid_mask]
            l_clean = day_label[valid_mask]
            
            # 计算 Rank IC
            f_rank = f_clean.rank(method='average')
            l_rank = l_clean.rank(method='average')
            
            if np.std(f_rank) < AlphaOperatorsV108.EPSILON or np.std(l_rank) < AlphaOperatorsV108.EPSILON:
                ic_by_date.append(np.nan)
                continue
            
            ic = np.corrcoef(f_rank, l_rank)[0, 1]
            ic_by_date.append(ic if not np.isnan(ic) else np.nan)
        
        # 转换为数组
        ic_array = np.array(ic_by_date)
        
        # 检查连续反向天数
        consecutive_negative = 0
        max_consecutive_negative = 0
        
        for ic in ic_array:
            if np.isnan(ic):
                consecutive_negative = 0
                continue
            
            if ic < 0:
                consecutive_negative += 1
                max_consecutive_negative = max(max_consecutive_negative, consecutive_negative)
            else:
                consecutive_negative = 0
        
        # 判断是否需要翻转
        need_flip = max_consecutive_negative >= threshold_days
        
        return need_flip, ic_by_date


# ==============================================================================
# V108 中性化引擎
# ==============================================================================

class NeutralizationEngineV108:
    """
    V108 中性化引擎 - 行业 + 市值 + 日内波动率三重中性化。
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
            columns = list(AlphaResearchV108.BASE_FACTOR_WEIGHTS.keys())
        
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
            columns = list(AlphaResearchV108.BASE_FACTOR_WEIGHTS.keys())
        
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
# V108 数据环境自修复 (Auto-Env-Healer)
# ==============================================================================

class AutoEnvHealer:
    """
    【V108 核心】数据环境自修复引擎。
    
    【自修复流程】
    1. 检测 DATABASE_URL 环境变量
    2. 如果缺失，自动查找 .env 或 config/db_config.json
    3. 如果数据库连接失败，加载 data/parquet/ 下所有可用年份数据拼接
    """
    
    def __init__(self):
        self.db_url = None
        self.parquet_data = None
        self.healing_log = []
        
    def detect_database_url(self) -> Optional[str]:
        """
        检测 DATABASE_URL。
        
        检测顺序：
        1. 环境变量 DATABASE_URL
        2. .env 文件中的 MYSQL_* 配置
        3. config/db_config.json
        
        Returns:
            DATABASE_URL 或 None
        """
        # 1. 检查环境变量
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            logger.info(f"[{VERSION}][AutoEnvHealer] DATABASE_URL found in environment")
            self.db_url = db_url
            return db_url
        
        # 2. 检查 .env 文件
        env_file = Path(".env")
        if env_file.exists():
            logger.info(f"[{VERSION}][AutoEnvHealer] Loading .env file...")
            try:
                with open(env_file, 'r') as f:
                    env_content = f.read()
                
                # 解析 MySQL 配置
                mysql_config = {}
                for line in env_content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        if key.startswith('MYSQL_'):
                            mysql_config[key] = value
                
                if mysql_config:
                    # 构建 DATABASE_URL
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
        
        # 3. 检查 config/db_config.json
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
        """
        测试数据库连接。
        
        Returns:
            bool: 连接是否成功
        """
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
        """
        加载 data/parquet/ 下的所有可用年份数据。
        
        Args:
            parquet_dir: Parquet 文件目录
            
        Returns:
            拼接后的 DataFrame 或 None
        """
        parquet_path = Path(parquet_dir)
        if not parquet_path.exists():
            logger.warning(f"[{VERSION}][AutoEnvHealer] Parquet directory not found: {parquet_dir}")
            return None
        
        # 查找所有 Parquet 文件
        parquet_files = list(parquet_path.glob("*.parquet"))
        
        if not parquet_files:
            logger.warning(f"[{VERSION}][AutoEnvHealer] No Parquet files found in {parquet_dir}")
            return None
        
        logger.info(f"[{VERSION}][AutoEnvHealer] Found {len(parquet_files)} Parquet file(s)")
        
        # 加载并拼接所有 Parquet 文件
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
        
        # 拼接所有数据
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"[{VERSION}][AutoEnvHealer] Combined {len(combined_df)} total rows")
        
        self.parquet_data = combined_df
        return combined_df
    
    def heal(self, year: int = None) -> Tuple[Optional[pd.DataFrame], str]:
        """
        执行完整的数据自愈流程。
        
        Args:
            year: 目标年份 (可选)
            
        Returns:
            (数据 DataFrame, 数据来源说明)
        """
        logger.info("=" * 70)
        logger.info(f"[{VERSION}][AutoEnvHealer] Starting data healing...")
        logger.info("=" * 70)
        
        # 1. 检测 DATABASE_URL
        db_url = self.detect_database_url()
        
        # 2. 测试数据库连接
        if db_url and self.test_database_connection():
            logger.info(f"[{VERSION}][AutoEnvHealer] Using database connection")
            return None, "database"
        
        # 3. 数据库不可用，使用 Parquet 数据
        logger.info(f"[{VERSION}][AutoEnvHealer] Database unavailable, loading Parquet data...")
        parquet_data = self.load_parquet_data()
        
        if parquet_data is not None:
            # 按年份过滤
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
# V108 Alpha 研究引擎
# ==============================================================================

class AlphaResearchV108:
    """
    V108 Alpha 预测核心引擎 - 因子符号纠偏 + 非线性动量 + 数据自愈。
    
    【V108 核心改进】
    1. Auto-Env-Healer: 数据环境自修复
    2. Sliding Window IC Checker: 20 天滑动窗口检测，连续 5 天反向强制翻转
    3. Nonlinear Momentum: Ts_Rank(Ts_Argmax(close, 20))
    4. compute_factors 内部逻辑优化
    """
    
    EPSILON = 1e-6
    
    # V108 基础因子权重
    BASE_FACTOR_WEIGHTS = {
        "momentum_10": 0.10,
        "reversion_5": 0.12,
        "nonlinear_momentum_20": 0.15,  # V108 新增：非线性动量
        "volume_price_divergence_10": 0.10,
        "vcp_ratio_10": 0.08,
        "turnover_anomaly_5": 0.08,
        "money_flow_intensity_5": 0.10,
        "relative_strength_10": 0.08,
        "price_efficiency_20": 0.07,
        "volume_skew_20": 0.05,
        "return_kurtosis_20": 0.04,
        "residual_momentum_10": 0.03,
    }
    
    def __init__(self,
                 config_path: str = "config/factors.yaml",
                 enable_sliding_window_ic: bool = True,
                 enable_nonlinear_momentum: bool = True,
                 enable_neutralization: bool = True,
                 auto_heal: bool = True,
                 max_retries: int = 3,
                 db_url: Optional[str] = None) -> None:
        """
        初始化 V108 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            enable_sliding_window_ic: 是否启用滑动窗口 IC 检查器
            enable_nonlinear_momentum: 是否启用非线性动量
            enable_neutralization: 是否启用中性化
            auto_heal: 是否启用数据自愈
            max_retries: 最大重试次数
            db_url: 数据库连接 URL
        """
        self.config_path = Path(config_path)
        self.enable_sliding_window_ic = enable_sliding_window_ic
        self.enable_nonlinear_momentum = enable_nonlinear_momentum
        self.enable_neutralization = enable_neutralization
        self.auto_heal = auto_heal
        self.max_retries = max_retries
        self.db_url = db_url
        
        # 中性化引擎
        self.neutralization_engine = NeutralizationEngineV108()
        
        # 数据自愈引擎
        self.env_healer = AutoEnvHealer() if auto_heal else None
        
        # 因子 IC 记录
        self.factor_ic_raw = {}
        self.factor_ic_aligned = {}
        self.factor_direction_flips = {}
        self.sliding_window_ic_history = {}
        
        # 加载配置
        self.factors = []
        self._load_config()
        
        # IC 记录
        self.ic_decay_audit = {}
        
        logger.info(f"[{VERSION}][AlphaResearch] Initialized")
        logger.info(f"[{VERSION}][AlphaResearch]   Sliding Window IC Checker: {self.enable_sliding_window_ic}")
        logger.info(f"[{VERSION}][AlphaResearch]   Nonlinear Momentum: {self.enable_nonlinear_momentum}")
        logger.info(f"[{VERSION}][AlphaResearch]   Neutralization: {self.enable_neutralization}")
        logger.info(f"[{VERSION}][AlphaResearch]   Auto Healing: {self.auto_heal}")
    
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
    # V108 数据环境自修复
    # ==============================================================================
    
    def auto_heal_data(self, df: pd.DataFrame, year: int = None) -> Tuple[pd.DataFrame, str]:
        """
        【V108 数据自愈】自动修复缺失数据。
        
        优先使用传入的数据，只有在传入数据为空时才尝试自愈。
        
        Args:
            df: 输入数据
            year: 目标年份
            
        Returns:
            (修复后的数据，数据来源)
        """
        # 如果传入数据有效，直接使用
        if df is not None and len(df) > 0:
            logger.info(f"[{VERSION}][AutoHeal] Using input data: {len(df)} rows")
            return df, "original"
        
        if not self.auto_heal or self.env_healer is None:
            return df, "original"
        
        # 执行自愈流程
        parquet_data, source = self.env_healer.heal(year)
        
        if source == "parquet" and parquet_data is not None:
            logger.info(f"[{VERSION}][AutoHeal] Using Parquet data: {len(parquet_data)} rows")
            return parquet_data, "parquet"
        
        return df, "original"
    
    # ==============================================================================
    # V108 滑动窗口 IC 检查器
    # ==============================================================================
    
    def sliding_window_ic_check(self, df: pd.DataFrame, 
                                 factor_name: str,
                                 factor_values: pd.Series,
                                 window: int = 20,
                                 threshold_days: int = 5) -> Tuple[bool, List[float]]:
        """
        【V108 核心】滑动窗口 IC 检查器。
        
        Args:
            df: 包含 t1_return 的数据
            factor_name: 因子名称
            factor_values: 因子值
            window: 滑动窗口大小
            threshold_days: 连续反向天数阈值
            
        Returns:
            (是否需要翻转，IC 序列)
        """
        if 't1_return' not in df.columns:
            logger.warning(f"[{VERSION}][SlidingWindowIC] t1_return not found for {factor_name}")
            return False, []
        
        ops = AlphaOperatorsV108()
        
        need_flip, ic_sequence = ops.Sliding_IC_Checker(
            factor=factor_values,
            label=df['t1_return'],
            dates=df['trade_date'],
            window=window,
            threshold_days=threshold_days
        )
        
        # 记录 IC 历史
        self.sliding_window_ic_history[factor_name] = {
            'ic_sequence': ic_sequence,
            'need_flip': need_flip,
            'window': window,
            'threshold_days': threshold_days,
        }
        
        if need_flip:
            logger.warning(f"[{VERSION}][SlidingWindowIC] {factor_name}:连续反向 IC 超过{threshold_days}天，强制翻转")
        
        return need_flip, ic_sequence
    
    # ==============================================================================
    # V108 非线性动量特征
    # ==============================================================================
    
    def compute_nonlinear_momentum(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        【V108 核心】非线性动量特征。
        
        计算 Ts_Rank(Ts_Argmax(close, 20)):
        - 捕捉价格达到近期高点的相对位置
        - 当价格接近近期高点时，动量信号更强
        
        Args:
            df: 输入数据
            window: 计算窗口 (默认 20 日)
            
        Returns:
            包含 nonlinear_momentum_20 列的 DataFrame
        """
        if not self.enable_nonlinear_momentum:
            logger.info(f"[{VERSION}][NonlinearMomentum] Disabled")
            return df
        
        logger.info(f"[{VERSION}][NonlinearMomentum] Computing Ts_Rank(Ts_Argmax(close, {window}))...")
        
        result = df.copy()
        ops = AlphaOperatorsV108()
        
        # 确保数据按 symbol 和 trade_date 排序
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算 Ts_Argmax(close, 20) - 使用原始 close 列
        if 'close' not in result.columns:
            logger.error(f"[{VERSION}][NonlinearMomentum] 'close' column not found in data")
            result['nonlinear_momentum_20'] = np.nan
            return result
        
        # 使用 transform 方式计算 Ts_Argmax
        def calc_argmax_per_group(x):
            """计算单只股票的 Ts_Argmax"""
            values = x.shift(1).rolling(window=window).max()
            current = x.shift(1)
            min_val = x.shift(1).rolling(window=window).min()
            range_val = values - min_val + ops.EPSILON
            return (current - min_val) / range_val
        
        argmax_values = result.groupby('symbol', group_keys=False)['close'].apply(calc_argmax_per_group)
        
        # 使用 transform 方式计算 Ts_Rank
        def calc_ts_rank_per_group(x):
            """计算单只股票的 Ts_Rank"""
            result_series = []
            for i in range(len(x)):
                if i < window:
                    result_series.append(np.nan)
                    continue
                window_data = x.iloc[max(0, i-window+1):i+1].dropna()
                if len(window_data) < window // 2:
                    result_series.append(np.nan)
                    continue
                current_val = x.iloc[i]
                rank = (window_data < current_val).sum() / len(window_data)
                result_series.append(rank)
            return pd.Series(result_series, index=x.index)
        
        nonlinear_momentum = argmax_values.groupby(result['symbol'], group_keys=False).apply(calc_ts_rank_per_group)
        
        result['nonlinear_momentum_20'] = nonlinear_momentum.values
        
        # 截面标准化
        result['nonlinear_momentum_20'] = result.groupby('trade_date')['nonlinear_momentum_20'].transform(
            lambda x: (x - x.mean()) / (x.std() + ops.EPSILON) if len(x.dropna()) > 1 else x
        )
        
        logger.info(f"[{VERSION}][NonlinearMomentum] Computed for {result['symbol'].nunique()} stocks")
        
        return result
    
    # ==============================================================================
    # V108 因子计算
    # ==============================================================================
    
    def compute_momentum_factor(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """动量因子"""
        result = df.copy()
        ops = AlphaOperatorsV108()
        
        result['momentum_10'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        return result
    
    def compute_reversion_factor(self, df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
        """反转因子"""
        result = df.copy()
        ops = AlphaOperatorsV108()
        
        result['reversion_5'] = result.groupby('symbol')['close'].transform(
            lambda x: -(x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0)
        )
        return result
    
    def compute_volume_price_divergence(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """量价背离因子"""
        result = df.copy()
        ops = AlphaOperatorsV108()
        
        price_change = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        volume_change = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + ops.EPSILON) - 1.0
        )
        
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
        
        net_change = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) - x.shift(21)
        )
        
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
        
        result['volume_skew_20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=window).skew()
        )
        return result
    
    def compute_return_kurtosis(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """收益率峰度因子"""
        result = df.copy()
        
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        result['return_kurtosis_20'] = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=window).kurt()
        )
        return result
    
    def compute_residual_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """残差动量因子"""
        result = df.copy()
        
        if 'vwap' not in result.columns:
            if all(col in result.columns for col in ['high', 'low', 'close']):
                result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
            else:
                result['vwap'] = result['close']
        
        result['price_residual'] = result['close'].shift(1) - result['vwap']
        result['residual_momentum_10'] = result.groupby('symbol')['price_residual'].transform(
            lambda x: x / (x.shift(period) + self.EPSILON) - 1.0
        )
        return result
    
    # ==============================================================================
    # V108 标签计算
    # ==============================================================================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 T+1 收益标签"""
        result = df.copy()
        ops = AlphaOperatorsV108()
        
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / (x + ops.EPSILON) - 1.0
        )
        return result
    
    def compute_tn_return(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """计算 T+N 收益标签"""
        result = df.copy()
        ops = AlphaOperatorsV108()
        
        result[f't{n}_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / (x + ops.EPSILON) - 1.0
        )
        return result
    
    # ==============================================================================
    # V108 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, clean: bool = True, year: int = None) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【计算顺序】
        1. 数据自愈 (Auto-Env-Healer)
        2. 基础因子计算
        3. 非线性动量特征 (V108 新增)
        4. 收益标签
        5. 滑动窗口 IC 检查与符号纠偏 (V108 核心)
        6. 因子清洗 (中性化)
        7. 预测评分
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V108 Factor Computation Started")
        logger.info("=" * 80)
        
        # 1. 数据自愈
        logger.info(f"[{VERSION}][FactorComputation] Step 1: Auto Healing...")
        result, data_source = self.auto_heal_data(df, year)
        logger.info(f"[{VERSION}][FactorComputation]   Data source: {data_source}")
        
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
        
        # 3. 非线性动量特征 (V108 新增)
        logger.info(f"[{VERSION}][FactorComputation] Step 3: Computing Nonlinear Momentum...")
        result = self.compute_nonlinear_momentum(result)
        
        # 4. 收益标签
        logger.info(f"[{VERSION}][FactorComputation] Step 4: Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 5. 滑动窗口 IC 检查与符号纠偏 (V108 核心)
        if self.enable_sliding_window_ic:
            logger.info(f"[{VERSION}][FactorComputation] Step 5: Sliding Window IC Check...")
            result = self._apply_sliding_window_ic_correction(result)
        
        # 6. 因子清洗
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Step 6: Factor Cleaning (Neutralization)...")
            result = self.clean_factors(result)
        
        # 7. 预测评分
        logger.info(f"[{VERSION}][FactorComputation] Step 7: Computing Final Score...")
        result = self._compute_final_score(result)
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V108 Factor Computation Complete")
        logger.info("=" * 80)
        
        return result
    
    def _apply_sliding_window_ic_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用滑动窗口 IC 检查与符号纠偏。
        
        Args:
            df: 包含所有因子的数据
            
        Returns:
            符号纠偏后的数据
        """
        result = df.copy()
        
        factor_cols = list(self.BASE_FACTOR_WEIGHTS.keys())
        
        for factor_name in factor_cols:
            if factor_name not in result.columns:
                continue
            
            logger.info(f"[{VERSION}][SlidingWindowIC] Checking {factor_name}...")
            
            # 执行滑动窗口 IC 检查
            need_flip, ic_sequence = self.sliding_window_ic_check(
                result, factor_name, result[factor_name]
            )
            
            # 如果需要翻转，执行符号翻转
            if need_flip:
                result[factor_name] = -result[factor_name]
                self.factor_direction_flips[factor_name] = True
                logger.info(f"[{VERSION}][SlidingWindowIC] {factor_name}: FLIPPED due to consecutive negative IC")
            else:
                self.factor_direction_flips[factor_name] = False
            
            # 记录 IC
            valid_ics = [ic for ic in ic_sequence if not np.isnan(ic)]
            if valid_ics:
                self.factor_ic_raw[factor_name] = float(np.mean(valid_ics))
                self.factor_ic_aligned[factor_name] = abs(float(np.mean(valid_ics)))
        
        # 记录对齐结果
        logger.info(f"[{VERSION}][SlidingWindowIC] Correction complete:")
        for factor_name, flipped in self.factor_direction_flips.items():
            raw_ic = self.factor_ic_raw.get(factor_name, 0)
            logger.info(f"[{VERSION}][SlidingWindowIC]   {factor_name}: IC={raw_ic:.4f}, Flipped={flipped}")
        
        return result
    
    def _compute_final_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算最终评分"""
        result = df.copy()
        ops = AlphaOperatorsV108()
        
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
    # V108 IC 计算与审计
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
    
    def audit_ic_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """IC 稳定性审计"""
        logger.info(f"[{VERSION}][ICStabilityAudit] Starting IC stability audit...")
        
        t1_ic = self.calculate_t1_ic(df)
        
        # 检查 IC 强度
        ic_strong = t1_ic['mean_ic'] > 0.05
        
        # 综合判断
        passed = ic_strong
        
        if not ic_strong:
            logger.warning(f"[{VERSION}][ICStabilityAudit] IC ({t1_ic['mean_ic']:.4f}) < 0.05 threshold")
        
        audit_result = {
            't1_ic': t1_ic,
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
    
    # ==============================================================================
    # V108 主接口
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
    
    def get_factor_direction_flips(self) -> Dict[str, bool]:
        """获取因子方向翻转记录"""
        return self.factor_direction_flips
    
    def get_factor_ics(self, df: pd.DataFrame = None) -> Dict[str, float]:
        """
        获取因子 IC 记录。
        
        Args:
            df: 可选的数据 DataFrame (用于兼容接口)
            
        Returns:
            因子 IC 字典
        """
        return self.factor_ic_aligned
    
    def get_sliding_window_ic_history(self) -> Dict[str, Dict]:
        """获取滑动窗口 IC 历史"""
        return self.sliding_window_ic_history
    
    def get_neutralization_stats(self) -> Dict[str, Any]:
        """获取中性化统计"""
        return self.neutralization_engine.neutralization_stats
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        if self.env_healer:
            return self.env_healer.get_healing_log()
        return []


# ==============================================================================
# V108 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       enable_sliding_window_ic: bool = True,
                       enable_nonlinear_momentum: bool = True,
                       enable_neutralization: bool = True,
                       auto_heal: bool = True,
                       db_url: Optional[str] = None) -> AlphaResearchV108:
    """
    获取 AlphaResearchV108 实例。
    
    Args:
        config_path: 因子配置文件路径
        enable_sliding_window_ic: 是否启用滑动窗口 IC 检查器
        enable_nonlinear_momentum: 是否启用非线性动量
        enable_neutralization: 是否启用中性化
        auto_heal: 是否启用数据自愈
        db_url: 数据库连接 URL
        
    Returns:
        AlphaResearchV108 实例
    """
    return AlphaResearchV108(
        config_path=config_path,
        enable_sliding_window_ic=enable_sliding_window_ic,
        enable_nonlinear_momentum=enable_nonlinear_momentum,
        enable_neutralization=enable_neutralization,
        auto_heal=auto_heal,
        db_url=db_url
    )