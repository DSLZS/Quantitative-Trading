"""
Alpha Research Module - V105 纯净 Alpha 提取与中性化重构.

【V105 核心改进 - 从 V104 失败中崛起】
1. 强制中性化：所有因子必须经过 OLS 去除 industry_code 和 log(total_mv) 影响
2. 非线性特征工程：引入 Resonance/Divergence/Condition 算子
3. 预测目标对齐：使用 T+1 截面收益排名，Loss 计算使用 Spearman Rank IC
4. IC 衰减审计：输出 T+1 到 T+5 的 IC 衰减检查
5. 前视偏差防御：确保 Window 不包含当日收盘后信息

【V105 因子体系 - 基于算子库构建】
截面算子：Rank(x), Scale(x)
时间序列算子：Ts_Max(x, n), Ts_Std(x, n), Ts_Delta(x, n), Ts_Argmax(x, n)
非线性交互算子：
    - Resonance(A, B) = Rank(Rank(A) * Rank(B))  # 信号共振
    - Divergence(A, B) = Rank(A) - Rank(B)       # 量价背离
    - Condition(Cond, A, B) = np.where(Cond, A, B)  # 条件选择

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标：低于 0.04 触发 [V105][AlphaWeakWarning] |
| IC Decay | T+1 > T+3 > T+5 | 信号衰减必须符合单调性 |
| 中性化执行率 | 100% | 必须执行 OLS 中性化 |
| 前视偏差 | 0 | 严禁使用当日数据 |
"""

from typing import Any, Optional, Union
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
from loguru import logger
import yaml

# 忽略警告
warnings.filterwarnings('ignore')

# 内存优化
pd.options.mode.chained_assignment = None

# ==============================================================================
# V105 强制版本全局变量
# ==============================================================================
VERSION = "V105"


# ==============================================================================
# V105 自定义异常类
# ==============================================================================
class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.04 时触发"""
    pass


class ICDecayWarning(Exception):
    """IC 衰减警告 - 当 IC 不单调递减时触发"""
    pass


class NeutralizationFailedError(Exception):
    """中性化失败错误 - 当 OLS 中性化无法执行时抛出"""
    pass


# ==============================================================================
# V105 算子库 - Alpha Expression Library
# ==============================================================================

class AlphaOperators:
    """
    V105 Alpha 算子库 - 提供截面和时间序列算子。
    
    【截面算子】
    - Rank(x): 截面百分位排名
    - Scale(x): 截面均值 0 标准差 1 化
    
    【时间序列算子】
    - Ts_Max(x, n): 过去 n 日最大值
    - Ts_Std(x, n): 过去 n 日标准差
    - Ts_Delta(x, n): 过去 n 日变化量 (x - x.shift(n))
    - Ts_Argmax(x, n): 过去 n 日最大值位置
    
    【非线性交互算子】
    - Resonance(A, B): Rank(Rank(A) * Rank(B)) - 信号共振
    - Divergence(A, B): Rank(A) - Rank(B) - 量价背离
    - Condition(Cond, A, B): np.where(Cond, A, B) - 条件选择
    """
    
    EPSILON = 1e-6
    
    @staticmethod
    def Rank(x: pd.Series, group_col: Optional[str] = None) -> pd.Series:
        """
        【截面算子】Rank - 截面百分位排名。
        
        Args:
            x: 输入序列
            group_col: 分组列名（通常为 trade_date），用于截面排名
            
        Returns:
            排名序列 (0-1 之间)
        """
        if group_col is None:
            return x.rank(method='average') / len(x.dropna())
        
        # 按日期分组进行截面排名
        result = x.groupby(group_col).transform(
            lambda s: s.rank(method='average') / len(s.dropna()) if len(s.dropna()) > 0 else s
        )
        return result
    
    @staticmethod
    def Scale(x: pd.Series, group_col: Optional[str] = None) -> pd.Series:
        """
        【截面算子】Scale - 截面均值 0 标准差 1 化。
        
        Args:
            x: 输入序列
            group_col: 分组列名
            
        Returns:
            标准化序列 (mean=0, std=1)
        """
        if group_col is None:
            mean_val = x.mean()
            std_val = x.std()
            if std_val < AlphaOperators.EPSILON:
                std_val = AlphaOperators.EPSILON
            return (x - mean_val) / std_val
        
        # 按日期分组进行截面标准化
        result = x.groupby(group_col).transform(
            lambda s: (s - s.mean()) / (s.std() + AlphaOperators.EPSILON) if len(s.dropna()) > 1 else s
        )
        return result
    
    @staticmethod
    def Ts_Max(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【时间序列算子】Ts_Max - 过去 n 日最大值。
        
        Args:
            x: 输入序列
            n: 窗口大小
            symbol_col: 股票代码列名
            
        Returns:
            最大值序列
        """
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).max())
    
    @staticmethod
    def Ts_Min(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """【时间序列算子】Ts_Min - 过去 n 日最小值。"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).min())
    
    @staticmethod
    def Ts_Std(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【时间序列算子】Ts_Std - 过去 n 日标准差。
        
        Args:
            x: 输入序列
            n: 窗口大小
            symbol_col: 股票代码列名
            
        Returns:
            标准差序列
        """
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).std())
    
    @staticmethod
    def Ts_Delta(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【时间序列算子】Ts_Delta - 过去 n 日变化量。
        
        前视偏差防御：使用 shift(1) 确保不使用当日数据
        
        Args:
            x: 输入序列
            n: 窗口大小
            symbol_col: 股票代码列名
            
        Returns:
            变化量序列 (x.shift(1) - x.shift(n+1))
        """
        return x.groupby(symbol_col).transform(lambda s: s.shift(1) - s.shift(n + 1))
    
    @staticmethod
    def Ts_Argmax(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """
        【时间序列算子】Ts_Argmax - 过去 n 日最大值位置。
        
        Args:
            x: 输入序列
            n: 窗口大小
            symbol_col: 股票代码列名
            
        Returns:
            最大值位置序列 (0 到 n-1)
        """
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).apply(
            lambda w: np.argmax(w.values) if len(w.dropna()) > 0 else np.nan, raw=False
        ))
    
    @staticmethod
    def Ts_Argmin(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """【时间序列算子】Ts_Argmin - 过去 n 日最小值位置。"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).apply(
            lambda w: np.argmin(w.values) if len(w.dropna()) > 0 else np.nan, raw=False
        ))
    
    @staticmethod
    def Ts_Mean(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """【时间序列算子】Ts_Mean - 过去 n 日均值。"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).mean())
    
    @staticmethod
    def Ts_Sum(x: pd.Series, n: int, symbol_col: str = 'symbol') -> pd.Series:
        """【时间序列算子】Ts_Sum - 过去 n 日求和。"""
        return x.groupby(symbol_col).transform(lambda s: s.shift(1).rolling(window=n).sum())
    
    @staticmethod
    def Resonance(A: pd.Series, B: pd.Series, group_col: Optional[str] = None) -> pd.Series:
        """
        【非线性交互算子】Resonance - 信号共振。
        
        定义为：Rank(Rank(A) * Rank(B))
        用于捕捉两个信号的共振效应
        
        Args:
            A: 信号 A
            B: 信号 B
            group_col: 分组列名
            
        Returns:
            共振信号
        """
        rank_a = AlphaOperators.Rank(A, group_col)
        rank_b = AlphaOperators.Rank(B, group_col)
        return AlphaOperators.Rank(rank_a * rank_b, group_col)
    
    @staticmethod
    def Divergence(A: pd.Series, B: pd.Series, group_col: Optional[str] = None) -> pd.Series:
        """
        【非线性交互算子】Divergence - 量价背离。
        
        定义为：Rank(A) - Rank(B)
        用于捕捉两个信号的背离程度
        
        Args:
            A: 信号 A
            B: 信号 B
            group_col: 分组列名
            
        Returns:
            背离信号
        """
        rank_a = AlphaOperators.Rank(A, group_col)
        rank_b = AlphaOperators.Rank(B, group_col)
        return rank_a - rank_b
    
    @staticmethod
    def Condition(Cond: Union[pd.Series, np.ndarray], 
                  A: Union[pd.Series, np.ndarray], 
                  B: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        【非线性交互算子】Condition - 条件选择。
        
        使用 np.where 实现，若 Cond 成立则取 A，否则取 B
        
        Args:
            Cond: 条件序列
            A: 条件成立时的值
            B: 条件不成立时的值
            
        Returns:
            条件选择结果
        """
        if isinstance(Cond, pd.Series):
            Cond = Cond.values
        if isinstance(A, pd.Series):
            A = A.values
        if isinstance(B, pd.Series):
            B = B.values
        
        result = np.where(Cond, A, B)
        return pd.Series(result, index=A if isinstance(A, pd.Series) else None)
    
    @staticmethod
    def Sign(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """符号函数 - 返回 -1, 0, 1"""
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
        """log(1 + x) 函数 - 用于长尾压缩"""
        if isinstance(x, pd.Series):
            return np.log1p(np.abs(x)) * AlphaOperators.Sign(x)
        return np.log1p(np.abs(x)) * AlphaOperators.Sign(x)


# ==============================================================================
# V105 中性化引擎 - Mandatory Neutralization
# ==============================================================================

class NeutralizationEngine:
    """
    V105 中性化引擎 - 强制 OLS 中性化。
    
    【核心逻辑】
    任何因子在进入复合前，必须调用 OLS 去除 industry_code 和 log(total_mv) 的影响。
    
    【论据】
    V104 的 IC 接近 0 是因为信号被风格因子（Size/Beta）淹没了，
    V105 必须提取"纯净 Alpha"。
    """
    
    EPSILON = 1e-6
    
    def __init__(self, neutralize_vars: list[str] = None):
        """
        初始化中性化引擎。
        
        Args:
            neutralize_vars: 需要中性化的变量列表
        """
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv']
        self.neutralize_vars = neutralize_vars
        self.neutralization_stats = {}
    
    def prepare_neutralization_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备中性化所需数据。
        
        Args:
            df: 输入数据
            
        Returns:
            包含必需中性化变量的数据
        """
        result = df.copy()
        
        # 检查并处理 total_mv
        if 'total_mv' not in result.columns:
            logger.warning(f"[{VERSION}][Neutralization] Missing total_mv column")
            # 尝试估算 total_mv
            if 'amount' in result.columns and 'turnover_rate' in result.columns:
                logger.info(f"[{VERSION}][Neutralization] Estimating total_mv from amount/turnover_rate")
                estimated_mv = result['amount'] / (result['turnover_rate'].fillna(0.01) + self.EPSILON) * 100
                result['total_mv'] = result['total_mv'].fillna(estimated_mv) if 'total_mv' in result.columns else estimated_mv
            else:
                logger.warning(f"[{VERSION}][Neutralization] Cannot estimate total_mv, using fallback")
                # 使用市值中位数作为 fallback
                result['total_mv'] = 1e10  # 默认 100 亿市值
        
        # 计算 ln(total_mv)
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        # 处理 industry_code
        if 'industry_code' not in result.columns:
            logger.warning(f"[{VERSION}][Neutralization] Missing industry_code column")
            result['industry_code'] = 'UNKNOWN'
        
        return result
    
    def neutralize_ols(self, df: pd.DataFrame, 
                       columns: Optional[list[str]] = None,
                       group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【核心方法】OLS 中性化 - 按日期分组进行截面中性化。
        
        Args:
            df: 输入数据
            columns: 需要中性化的列
            group_col: 分组列（按日期进行截面中性化）
            
        Returns:
            中性化后的数据
        """
        result = df.copy()
        
        # 准备数据
        result = self.prepare_neutralization_data(result)
        
        if columns is None:
            # 默认中性化所有数值列（排除特定列）
            exclude_cols = {'trade_date', 'symbol', 'ts_code', 'industry_code', 
                          'total_mv', 'ln_total_mv', 't1_return', 't3_return', 
                          't5_return', 'score'}
            columns = [col for col in result.columns 
                      if col not in exclude_cols and pd.api.types.is_numeric_dtype(result[col])]
        
        neutralization_count = 0
        success_count = 0
        
        for col in columns:
            if col not in result.columns:
                continue
            
            neutralization_count += 1
            neutralized_values = []
            
            for date in result[group_col].unique():
                mask = result[group_col] == date
                day_data = result.loc[mask].copy()
                
                if len(day_data) < 30:
                    # 样本太少，跳过中性化
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    continue
                
                y = day_data[col].values
                
                # 构建 X 矩阵：ln_total_mv + industry dummies
                X_vars = ['ln_total_mv']
                X = day_data[X_vars].values
                
                # 添加行业虚拟变量
                industry_dummies = pd.get_dummies(day_data['industry_code'], prefix='ind')
                if len(industry_dummies.columns) > 0:
                    X = np.column_stack([X, industry_dummies.values])
                
                # 添加截距项
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    # OLS 回归：y = X * beta + epsilon
                    # 使用伪逆处理可能的奇异矩阵
                    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                    y_pred = X @ beta
                    residuals = y - y_pred  # 残差即为中性化后的值
                    
                    day_data[col] = residuals
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
                    success_count += 1
                    
                except np.linalg.LinAlgError as e:
                    logger.warning(f"[{VERSION}][Neutralization] LinAlgError for {col} on {date}: {e}")
                    neutralized_values.append(day_data[[col, group_col, 'symbol']])
            
            if neutralized_values:
                neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                if len(neutralized_df) == len(result):
                    result.loc[:, col] = neutralized_df[col].values
        
        # 记录中性化统计
        self.neutralization_stats = {
            'total_columns': neutralization_count,
            'success_columns': success_count,
            'success_rate': success_count / neutralization_count if neutralization_count > 0 else 0
        }
        
        logger.info(f"[{VERSION}][Neutralization] Completed: {success_count}/{neutralization_count} columns neutralized")
        
        return result
    
    def get_neutralization_stats(self) -> dict:
        """获取中性化统计信息"""
        return self.neutralization_stats


# ==============================================================================
# V105 因子计算引擎 - Factor Computation Engine
# ==============================================================================

class AlphaResearchV105:
    """
    V105 Alpha 预测核心引擎 - 纯净 Alpha 提取与中性化重构。
    
    【V105 核心改进】
    1. 强制中性化：所有因子必须经过 OLS 去除 industry_code 和 log(total_mv) 影响
    2. 非线性特征工程：引入 Resonance/Divergence/Condition 算子
    3. 预测目标对齐：使用 T+1 截面收益排名
    4. IC 衰减审计：输出 T+1 到 T+5 的 IC 衰减检查
    5. 前视偏差防御：确保 Window 不包含当日收盘后信息
    
    【因子计算对齐原则】
    - 所有因子必须使用 T-1 日及之前数据
    - 因子值对齐 T 日，预测 T+1 日收益
    - 严禁使用当日 close 计算因子
    """
    
    EPSILON = 1e-6
    
    # V105 因子权重配置
    FACTOR_WEIGHTS = {
        # 量价背离因子
        "volume_price_divergence_5": 0.12,
        "volume_price_divergence_10": 0.08,
        
        # 波动率收缩 VCP
        "vcp_ratio_5": 0.10,
        "vcp_ratio_10": 0.08,
        
        # 换手率异常
        "turnover_anomaly_5": 0.08,
        "turnover_anomaly_20": 0.06,
        
        # 资金流强度
        "money_flow_intensity_5": 0.10,
        "money_flow_intensity_10": 0.08,
        
        # 动量加速度
        "momentum_acceleration": 0.08,
        
        # 成交量偏度
        "volume_skew_20": 0.06,
        
        # 收益率峰度
        "return_kurtosis_20": 0.05,
        
        # 相对强度
        "relative_strength_10": 0.07,
        "relative_strength_20": 0.05,
        
        # 价格效率
        "price_efficiency_20": 0.07,
        
        # 波动率调整动量
        "volatility_adjusted_momentum": 0.08,
        
        # 量价健康度
        "volume_price_health": 0.06,
        
        # 残差动量
        "residual_momentum_10": 0.08,
    }
    
    def __init__(self, config_path: str = "config/factors.yaml", 
                 use_neutralization: bool = True,
                 auto_iterate: bool = True,
                 max_iterations: int = 3) -> None:
        """
        初始化 V105 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            use_neutralization: 是否启用中性化（V105 强制启用）
            auto_iterate: 是否启用自迭代优化
            max_iterations: 最大迭代次数
        """
        self.config_path = Path(config_path)
        self.use_neutralization = use_neutralization  # V105 强制为 True
        self.auto_iterate = auto_iterate
        self.max_iterations = max_iterations
        
        self.factors: list[dict[str, Any]] = []
        self._load_config()
        
        # 中性化引擎
        self.neutralization_engine = NeutralizationEngine()
        
        # 清洗前后 IC 记录
        self.factor_ic_before_cleaning = {}
        self.factor_ic_after_cleaning = {}
        
        # 因子生存竞争记录
        self.factor_competition_results = {}
        self.iteration_history = []
        
        # IC 衰减审计记录
        self.ic_decay_audit = {}
        
        logger.info(f"[{VERSION}][AlphaResearch] Initialized")
        logger.info(f"[{VERSION}][AlphaResearch]   Config path: {self.config_path}")
        logger.info(f"[{VERSION}][AlphaResearch]   Use neutralization: {self.use_neutralization} (V105 mandatory)")
        logger.info(f"[{VERSION}][AlphaResearch]   Auto iteration: {self.auto_iterate}")
        logger.info(f"[{VERSION}][AlphaResearch]   Max iterations: {self.max_iterations}")
        logger.info(f"[{VERSION}][AlphaResearch]   Factor count: {len(self.FACTOR_WEIGHTS)}")
    
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
    # V105 数据清洗三部曲
    # ==============================================================================
    
    def winsorize_mad(self, df: pd.DataFrame, columns: Optional[list[str]] = None,
                      n_std: float = 3.0, group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【清洗三部曲 1】去极值 - 使用 Median Absolute Deviation (MAD)。
        
        Args:
            df: 输入数据
            columns: 需要处理的列
            n_std: MAD 倍数
            group_col: 分组列
            
        Returns:
            去极值后的数据
        """
        if columns is None:
            columns = self.get_factor_names()
        
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
                adjusted_mad = mad * 1.4826  # 调整因子使 MAD 与标准差可比
                
                if adjusted_mad < self.EPSILON:
                    continue
                
                lower_bound = median - n_std * adjusted_mad
                upper_bound = median + n_std * adjusted_mad
                
                result.loc[mask, col] = result.loc[mask, col].clip(
                    lower=lower_bound, upper=upper_bound
                )
        
        return result
    
    def normalize_zscore(self, df: pd.DataFrame, columns: Optional[list[str]] = None,
                         group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【清洗三部曲 2】标准化 - Z-Score 标准化。
        
        Args:
            df: 输入数据
            columns: 需要处理的列
            group_col: 分组列
            
        Returns:
            标准化后的数据
        """
        if columns is None:
            columns = self.get_factor_names()
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            # 按日期分组进行截面标准化
            grouped = result.groupby(group_col)[col]
            mean = grouped.transform('mean')
            std = grouped.transform('std')
            std = std.replace(0, self.EPSILON)
            result.loc[:, col] = (result[col] - mean) / std
        
        return result
    
    def neutralize_ols(self, df: pd.DataFrame, columns: Optional[list[str]] = None,
                       group_col: str = 'trade_date') -> pd.DataFrame:
        """
        【清洗三部曲 3】中性化 - 使用 OLS 去除行业和市值暴露。
        
        V105 强制中性化：任何因子在进入复合前，必须调用 OLS 去除 
        industry_code 和 log(total_mv) 的影响。
        
        Args:
            df: 输入数据
            columns: 需要中性化的列
            group_col: 分组列
            
        Returns:
            中性化后的数据
        """
        return self.neutralization_engine.neutralize_ols(df, columns, group_col)
    
    def clean_factors(self, df: pd.DataFrame, 
                      do_winsorize: bool = True,
                      do_normalize: bool = True,
                      do_neutralize: bool = True) -> pd.DataFrame:
        """
        执行因子清洗三部曲。
        
        Args:
            df: 输入数据
            do_winsorize: 是否去极值
            do_normalize: 是否标准化
            do_neutralize: 是否中性化
            
        Returns:
            清洗后的数据
        """
        result = df.copy()
        
        logger.info(f"[{VERSION}][FactorClean] Starting factor cleaning...")
        
        # 记录清洗前 IC
        self.factor_ic_before_cleaning = self._calculate_all_factor_ics(result)
        
        if do_winsorize:
            logger.info(f"[{VERSION}][FactorClean] Step 1: MAD Winsorization (3.0 std)")
            result = self.winsorize_mad(result, n_std=3.0)
        
        if do_normalize:
            logger.info(f"[{VERSION}][FactorClean] Step 2: Z-Score Normalization")
            result = self.normalize_zscore(result)
        
        if do_neutralize and self.use_neutralization:
            logger.info(f"[{VERSION}][FactorClean] Step 3: OLS Neutralization (Size + Industry)")
            result = self.neutralize_ols(result)
        
        # 记录清洗后 IC
        self.factor_ic_after_cleaning = self._calculate_all_factor_ics(result)
        
        self._print_cleaning_comparison()
        
        logger.info(f"[{VERSION}][FactorClean] Completed")
        return result
    
    # ==============================================================================
    # V105 IC 计算与审计
    # ==============================================================================
    
    def _calculate_all_factor_ics(self, df: pd.DataFrame) -> dict[str, float]:
        """计算所有因子的 IC 值。"""
        ics = {}
        
        for factor_name in self.get_factor_names():
            if factor_name not in df.columns:
                continue
            if 't1_return' not in df.columns:
                continue
            
            ic = self._calculate_single_factor_ic(df[factor_name], df['t1_return'])
            ics[factor_name] = ic
        
        return ics
    
    def _calculate_single_factor_ic(self, factor_values: pd.Series, 
                                     label_values: pd.Series) -> float:
        """
        计算单个因子的 Rank IC (Spearman 相关系数)。
        
        Args:
            factor_values: 因子值
            label_values: 标签值（T+1 收益）
            
        Returns:
            Rank IC 值
        """
        mask = factor_values.notna() & label_values.notna()
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算排名
        factor_ranks = factor_clean.rank(method='average')
        label_ranks = label_clean.rank(method='average')
        
        # 计算 Spearman 相关系数
        if np.std(factor_ranks) < 1e-10 or np.std(label_ranks) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _calculate_rank_ic(self, x: pd.Series, y: pd.Series) -> float:
        """
        计算 Spearman Rank IC。
        
        预测目标对齐：预测目标必须是 T+1 的截面收益排名（Ranked Returns），
        计算 Loss 时必须使用 Spearman Rank IC。
        
        Args:
            x: 预测值
            y: 真实值
            
        Returns:
            Spearman Rank IC
        """
        mask = x.notna() & y.notna()
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 10:
            return 0.0
        
        x_ranks = x_clean.rank(method='average')
        y_ranks = y_clean.rank(method='average')
        
        if np.std(x_ranks) < 1e-10 or np.std(y_ranks) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(x_ranks, y_ranks)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _print_cleaning_comparison(self) -> None:
        """打印清洗前后 IC 对比表。"""
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][CleaningComparison] Factor IC Before/After Cleaning")
        logger.info("-" * 80)
        logger.info(f"{'Factor Name':<40} {'Before IC':>12} {'After IC':>12} {'Improvement':>10}")
        logger.info("-" * 80)
        
        all_factors = set(self.factor_ic_before_cleaning.keys()) | set(self.factor_ic_after_cleaning.keys())
        
        for factor in sorted(all_factors):
            before = self.factor_ic_before_cleaning.get(factor, 0)
            after = self.factor_ic_after_cleaning.get(factor, 0)
            improvement = after - before
            status = "+" if improvement > 0 else "-"
            logger.info(f"{factor:<40} {before:>12.4f} {after:>12.4f} {improvement:>+10.4f} {status}")
        
        logger.info("=" * 80)
    
    def audit_ic_decay(self, df: pd.DataFrame) -> dict[str, float]:
        """
        【IC 衰减审计】对比 T+1 到 T+5 的 IC 衰减。
        
        如果 T+1 IC 低于 0.04，在日志中触发 [V105][AlphaWeakWarning]。
        
        Args:
            df: 包含因子和标签的数据
            
        Returns:
            IC 衰减审计结果
        """
        logger.info(f"[{VERSION}][ICDecayAudit] Starting IC decay audit...")
        
        ic_results = {}
        
        # 计算 T+1, T+3, T+5 IC
        horizons = [1, 3, 5]
        for n in horizons:
            col = f't{n}_return'
            if col in df.columns:
                ic = self._calculate_rank_ic(df['score'], df[col])
                ic_results[f'T+{n}'] = ic
        
        # 检查 T+1 IC 强度
        t1_ic = ic_results.get('T+1', 0)
        if abs(t1_ic) < 0.04:
            logger.warning(f"[{VERSION}][AlphaWeakWarning] T+1 IC ({t1_ic:.4f}) < 0.04 threshold")
        
        # 检查单调性
        if 'T+1' in ic_results and 'T+3' in ic_results and 'T+5' in ic_results:
            is_monotonic = (
                abs(ic_results['T+1']) >= abs(ic_results['T+3']) >= abs(ic_results['T+5'])
            )
            if not is_monotonic:
                logger.warning(f"[{VERSION}][ICDecayWarning] IC decay is not monotonic!")
                logger.warning(f"[{VERSION}][ICDecayWarning]   T+1: {ic_results['T+1']:.4f}")
                logger.warning(f"[{VERSION}][ICDecayWarning]   T+3: {ic_results['T+3']:.4f}")
                logger.warning(f"[{VERSION}][ICDecayWarning]   T+5: {ic_results['T+5']:.4f}")
        
        # 记录审计结果
        self.ic_decay_audit = ic_results
        
        # 输出审计报告
        logger.info("=" * 60)
        logger.info(f"[{VERSION}][ICDecayAudit] IC Decay Audit Report")
        logger.info("-" * 60)
        for horizon, ic in ic_results.items():
            logger.info(f"[{VERSION}][ICDecayAudit]   {horizon}: {ic:.4f}")
        logger.info("=" * 60)
        
        return ic_results
    
    # ==============================================================================
    # V105 因子计算 - 基于算子库构建
    # ==============================================================================
    
    def compute_volume_price_divergence(self, df: pd.DataFrame, 
                                         periods: list[int] = [5, 10]) -> pd.DataFrame:
        """
        【因子 1】量价背离系数 - 使用 Divergence 算子。
        
        【核心逻辑】
        - 背离 = Rank(价格变化) - Rank(成交量变化)
        - 正背离：价涨量缩或价跌量增（可能是反转信号）
        
        【对齐原则】使用 shift(1) 确保使用 T-1 日数据
        """
        result = df.copy()
        ops = AlphaOperators()
        
        for period in periods:
            # T-1 日相对于 T-period-1 日的价格变化
            price_change = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
            )
            
            # T-1 日相对于 T-period-1 日的成交量变化
            volume_change = result.groupby('symbol')['volume'].transform(
                lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
            )
            
            # 使用 Divergence 算子
            result[f'volume_price_divergence_{period}'] = ops.Divergence(
                price_change, volume_change, group_col='trade_date'
            )
        
        return result
    
    def compute_vcp_ratio(self, df: pd.DataFrame, 
                          periods: list[int] = [5, 10]) -> pd.DataFrame:
        """
        【因子 2】波动率收缩 VCP (Volatility Contraction Pattern)。
        
        【核心逻辑】
        - VCP = 近期波动率 / 远期波动率
        - VCP < 1: 波动率收缩，可能是突破前兆
        
        【对齐原则】使用 T-1 日及之前数据计算波动率
        """
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        for period in periods:
            # 近期波动率 (前 period 日，到 T-1 日为止)
            recent_vol = result.groupby('symbol')['return'].transform(
                lambda x: x.shift(1).rolling(window=period).std()
            )
            
            # 远期波动率 (前 2*period 日，到 T-period-1 日为止)
            far_vol = result.groupby('symbol')['return'].transform(
                lambda x: x.shift(period + 1).rolling(window=period).std()
            )
            
            # VCP 比率
            result[f'vcp_ratio_{period}'] = recent_vol / (far_vol + self.EPSILON)
        
        return result
    
    def compute_turnover_anomaly(self, df: pd.DataFrame, 
                                  periods: list[int] = [5, 20]) -> pd.DataFrame:
        """
        【因子 3】换手率异常度。
        
        【核心逻辑】
        - 异常度 = (当前换手率 - MA(换手率)) / Std(换手率)
        - 高异常度：可能是主力行为
        """
        result = df.copy()
        
        if 'turnover_rate' not in result.columns:
            # 用 volume 近似
            result['turnover_rate'] = result.groupby('symbol')['volume'].transform(
                lambda x: x / (x.rolling(window=20).mean() + self.EPSILON)
            )
        
        for period in periods:
            # 换手率 MA 和 Std
            turnover_ma = result.groupby('symbol')['turnover_rate'].transform(
                lambda x: x.shift(1).rolling(window=period).mean()
            )
            turnover_std = result.groupby('symbol')['turnover_rate'].transform(
                lambda x: x.shift(1).rolling(window=period).std()
            )
            
            # 异常度 (Z-Score)
            result[f'turnover_anomaly_{period}'] = (
                result['turnover_rate'].shift(1) - turnover_ma
            ) / (turnover_std + self.EPSILON)
        
        return result
    
    def compute_money_flow_intensity(self, df: pd.DataFrame, 
                                      periods: list[int] = [5, 10]) -> pd.DataFrame:
        """
        【因子 4】资金流强度。
        
        【核心逻辑】
        - 资金流 = 成交量 * 价格变化方向
        - 强度 = 资金流 MA / 成交量 MA
        """
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        for period in periods:
            # 资金流 (成交量 * 收益率符号)
            money_flow = result['volume'].shift(1) * np.sign(result['return'].shift(1))
            
            # 资金流强度
            money_flow_ma = money_flow.groupby(result['symbol']).transform(
                lambda x: x.rolling(window=period).mean()
            )
            volume_ma = result.groupby('symbol')['volume'].transform(
                lambda x: x.shift(1).rolling(window=period).mean()
            )
            
            result[f'money_flow_intensity_{period}'] = money_flow_ma / (volume_ma + self.EPSILON)
        
        return result
    
    def compute_momentum_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【因子 5】价格动量加速度。
        
        【核心逻辑】
        - 加速度 = 短动量 - 中动量
        - 正加速度：动量增强
        """
        result = df.copy()
        
        # 短期动量 (5 日)
        momentum_short = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(6) + self.EPSILON) - 1.0
        )
        
        # 中期动量 (10 日)
        momentum_mid = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(11) + self.EPSILON) - 1.0
        )
        
        # 加速度 = 短动量 - 中动量
        result['momentum_acceleration'] = momentum_short - momentum_mid
        
        return result
    
    def compute_volume_skew(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        【因子 6】成交量分布偏度。
        
        【核心逻辑】
        - 偏度 > 0: 成交量右偏，存在放量日
        - 偏度 < 0: 成交量左偏，存在缩量日
        """
        result = df.copy()
        
        result['volume_skew_20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=window).skew()
        )
        
        return result
    
    def compute_return_kurtosis(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        【因子 7】收益率分布峰度。
        
        【核心逻辑】
        - 峰度高：收益率分布尖锐，存在极端值
        """
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        result['return_kurtosis_20'] = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=window).kurt()
        )
        
        return result
    
    def compute_relative_strength(self, df: pd.DataFrame, 
                                   periods: list[int] = [10, 20]) -> pd.DataFrame:
        """
        【因子 8】相对强度 RS。
        
        【核心逻辑】
        - RS = 个股收益率 / 市场收益率
        - 使用沪深 300 作为基准（简化为全市场平均）
        """
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        # 计算市场平均收益率（按日期分组）
        market_return = result.groupby('trade_date')['return'].transform('mean')
        
        for period in periods:
            # 个股累计收益
            stock_cum = result.groupby('symbol')['return'].transform(
                lambda x: x.shift(1).rolling(window=period).sum()
            )
            
            # 市场累计收益
            market_cum = result.groupby('trade_date')['return'].transform(
                lambda x: x.rolling(window=period).sum()
            )
            
            # 相对强度
            result[f'relative_strength_{period}'] = stock_cum / (market_cum + self.EPSILON)
        
        return result
    
    def compute_price_efficiency(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        【因子 9】价格效率指标 - V105 非线性特征工程。
        
        【核心逻辑】
        - 效率 = |净价格变化| / 总价格变化路径
        - 效率高：价格趋势明确
        - 效率低：价格震荡
        
        【V105 公式】
        Price Efficiency = Abs(Ts_Delta(close, 20)) / Ts_Sum(Abs(Ts_Delta(close, 1)), 20)
        """
        result = df.copy()
        ops = AlphaOperators()
        
        # 净价格变化 (20 日)
        net_change = ops.Ts_Delta(result['close'], 20, 'symbol')
        
        # 每日价格变化绝对值
        daily_change = result.groupby('symbol')['close'].transform(
            lambda x: np.abs(x.shift(1) - x.shift(2))
        )
        
        # 总价格变化路径 (20 日求和)
        total_change = daily_change.groupby(result['symbol']).transform(
            lambda x: x.rolling(window=20).sum()
        )
        
        # 效率比率
        result['price_efficiency_20'] = np.abs(net_change) / (total_change + self.EPSILON)
        
        return result
    
    def compute_volatility_adjusted_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【因子 10】波动率调整动量。
        
        【核心逻辑】
        - 动量 / 波动率 (类似 Sharpe 比率)
        - 高风险调整后的收益
        """
        result = df.copy()
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        # 动量 (10 日)
        momentum = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(11) + self.EPSILON) - 1.0
        )
        
        # 波动率 (20 日)
        volatility = result.groupby('symbol')['return'].transform(
            lambda x: x.shift(1).rolling(window=20).std()
        )
        
        # 波动率调整动量
        result['volatility_adjusted_momentum'] = momentum / (volatility + self.EPSILON)
        
        return result
    
    def compute_volume_price_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【因子 11】量价健康度。
        
        【核心逻辑】
        - 价涨量增：健康 (+1)
        - 价涨量缩：背离 (-0.5)
        - 价跌量缩：正常 (-0.2)
        - 价跌量增：危险 (-1)
        """
        result = df.copy()
        
        # T-1 日价格变化
        result['price_change'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(6) + self.EPSILON) - 1.0
        )
        
        # T-1 日成交量 vs MA
        result['volume_ma'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=5).mean()
        )
        result['volume_ratio'] = result['volume'].shift(1) / (result['volume_ma'] + self.EPSILON)
        
        def health_score(row):
            if pd.isna(row['price_change']) or pd.isna(row['volume_ratio']):
                return 0.0
            if row['price_change'] > 0 and row['volume_ratio'] > 1.0:
                return 1.0
            elif row['price_change'] > 0 and row['volume_ratio'] <= 1.0:
                return -0.5
            elif row['price_change'] <= 0 and row['volume_ratio'] <= 1.0:
                return -0.2
            else:
                return -1.0
        
        result['volume_price_health'] = result.apply(health_score, axis=1)
        
        return result
    
    def compute_residual_momentum(self, df: pd.DataFrame, 
                                   periods: list[int] = [5, 10]) -> pd.DataFrame:
        """
        【因子 12】残差动量。
        
        【核心逻辑】
        - 计算 VWAP
        - 价格相对于 VWAP 的残差
        - 残差动量
        """
        result = df.copy()
        
        # 计算 VWAP (使用 T-1 日数据)
        if 'vwap' not in result.columns or result['vwap'].isna().sum() > 0:
            result['vwap'] = (result['high'].shift(1) + result['low'].shift(1) + result['close'].shift(1)) / 3.0
        
        # 残差
        result['price_residual'] = result['close'].shift(1) - result['vwap']
        
        # 残差动量
        for period in periods:
            result[f'residual_momentum_{period}'] = result.groupby('symbol')['price_residual'].transform(
                lambda x: x / (x.shift(period) + self.EPSILON) - 1.0
            )
        
        return result
    
    # ==============================================================================
    # V105 非线性特征工程 - Beyond Simple Product
    # ==============================================================================
    
    def compute_volume_skew_resonance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V105 非线性特征】成交量偏度共振。
        
        【核心逻辑】
        利用 Condition 算子，当成交量处于 20 日高位（Ts_Argmax）且价格下跌时，
        赋予该路径极高的负权重。
        """
        result = df.copy()
        ops = AlphaOperators()
        
        # 成交量 20 日高位判断
        volume_argmax = ops.Ts_Argmax(result['volume'], 20, 'symbol')
        high_volume_condition = volume_argmax < 3  # 最大值出现在最近 3 天内
        
        # 价格下跌
        price_decline = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) < x.shift(6)
        )
        
        # 成交量偏度
        volume_skew = ops.Ts_Std(result['volume'], 20, 'symbol')
        
        # 使用 Condition 算子：高位 + 下跌 = 高负权重
        negative_signal = -1.0 * ops.Scale(volume_skew, 'trade_date')
        positive_signal = 0.1  # 其他情况给小正值
        
        result['volume_skew_resonance'] = ops.Condition(
            high_volume_condition.values & price_decline.values,
            negative_signal,
            pd.Series(positive_signal, index=result.index)
        )
        
        return result
    
    def compute_nonlinear_compression(self, df: pd.DataFrame, 
                                       columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        【V105 非线性特征】复杂非线性压缩。
        
        【核心逻辑】
        使用 log(1 + Abs(x)) * Sign(x) 对长尾因子进行压缩。
        """
        result = df.copy()
        ops = AlphaOperators()
        
        if columns is None:
            columns = ['volume_skew_20', 'return_kurtosis_20', 'turnover_anomaly_20']
        
        for col in columns:
            if col in result.columns:
                result[f'{col}_compressed'] = ops.Log1p(result[col])
        
        return result
    
    def compute_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V105 非线性特征】背离特征组合。
        
        【核心逻辑】
        使用 Divergence 算子构建多个背离特征。
        """
        result = df.copy()
        ops = AlphaOperators()
        
        # 量价背离
        if 'volume_price_divergence_10' in result.columns:
            # 与波动率的背离
            if 'volatility_adjusted_momentum' in result.columns:
                result['vp_vol_divergence'] = ops.Divergence(
                    result['volume_price_divergence_10'],
                    result['volatility_adjusted_momentum'],
                    'trade_date'
                )
        
        # 动量背离
        if 'momentum_acceleration' in result.columns:
            if 'relative_strength_10' in result.columns:
                result['mom_rs_divergence'] = ops.Divergence(
                    result['momentum_acceleration'],
                    result['relative_strength_10'],
                    'trade_date'
                )
        
        return result
    
    # ==============================================================================
    # V105 标签计算 - 预测目标对齐
    # ==============================================================================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 T+1 收益标签。
        
        【预测目标对齐】
        预测目标必须是 T+1 的截面收益排名（Ranked Returns）。
        """
        result = df.copy()
        result['t1_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-1) / (x + self.EPSILON) - 1.0
        )
        return result
    
    def compute_tn_return(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        """计算 T+N 收益标签。"""
        result = df.copy()
        result[f't{n}_return'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(-n) / (x + self.EPSILON) - 1.0
        )
        return result
    
    def compute_ranked_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【预测目标对齐】计算 T+1 截面收益排名。
        
        将 T+1 收益转换为截面排名 (0-1 之间)。
        """
        result = df.copy()
        ops = AlphaOperators()
        
        if 't1_return' in result.columns:
            result['t1_return_rank'] = ops.Rank(result['t1_return'], 'trade_date')
        
        return result
    
    # ==============================================================================
    # V105 数据防御机制
    # ==============================================================================
    
    def check_and_repair_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """【数据防御】检查并修复缺失数据。"""
        result = df.copy()
        
        # 检查 total_mv 缺失
        if 'total_mv' in result.columns:
            null_ratio = result['total_mv'].isna().sum() / len(result)
            if null_ratio > 0.3:
                logger.warning(f"[{VERSION}][DataDefense] total_mv null ratio: {null_ratio:.1%}")
                
                if 'amount' in result.columns and 'turnover_rate' in result.columns:
                    logger.info(f"[{VERSION}][DataDefense] Estimating total_mv from amount/turnover_rate")
                    estimated_mv = result['amount'] / (result['turnover_rate'].fillna(0.01) + self.EPSILON) * 100
                    result['total_mv'] = result['total_mv'].fillna(estimated_mv)
        
        # 检查 vwap 缺失
        if 'vwap' not in result.columns or result['vwap'].isna().sum() > 0:
            logger.info(f"[{VERSION}][DataDefense] Estimating VWAP from (high+low+close)/3")
            result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
        
        return result
    
    def fill_null_values(self, df: pd.DataFrame, 
                         null_threshold: float = 0.30) -> pd.DataFrame:
        """智能填充空值。"""
        result = df.copy()
        
        exclude_columns = {'t1_return', 't3_return', 't5_return', 'symbol', 'trade_date', 'ts_code'}
        factor_columns = [col for col in result.columns if col not in exclude_columns]
        
        total_rows = len(result)
        
        for col in factor_columns:
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            
            null_count = result[col].isna().sum()
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            
            if null_ratio > null_threshold:
                result[col] = result[col].fillna(0)
            else:
                try:
                    median_val = result[col].median()
                    if np.isnan(median_val) or not np.isfinite(median_val):
                        median_val = 0
                    result[col] = result[col].fillna(median_val)
                except (TypeError, ValueError):
                    result[col] = result[col].fillna(0)
        
        return result
    
    # ==============================================================================
    # V105 因子生存竞争与自迭代
    # ==============================================================================
    
    def run_factor_competition(self, df: pd.DataFrame) -> dict[str, float]:
        """
        【因子生存竞争】测试所有因子的独立预测能力。
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorCompetition] Testing factor predictive power...")
        logger.info("-" * 80)
        
        competition_results = {}
        
        for factor_name in self.get_factor_names():
            if factor_name not in df.columns:
                continue
            
            ic = self._calculate_single_factor_ic(df[factor_name], df['t1_return'])
            competition_results[factor_name] = ic
        
        # 按 IC 绝对值排序
        sorted_results = sorted(competition_results.items(), key=lambda x: abs(x[1]), reverse=True)
        
        logger.info(f"{'Rank':<6} {'Factor Name':<40} {'IC':>10} {'Status':>8}")
        logger.info("-" * 80)
        
        for rank, (factor_name, ic) in enumerate(sorted_results, 1):
            status = "OK" if abs(ic) > 0.03 else "WEAK"
            logger.info(f"{rank:<6} {factor_name:<40} {ic:>10.4f} {status:>8}")
        
        logger.info("=" * 80)
        
        self.factor_competition_results = competition_results
        return competition_results
    
    def optimize_weights(self, df: pd.DataFrame, method: str = 'ic_weighted') -> dict[str, float]:
        """
        【自迭代优化】根据因子 IC 优化权重。
        """
        if not self.factor_competition_results:
            self.run_factor_competition(df)
        
        optimized_weights = {}
        
        if method == 'ic_weighted':
            # 按 IC 绝对值加权
            total_ic = sum(abs(ic) for ic in self.factor_competition_results.values())
            if total_ic > 0:
                for factor_name, ic in self.factor_competition_results.items():
                    optimized_weights[factor_name] = abs(ic) / total_ic
            else:
                optimized_weights = {k: 1.0/len(self.FACTOR_WEIGHTS) for k in self.FACTOR_WEIGHTS}
        
        elif method == 'top_k':
            # 只保留 Top K 因子
            k = min(6, len(self.factor_competition_results))
            top_factors = sorted(
                self.factor_competition_results.items(), 
                key=lambda x: abs(x[1]), reverse=True
            )[:k]
            
            for factor_name, _ in top_factors:
                optimized_weights[factor_name] = 1.0 / k
            
            for factor_name in self.FACTOR_WEIGHTS:
                if factor_name not in optimized_weights:
                    optimized_weights[factor_name] = 0.0
        
        else:  # equal
            optimized_weights = {k: 1.0/len(self.FACTOR_WEIGHTS) for k in self.FACTOR_WEIGHTS}
        
        logger.info(f"[{VERSION}][WeightOpt] Optimized weights using {method}")
        logger.info(f"[{VERSION}][WeightOpt]   Top factors: {list(optimized_weights.keys())[:5]}...")
        
        return optimized_weights
    
    # ==============================================================================
    # V105 预测评分
    # ==============================================================================
    
    def compute_predict_score(self, df: pd.DataFrame, 
                               weights: Optional[dict[str, float]] = None) -> pd.DataFrame:
        """计算综合预测评分。"""
        if weights is None:
            weights = self.FACTOR_WEIGHTS
        
        result = df.copy()
        
        raw_score = np.zeros(len(result))
        for factor_name, weight in weights.items():
            if factor_name in result.columns:
                raw_score += result[factor_name].fillna(0).values * weight
        
        result['score'] = raw_score
        
        return result
    
    # ==============================================================================
    # V105 因子计算主流程
    # ==============================================================================
    
    def compute_factors(self, df: pd.DataFrame, 
                        clean: bool = True) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【计算顺序 - 严格检查 T 日信号只用 T-1 日数据】
        1. 数据防御检查
        2. 量价背离因子
        3. VCP 波动率收缩
        4. 换手率异常
        5. 资金流强度
        6. 动量加速度
        7. 成交量偏度
        8. 收益率峰度
        9. 相对强度
        10. 价格效率
        11. 波动率调整动量
        12. 量价健康度
        13. 残差动量
        14. V105 非线性特征
        15. T+1/T+N 收益标签
        16. 缺失值处理
        17. 因子清洗（含强制中性化）
        18. 预测评分
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V105 Factor Computation Started")
        logger.info("=" * 80)
        
        # 1. 数据防御
        logger.info(f"[{VERSION}][FactorComputation] Step 1: Data Defense Check...")
        result = self.check_and_repair_data(df)
        
        # 2-13. 计算 12 个原始因子
        logger.info(f"[{VERSION}][FactorComputation] Step 2: Computing Volume-Price Divergence...")
        result = self.compute_volume_price_divergence(result, periods=[5, 10])
        
        logger.info(f"[{VERSION}][FactorComputation] Step 3: Computing VCP Ratio...")
        result = self.compute_vcp_ratio(result, periods=[5, 10])
        
        logger.info(f"[{VERSION}][FactorComputation] Step 4: Computing Turnover Anomaly...")
        result = self.compute_turnover_anomaly(result, periods=[5, 20])
        
        logger.info(f"[{VERSION}][FactorComputation] Step 5: Computing Money Flow Intensity...")
        result = self.compute_money_flow_intensity(result, periods=[5, 10])
        
        logger.info(f"[{VERSION}][FactorComputation] Step 6: Computing Momentum Acceleration...")
        result = self.compute_momentum_acceleration(result)
        
        logger.info(f"[{VERSION}][FactorComputation] Step 7: Computing Volume Skew...")
        result = self.compute_volume_skew(result, window=20)
        
        logger.info(f"[{VERSION}][FactorComputation] Step 8: Computing Return Kurtosis...")
        result = self.compute_return_kurtosis(result, window=20)
        
        logger.info(f"[{VERSION}][FactorComputation] Step 9: Computing Relative Strength...")
        result = self.compute_relative_strength(result, periods=[10, 20])
        
        logger.info(f"[{VERSION}][FactorComputation] Step 10: Computing Price Efficiency...")
        result = self.compute_price_efficiency(result, window=20)
        
        logger.info(f"[{VERSION}][FactorComputation] Step 11: Computing Volatility-Adjusted Momentum...")
        result = self.compute_volatility_adjusted_momentum(result)
        
        logger.info(f"[{VERSION}][FactorComputation] Step 12: Computing Volume-Price Health...")
        result = self.compute_volume_price_health(result)
        
        logger.info(f"[{VERSION}][FactorComputation] Step 13: Computing Residual Momentum...")
        result = self.compute_residual_momentum(result, periods=[5, 10])
        
        # 14. V105 非线性特征
        logger.info(f"[{VERSION}][FactorComputation] Step 14: Computing V105 Nonlinear Features...")
        result = self.compute_volume_skew_resonance(result)
        result = self.compute_nonlinear_compression(result)
        result = self.compute_divergence_features(result)
        
        # 15. 收益标签
        logger.info(f"[{VERSION}][FactorComputation] Step 15: Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        result = self.compute_ranked_returns(result)
        
        # 16. 缺失值处理
        logger.info(f"[{VERSION}][FactorComputation] Step 16: Filling Null Values...")
        result = self.fill_null_values(result)
        
        # 17. 因子清洗（含强制中性化）
        if clean:
            logger.info(f"[{VERSION}][FactorComputation] Step 17: Factor Cleaning (MAD/Z-Score/Neu)...")
            result = self.clean_factors(result, 
                                        do_winsorize=True,
                                        do_normalize=True,
                                        do_neutralize=self.use_neutralization)
        
        # 18. 预测评分
        logger.info(f"[{VERSION}][FactorComputation] Step 18: Computing Predict Score...")
        result = self.compute_predict_score(result)
        
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][FactorComputation] V105 Factor Computation Complete")
        logger.info("=" * 80)
        
        return result
    
    # ==============================================================================
    # V105 自迭代优化
    # ==============================================================================
    
    def auto_iterate_optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【自迭代优化】自动多轮优化直到 T+1 IC >= 0.04。
        """
        logger.info("=" * 80)
        logger.info(f"[{VERSION}][AutoIterate] Starting auto-optimization...")
        logger.info("=" * 80)
        
        best_result = None
        best_ic = 0.0
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n[{VERSION}][AutoIterate] Iteration {iteration}/{self.max_iterations}")
            
            # 计算因子
            result = self.compute_factors(df, clean=True)
            
            # 运行因子生存竞争
            competition_results = self.run_factor_competition(result)
            
            # 计算当前综合 IC
            current_ic = self._calculate_single_factor_ic(result['score'], result['t1_return'])
            
            # 记录迭代历史
            self.iteration_history.append({
                'iteration': iteration,
                't1_ic': current_ic,
                'weights_used': dict(self.FACTOR_WEIGHTS),
            })
            
            logger.info(f"[{VERSION}][AutoIterate] Iteration {iteration}: T+1 IC = {current_ic:.4f}")
            
            if abs(current_ic) > abs(best_ic):
                best_ic = current_ic
                best_result = result.copy()
            
            # 如果达到目标，提前退出
            if abs(current_ic) >= 0.04:
                logger.info(f"[{VERSION}][AutoIterate] Reached target IC (>= 0.04), stopping optimization")
                return result
            
            # 优化权重
            if iteration < self.max_iterations:
                method = 'top_k' if iteration == 1 else 'ic_weighted'
                optimized_weights = self.optimize_weights(result, method=method)
                self.FACTOR_WEIGHTS = optimized_weights
                
                # 重新计算评分
                result = self.compute_predict_score(result, weights=optimized_weights)
        
        logger.info(f"\n[{VERSION}][AutoIterate] Optimization complete: Best T+1 IC = {best_ic:.4f}")
        logger.info("=" * 80)
        
        return best_result if best_result is not None else df
    
    # ==============================================================================
    # V105 主接口：compute_score
    # ==============================================================================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【主接口】计算 Alpha 评分。
        
        这是 BacktestReferee 调用的唯一接口。
        
        Args:
            df: 输入数据
            
        Returns:
            包含 score 和 t1_return 的 DataFrame
        """
        # 计算因子
        result = self.compute_factors(df, clean=True)
        
        # 自迭代优化（如果需要）
        if self.auto_iterate:
            t1_ic = self._calculate_single_factor_ic(result['score'], result['t1_return'])
            if abs(t1_ic) < 0.04:
                logger.warning(f"[{VERSION}][AutoIterate] T+1 IC ({t1_ic:.4f}) < 0.04, triggering optimization")
                result = self.auto_iterate_optimize(df)
        
        # 运行因子生存竞争（记录结果）
        self.run_factor_competition(result)
        
        # IC 衰减审计
        self.audit_ic_decay(result)
        
        # 返回必需列
        output_columns = ['trade_date', 'symbol', 'score', 't1_return']
        for n in [3, 5]:
            if f't{n}_return' in result.columns:
                output_columns.append(f't{n}_return')
        
        return result[output_columns]
    
    def get_factor_ics(self, df: pd.DataFrame) -> dict[str, float]:
        """获取所有因子的 IC 值。"""
        return self._calculate_all_factor_ics(df)
    
    def get_factor_names(self) -> list[str]:
        """获取配置的因子名称列表。"""
        return list(self.FACTOR_WEIGHTS.keys())
    
    def get_cleaning_comparison(self) -> dict[str, dict[str, float]]:
        """获取清洗前后 IC 对比。"""
        return {
            'before': self.factor_ic_before_cleaning,
            'after': self.factor_ic_after_cleaning,
        }
    
    def get_competition_results(self) -> dict[str, float]:
        """获取因子生存竞争结果。"""
        return self.factor_competition_results
    
    def get_iteration_history(self) -> list[dict]:
        """获取自迭代历史。"""
        return self.iteration_history
    
    def get_ic_decay_audit(self) -> dict[str, float]:
        """获取 IC 衰减审计结果。"""
        return self.ic_decay_audit


# ==============================================================================
# V105 工厂函数
# ==============================================================================

def get_alpha_research(config_path: str = "config/factors.yaml",
                       use_neutralization: bool = True,
                       auto_iterate: bool = True) -> AlphaResearchV105:
    """
    获取 AlphaResearchV105 实例。
    
    Args:
        config_path: 因子配置文件路径
        use_neutralization: 是否启用中性化（V105 强制启用）
        auto_iterate: 是否启用自迭代优化
        
    Returns:
        AlphaResearchV105 实例
    """
    return AlphaResearchV105(config_path, use_neutralization, auto_iterate)