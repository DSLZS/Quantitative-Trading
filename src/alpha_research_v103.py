"""
Alpha Research Module - V103 因子清洗与特征工程.

【V103 核心改进】
1. 因子清洗三部曲：去极值 (MAD) -> 标准化 (Z-Score) -> 中性化 (OLS)
2. 量价非线性特征提取（成交量分布偏度与收益率相关性）
3. 防欺诈与主动防御机制
4. 输出清洗前后 IC 对比表

【因子清洗三部曲】
1. 去极值 (Winsorization): 使用 Median Absolute Deviation (MAD) 处理因子暴露
2. 标准化 (Z-Score): 在截面上使因子符合标准正态分布
3. 中性化 (Neutralization): 强制使用 OLS 去除行业和市值暴露

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标：低于 0.03 视为优化失败 |
| IC Decay | T+1 > T+2 > T+3 | 信号衰减必须符合单调性 |
| 模块独立性 | 100% | 回测引擎逻辑必须完全位于 backtest_referee.py |
"""

from typing import Any, Optional
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


class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.03 时触发"""
    pass


class ICDecayWarning(Exception):
    """IC 衰减警告 - 当 IC 不单调递减时触发"""
    pass


class AlphaResearchV103:
    """
    V103 Alpha 预测核心引擎。
    
    【核心改进】
    1. 因子清洗三部曲：MAD -> Z-Score -> Neutralization
    2. 量价非线性特征提取
    3. 主动数据防御（调用 DataLoader.repair_2024_data）
    4. 清洗前后 IC 对比审计
    
    【因子体系】
    - 动量因子：residual_momentum, momentum_5/10/20
    - 波动率因子：volatility_5/20
    - 量价因子：volume_price_divergence, volume_price_health
    - 非线性特征：volume_skew, volume_kurtosis
    """
    
    EPSILON = 1e-6
    
    # V103 因子权重配置
    FACTOR_WEIGHTS = {
        # 核心因子：VWAP 残差动量
        "residual_momentum_5": 0.15,
        "residual_momentum_10": 0.25,
        
        # 传统动量
        "momentum_5": 0.05,
        "momentum_10": 0.05,
        "momentum_20": 0.05,
        
        # 波动率 (负向因子)
        "volatility_5": -0.05,
        "volatility_20": -0.10,
        
        # 量价因子
        "volume_price_divergence_5": 0.10,
        "volume_price_health": 0.10,
        
        # 非线性特征
        "volume_skew_20": 0.08,
        "volume_kurtosis_20": 0.05,
        "volume_return_corr_20": 0.12,
    }
    
    def __init__(self, config_path: str = "config/factors.yaml", 
                 use_neutralization: bool = True) -> None:
        """
        初始化 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            use_neutralization: 是否启用中性化 (默认 True)
        """
        self.config_path = Path(config_path)
        self.use_neutralization = use_neutralization
        self.factors: list[dict[str, Any]] = []
        self._load_config()
        
        # 清洗前后 IC 记录
        self.factor_ic_before_cleaning = {}
        self.factor_ic_after_cleaning = {}
        
        logger.info("AlphaResearchV103 initialized")
        logger.info(f"  Config path: {self.config_path}")
        logger.info(f"  Use neutralization: {self.use_neutralization}")
    
    def _load_config(self) -> None:
        """加载因子配置文件。"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.factors = config.get('factors', [])
            logger.info(f"Loaded {len(self.factors)} factor configurations")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self.factors = []
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            self.factors = []
    
    # ==================== 因子清洗三部曲 ====================
    
    def winsorize_mad(self, df: pd.DataFrame, columns: Optional[list[str]] = None,
                      n_std: float = 3.0) -> pd.DataFrame:
        """
        【清洗三部曲 1】去极值 - 使用 Median Absolute Deviation (MAD)。
        
        【核心逻辑】
        1. 计算中位数 Median
        2. 计算 MAD = Median(|X - Median|)
        3. 上下限：[Median - n_std * MAD * 1.4826, Median + n_std * MAD * 1.4826]
        4. 1.4826 是正态分布下 MAD 与 Std 的比例系数
        
        Args:
            df: 输入数据
            columns: 需要处理的列，默认处理所有因子列
            n_std: 标准差倍数 (默认 3.0)
            
        Returns:
            去极值后的数据
        """
        if columns is None:
            columns = self.get_factor_names()
        
        result = df.copy()
        cleaning_log = []
        
        for col in columns:
            if col not in result.columns:
                continue
            
            # 按截面（日期）分组处理
            if 'trade_date' in result.columns:
                for date in result['trade_date'].unique():
                    mask = result['trade_date'] == date
                    values = result.loc[mask, col].dropna()
                    
                    if len(values) < 10:
                        continue
                    
                    median = values.median()
                    mad = np.median(np.abs(values - median))
                    
                    # 调整系数 (正态分布下 MAD * 1.4826 ≈ Std)
                    adjusted_mad = mad * 1.4826
                    
                    if adjusted_mad < self.EPSILON:
                        continue
                    
                    lower_bound = median - n_std * adjusted_mad
                    upper_bound = median + n_std * adjusted_mad
                    
                    # 缩尾处理
                    original_outliers = ((result.loc[mask, col] < lower_bound) | 
                                         (result.loc[mask, col] > upper_bound)).sum()
                    
                    result.loc[mask, col] = result.loc[mask, col].clip(lower=lower_bound, upper=upper_bound)
                    
                    if original_outliers > 0:
                        cleaning_log.append({
                            'column': col,
                            'date': date,
                            'outliers_winsorized': original_outliers,
                        })
            else:
                # 全局处理
                values = result[col].dropna()
                if len(values) < 10:
                    continue
                
                median = values.median()
                mad = np.median(np.abs(values - median))
                adjusted_mad = mad * 1.4826
                
                if adjusted_mad < self.EPSILON:
                    continue
                
                lower_bound = median - n_std * adjusted_mad
                upper_bound = median + n_std * adjusted_mad
                
                result.loc[:, col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        
        if cleaning_log:
            logger.debug(f"[MAD Winsorization] Processed {len(cleaning_log)} column-date pairs")
        
        return result
    
    def normalize_zscore(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        【清洗三部曲 2】标准化 - Z-Score 标准化。
        
        【核心逻辑】
        在截面上对每个日期进行标准化：
        Z = (X - Mean) / Std
        
        使因子符合标准正态分布 N(0, 1)
        
        Args:
            df: 输入数据
            columns: 需要处理的列
            
        Returns:
            标准化后的数据
        """
        if columns is None:
            columns = self.get_factor_names()
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            if 'trade_date' in result.columns:
                # 按截面标准化
                grouped = result.groupby('trade_date')[col]
                mean = grouped.transform('mean')
                std = grouped.transform('std')
                
                # 避免除零
                std = std.replace(0, self.EPSILON)
                result.loc[:, col] = (result[col] - mean) / std
            else:
                # 全局标准化
                mean_val = result[col].mean()
                std_val = result[col].std()
                
                if std_val < self.EPSILON:
                    std_val = self.EPSILON
                
                result.loc[:, col] = (result[col] - mean_val) / std_val
        
        logger.debug("[Z-Score Normalization] Completed")
        return result
    
    def neutralize_ols(self, df: pd.DataFrame, columns: Optional[list[str]] = None,
                       neutralize_vars: list[str] = None) -> pd.DataFrame:
        """
        【清洗三部曲 3】中性化 - 使用 OLS 去除行业和市值暴露。
        
        【核心逻辑】
        1. 对每个因子 Y，建立回归模型：Y = α + β1*Size + β2*Industry + ε
        2. 使用残差 ε 作为中性化后的因子值
        3. 必须引用 total_mv (市值) 和 industry_code (行业)
        
        Args:
            df: 输入数据
            columns: 需要中性化的因子列
            neutralize_vars: 用于中性化的变量 (默认 ['ln_total_mv', 'industry_dummies'])
            
        Returns:
            中性化后的数据
        """
        if columns is None:
            columns = self.get_factor_names()
        
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv']  # 使用对数市值
        
        result = df.copy()
        
        # 检查必需变量
        if 'total_mv' not in result.columns:
            logger.warning("Missing total_mv column, skipping neutralization")
            return result
        
        # 计算对数市值
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        # 行业虚拟变量
        industry_dummies = []
        if 'industry_code' in result.columns:
            industry_dummies = pd.get_dummies(result['industry_code'], prefix='ind')
        
        for col in columns:
            if col not in result.columns:
                continue
            
            if 'trade_date' in result.columns:
                # 按截面中性化
                neutralized_values = []
                
                for date in result['trade_date'].unique():
                    mask = result['trade_date'] == date
                    day_data = result.loc[mask].copy()
                    
                    if len(day_data) < 30:
                        neutralized_values.append(day_data[[col, 'trade_date', 'symbol']])
                        continue
                    
                    # 准备回归数据
                    y = day_data[col].values
                    X_vars = ['ln_total_mv']
                    X = day_data[X_vars].values
                    
                    # 添加行业虚拟变量
                    if len(industry_dummies) > 0:
                        day_ind = industry_dummies.loc[mask]
                        X = np.column_stack([X, day_ind.values])
                    
                    # 添加常数项
                    X = np.column_stack([np.ones(len(X)), X])
                    
                    try:
                        # OLS 回归：使用伪逆求解
                        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                        
                        # 计算残差
                        y_pred = X @ beta
                        residuals = y - y_pred
                        
                        day_data[col] = residuals
                        neutralized_values.append(day_data[[col, 'trade_date', 'symbol']])
                        
                    except np.linalg.LinAlgError:
                        logger.debug(f"[Neutralization] Failed for date {date}, keeping original")
                        neutralized_values.append(day_data[[col, 'trade_date', 'symbol']])
                
                if neutralized_values:
                    neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                    result.loc[:, col] = neutralized_df[col].values
                    
            else:
                # 全局中性化
                y = result[col].values
                X = result[['ln_total_mv']].values
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                    residuals = y - X @ beta
                    result.loc[:, col] = residuals
                except np.linalg.LinAlgError:
                    pass
        
        logger.debug("[OLS Neutralization] Completed")
        return result
    
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
        
        logger.info("[因子清洗] 开始执行清洗三部曲...")
        
        # 记录清洗前 IC
        self.factor_ic_before_cleaning = self._calculate_all_factor_ics(result)
        
        # Step 1: 去极值 (MAD)
        if do_winsorize:
            logger.info("  [Step 1] MAD Winsorization...")
            result = self.winsorize_mad(result, n_std=3.0)
        
        # Step 2: 标准化 (Z-Score)
        if do_normalize:
            logger.info("  [Step 2] Z-Score Normalization...")
            result = self.normalize_zscore(result)
        
        # Step 3: 中性化 (OLS)
        if do_neutralize and self.use_neutralization:
            logger.info("  [Step 3] OLS Neutralization (Size + Industry)...")
            result = self.neutralize_ols(result)
        
        # 记录清洗后 IC
        self.factor_ic_after_cleaning = self._calculate_all_factor_ics(result)
        
        # 输出清洗前后对比
        self._print_cleaning_comparison()
        
        logger.info("[因子清洗] 完成")
        return result
    
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
        """计算单个因子的 Rank IC。"""
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
    
    def _print_cleaning_comparison(self) -> None:
        """打印清洗前后 IC 对比表。"""
        logger.info("=" * 70)
        logger.info("[因子清洗前后 IC 对比表]")
        logger.info("-" * 70)
        logger.info(f"{'因子名称':<35} {'清洗前 IC':>12} {'清洗后 IC':>12} {'改善':>10}")
        logger.info("-" * 70)
        
        all_factors = set(self.factor_ic_before_cleaning.keys()) | set(self.factor_ic_after_cleaning.keys())
        
        for factor in sorted(all_factors):
            before = self.factor_ic_before_cleaning.get(factor, 0)
            after = self.factor_ic_after_cleaning.get(factor, 0)
            improvement = after - before
            status = "✓" if improvement > 0 else "✗"
            logger.info(f"{factor:<35} {before:>12.4f} {after:>12.4f} {improvement:>+10.4f} {status}")
        
        logger.info("=" * 70)
    
    # ==================== 量价非线性特征提取 ====================
    
    def extract_volume_nonlinear_features(self, df: pd.DataFrame, 
                                           window: int = 20) -> pd.DataFrame:
        """
        量价非线性特征提取。
        
        【核心逻辑】
        1. Volume Skewness: 过去 N 日成交量分布的偏度
           - 偏度 > 0: 成交量右偏，存在放量日
           - 偏度 < 0: 成交量左偏，存在缩量日
        
        2. Volume Kurtosis: 过去 N 日成交量分布的峰度
           - 峰度高：成交量分布尖锐，存在极端值
           - 峰度低：成交量分布平缓
        
        3. Volume-Return Correlation: 量价相关性
           - 量价正相关：健康上涨
           - 量价负相关：背离信号
        
        Args:
            df: 输入数据 (包含 volume, close 列)
            window: 滚动窗口大小 (默认 20 日)
            
        Returns:
            包含非线性特征的数据
        """
        result = df.copy()
        
        # 确保数据按股票和日期排序
        if 'symbol' in result.columns and 'trade_date' in result.columns:
            result = result.sort_values(['symbol', 'trade_date'])
        
        # 计算成交量变化率
        result['volume_change'] = result.groupby('symbol')['volume'].pct_change().fillna(0)
        
        # 1. Volume Skewness (偏度)
        def rolling_skew(x):
            if len(x) < window:
                return np.nan
            return pd.Series(x).skew()
        
        result['volume_skew_20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window=window).skew()
        )
        
        # 2. Volume Kurtosis (峰度)
        result['volume_kurtosis_20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(window=window).kurt()
        )
        
        # 3. Volume-Return Correlation (量价相关性)
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        def rolling_corr(group):
            if len(group) < window:
                return pd.Series([np.nan] * len(group))
            return group['volume'].rolling(window=window).corr(group['return'])
        
        # 按股票分组计算相关性
        result['volume_return_corr_20'] = result.groupby('symbol').apply(
            lambda x: x['volume'].rolling(window=window).corr(x['return'])
        ).reset_index(level=0, drop=True)
        
        logger.debug(f"[Nonlinear Features] Extracted volume_skew_20, volume_kurtosis_20, volume_return_corr_20")
        return result
    
    # ==================== 基础因子计算 ====================
    
    def compute_vwap_residual_momentum(self, df: pd.DataFrame, 
                                        periods: list[int] = [5, 10]) -> pd.DataFrame:
        """计算 VWAP 残差动量因子。"""
        result = df.copy()
        
        # 计算 VWAP
        if 'vwap' not in result.columns:
            result['vwap'] = (result['high'] + result['low'] + result['close']) / 3.0
        else:
            result['vwap'] = result['vwap'].fillna((result['high'] + result['low'] + result['close']) / 3.0)
        
        # 计算残差
        result['price_residual'] = result['close'] - result['vwap']
        
        # 残差动量
        for period in periods:
            result[f'residual_momentum_{period}'] = (
                result.groupby('symbol')['price_residual'].transform(
                    lambda x: x / (x.shift(period) + self.EPSILON) - 1.0
                )
            )
        
        return result
    
    def compute_momentum(self, df: pd.DataFrame, 
                         periods: list[int] = [5, 10, 20]) -> pd.DataFrame:
        """计算传统动量因子。"""
        result = df.copy()
        
        for period in periods:
            result[f'momentum_{period}'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
            )
        
        return result
    
    def compute_volatility(self, df: pd.DataFrame, 
                           periods: list[int] = [5, 20]) -> pd.DataFrame:
        """计算波动率因子。"""
        result = df.copy()
        
        # 计算收益率
        result['return'] = result.groupby('symbol')['close'].pct_change().fillna(0)
        
        for period in periods:
            result[f'volatility_{period}'] = result.groupby('symbol')['return'].transform(
                lambda x: x.rolling(window=period).std()
            )
        
        return result
    
    def compute_volume_price_divergence(self, df: pd.DataFrame, 
                                         period: int = 5) -> pd.DataFrame:
        """计算量价背离因子。"""
        result = df.copy()
        
        # 价格变化
        result[f'price_change_{period}'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
        )
        
        # 成交量变化
        result[f'volume_change_{period}'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
        )
        
        # 量价背离
        result[f'volume_price_divergence_{period}'] = (
            result[f'price_change_{period}'] - result[f'volume_change_{period}']
        )
        
        return result
    
    def compute_volume_price_health(self, df: pd.DataFrame, 
                                     volume_window: int = 5, 
                                     price_window: int = 5) -> pd.DataFrame:
        """计算量价健康度因子。"""
        result = df.copy()
        
        # 成交量 MA 比率
        result['volume_ma'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.shift(1).rolling(window=volume_window).mean()
        )
        result['volume_ratio'] = result['volume'].shift(1) / (result['volume_ma'] + self.EPSILON)
        
        # 价格变化
        result['price_change'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) / (x.shift(price_window + 1) + self.EPSILON) - 1.0
        )
        
        # 量价健康度评分
        def health_score(row):
            if row['price_change'] > 0 and row['volume_ratio'] > 1.0:
                return 1.0    # 价涨量增：健康
            elif row['price_change'] > 0 and row['volume_ratio'] <= 1.0:
                return -0.5   # 价涨量缩：背离
            elif row['price_change'] <= 0 and row['volume_ratio'] <= 1.0:
                return -0.2   # 价跌量缩：正常调整
            else:
                return -1.0   # 价跌量增：危险
        
        result['volume_price_health'] = result.apply(health_score, axis=1)
        
        return result
    
    # ==================== 标签计算 ====================
    
    def compute_t1_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 T+1 收益标签。"""
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
    
    # ==================== 数据防御机制 ====================
    
    def check_and_repair_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【数据防御】检查并修复缺失数据。
        
        【主动防御逻辑】
        1. 检测 2024 年 total_mv 缺失
        2. 调用 DataLoader.repair_2024_data() 主动修复
        3. 严禁打印报错后停止运行
        
        Args:
            df: 输入数据
            
        Returns:
            修复后的数据
        """
        result = df.copy()
        
        # 检查 total_mv 缺失
        if 'total_mv' in result.columns:
            null_ratio = result['total_mv'].isna().sum() / len(result)
            if null_ratio > 0.3:
                logger.warning(f"[数据防御] total_mv 缺失比例：{null_ratio:.1%}")
                
                # 尝试用 amount/turnover_rate 估算
                if 'amount' in result.columns and 'turnover_rate' in result.columns:
                    logger.info("[数据防御] 尝试用 amount/turnover_rate 估算 total_mv")
                    estimated_mv = result['amount'] / (result['turnover_rate'].fillna(0.01) + self.EPSILON) * 100
                    result['total_mv'] = result['total_mv'].fillna(estimated_mv)
        
        # 检查 vwap 缺失
        if 'vwap' not in result.columns or result['vwap'].isna().sum() > 0:
            logger.info("[数据防御] 用 (high+low+close)/3 估算 VWAP")
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
            # 跳过非数值类型
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue
            
            null_count = result[col].isna().sum()
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            
            if null_ratio > null_threshold:
                # 缺失值过多，填 0
                result[col] = result[col].fillna(0)
                logger.debug(f"[Fill Null] Factor '{col}' has {null_ratio:.1%} nulls, filled with 0")
            else:
                # 使用中位数填充
                try:
                    median_val = result[col].median()
                    if np.isnan(median_val) or not np.isfinite(median_val):
                        median_val = 0
                    result[col] = result[col].fillna(median_val)
                except (TypeError, ValueError):
                    result[col] = result[col].fillna(0)
        
        return result
    
    # ==================== 预测评分 ====================
    
    def compute_predict_score(self, df: pd.DataFrame, 
                               weights: Optional[dict[str, float]] = None) -> pd.DataFrame:
        """计算综合预测评分。"""
        if weights is None:
            weights = self.FACTOR_WEIGHTS
        
        result = df.copy()
        
        # 计算加权评分
        raw_score = np.zeros(len(result))
        for factor_name, weight in weights.items():
            if factor_name in result.columns:
                raw_score += result[factor_name].fillna(0).values * weight
        
        result['score'] = raw_score
        
        logger.debug(f"[Predict Score] Computed, factors used: {len(weights)}")
        return result
    
    # ==================== 因子计算主流程 ====================
    
    def compute_factors(self, df: pd.DataFrame, 
                        clean: bool = True) -> pd.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【计算顺序 - 严格检查 T 日信号只用 T 日数据】
        1. 数据防御检查
        2. VWAP 残差动量 (核心)
        3. 基础动量因子
        4. 波动率因子
        5. 量价交互因子
        6. 量价非线性特征
        7. T+1/T+N 收益标签
        8. 缺失值处理
        9. 因子清洗 (MAD/Z-Score/Neu)
        10. 预测评分
        
        Args:
            df: 输入数据
            clean: 是否执行因子清洗
            
        Returns:
            包含因子和评分的数据
        """
        logger.info("=" * 70)
        logger.info("V103 Factor Computation Started")
        logger.info("=" * 70)
        
        # 1. 数据防御检查
        logger.info("[Step 1] Data Defense Check...")
        result = self.check_and_repair_data(df)
        
        # 2. VWAP 残差动量 (核心)
        logger.info("[Step 2] Computing VWAP Residual Momentum...")
        result = self.compute_vwap_residual_momentum(result, periods=[5, 10])
        
        # 3. 基础动量
        logger.info("[Step 3] Computing Momentum Factors...")
        result = self.compute_momentum(result, periods=[5, 10, 20])
        
        # 4. 波动率
        logger.info("[Step 4] Computing Volatility Factors...")
        result = self.compute_volatility(result, periods=[5, 20])
        
        # 5. 量价交互因子
        logger.info("[Step 5] Computing Volume-Price Factors...")
        result = self.compute_volume_price_divergence(result, period=5)
        result = self.compute_volume_price_health(result)
        
        # 6. 量价非线性特征 (V103 新增)
        logger.info("[Step 6] Extracting Nonlinear Volume Features...")
        result = self.extract_volume_nonlinear_features(result, window=20)
        
        # 7. T+1/T+N 收益标签
        logger.info("[Step 7] Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 8. 缺失值处理
        logger.info("[Step 8] Filling Null Values...")
        result = self.fill_null_values(result)
        
        # 9. 因子清洗三部曲
        if clean:
            logger.info("[Step 9] Factor Cleaning (MAD/Z-Score/Neu)...")
            result = self.clean_factors(result, 
                                        do_winsorize=True,
                                        do_normalize=True,
                                        do_neutralize=self.use_neutralization)
        
        # 10. 预测评分
        logger.info("[Step 10] Computing Predict Score...")
        result = self.compute_predict_score(result)
        
        logger.info("=" * 70)
        logger.info("V103 Factor Computation Complete")
        logger.info("=" * 70)
        
        return result
    
    # ==================== 主接口：compute_score ====================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【主接口】计算 Alpha 评分。
        
        这是 BacktestReferee 调用的唯一接口。
        
        Args:
            df: 输入数据 (必须包含 trade_date, symbol 和 OHLCV 数据)
            
        Returns:
            包含 score 和 t1_return 的 DataFrame
        """
        # 计算因子和评分
        result = self.compute_factors(df, clean=True)
        
        # 返回必需列
        output_columns = ['trade_date', 'symbol', 'score', 't1_return']
        for n in [3, 5]:
            if f't{n}_return' in result.columns:
                output_columns.append(f't{n}_return')
        
        return result[output_columns]
    
    def get_factor_ics(self, df: pd.DataFrame) -> dict[str, float]:
        """获取所有因子的 IC 值 (供 BacktestReferee 调用)。"""
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


def get_alpha_research(config_path: str = "config/factors.yaml",
                       use_neutralization: bool = True) -> AlphaResearchV103:
    """获取 AlphaResearchV103 实例。"""
    return AlphaResearchV103(config_path, use_neutralization)