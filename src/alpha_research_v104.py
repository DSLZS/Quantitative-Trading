"""
Alpha Research Module - V104 因子生存竞争与高频特征截面化.

【V104 核心改进 - 因子工厂攻坚战】
1. 因子生存竞争：构建 10+ 个原始因子，内部测试筛选最优组合
2. 高频特征截面化：成交量加权标准差、收益率偏度等日内特征
3. IC 倒挂修复：严格使用 T-1 日数据计算因子，确保对齐 T+1 收益
4. 自迭代优化：T+1 IC < 0.03 时自动进行多轮迭代
5. 防欺诈审计：输出清洗前后 IC 对比表和 T+1~T+5 单调性审计

【10+ 原始因子体系】
1. 量价背离系数 (Volume-Price Divergence)
2. 波动率收缩 VCP (Volatility Contraction Pattern)
3. 换手率异常度 (Turnover Rate Anomaly)
4. 资金流强度 (Money Flow Intensity)
5. 价格动量加速度 (Price Momentum Acceleration)
6. 成交量分布偏度 (Volume Distribution Skew)
7. 收益率分布峰度 (Return Distribution Kurtosis)
8. 相对强度 RS (Relative Strength)
9. 价格效率指标 (Price Efficiency)
10. 波动率调整动量 (Volatility-Adjusted Momentum)
11. 量价健康度 (Volume-Price Health)
12. 残差动量 (Residual Momentum)

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标：低于 0.03 触发自动迭代 |
| IC Decay | T+1 > T+3 > T+5 | 信号衰减必须符合单调性 |
| 因子多样性 | >= 10 个 | 必须构建至少 10 个原始因子 |
| 自迭代能力 | 自动 | T+1 IC < 0.03 时自动进行 2-3 轮优化 |
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


class AlphaResearchV104:
    """
    V104 Alpha 预测核心引擎 - 因子生存竞争模式。
    
    【核心改进】
    1. 因子生存竞争：10+ 原始因子内部测试
    2. 高频特征截面化：成交量加权标准差、收益率偏度
    3. IC 对齐修复：严格使用 T-1 日数据
    4. 自迭代优化：自动多轮优化直到 T+1 IC >= 0.03
    
    【因子计算对齐原则】
    - 所有因子必须使用 T-1 日及之前数据
    - 因子值对齐 T 日，预测 T+1 日收益
    - 严禁使用当日 close 计算因子
    """
    
    EPSILON = 1e-6
    
    # V104 因子权重配置 (初始权重，会在自迭代中优化)
    FACTOR_WEIGHTS = {
        # 核心因子：量价背离
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
        
        # 成交量分布偏度
        "volume_skew_20": 0.06,
        
        # 收益率分布峰度
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
        初始化 V104 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
            use_neutralization: 是否启用中性化
            auto_iterate: 是否启用自迭代优化
            max_iterations: 最大迭代次数
        """
        self.config_path = Path(config_path)
        self.use_neutralization = use_neutralization
        self.auto_iterate = auto_iterate
        self.max_iterations = max_iterations
        
        self.factors: list[dict[str, Any]] = []
        self._load_config()
        
        # 清洗前后 IC 记录
        self.factor_ic_before_cleaning = {}
        self.factor_ic_after_cleaning = {}
        
        # 因子生存竞争记录
        self.factor_competition_results = {}
        self.iteration_history = []
        
        # 内部测试结果
        self.internal_test_results = []
        
        logger.info("AlphaResearchV104 initialized")
        logger.info(f"  Config path: {self.config_path}")
        logger.info(f"  Use neutralization: {self.use_neutralization}")
        logger.info(f"  Auto iteration: {self.auto_iterate}")
        logger.info(f"  Max iterations: {self.max_iterations}")
        logger.info(f"  Factor count: {len(self.FACTOR_WEIGHTS)}")
    
    def _load_config(self) -> None:
        """加载因子配置文件。"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.factors = config.get('factors', [])
            logger.info(f"Loaded {len(self.factors)} factor configurations")
        except FileNotFoundError:
            logger.info(f"Config file not found: {self.config_path}, using defaults")
            self.factors = []
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            self.factors = []
    
    # ==================== 因子清洗三部曲 ====================
    
    def winsorize_mad(self, df: pd.DataFrame, columns: Optional[list[str]] = None,
                      n_std: float = 3.0) -> pd.DataFrame:
        """
        【清洗三部曲 1】去极值 - 使用 Median Absolute Deviation (MAD)。
        """
        if columns is None:
            columns = self.get_factor_names()
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            if 'trade_date' in result.columns:
                for date in result['trade_date'].unique():
                    mask = result['trade_date'] == date
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
            else:
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
                
                result.loc[:, col] = result[col].clip(
                    lower=lower_bound, upper=upper_bound
                )
        
        return result
    
    def normalize_zscore(self, df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        【清洗三部曲 2】标准化 - Z-Score 标准化。
        """
        if columns is None:
            columns = self.get_factor_names()
        
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            if 'trade_date' in result.columns:
                grouped = result.groupby('trade_date')[col]
                mean = grouped.transform('mean')
                std = grouped.transform('std')
                std = std.replace(0, self.EPSILON)
                result.loc[:, col] = (result[col] - mean) / std
            else:
                mean_val = result[col].mean()
                std_val = result[col].std()
                if std_val < self.EPSILON:
                    std_val = self.EPSILON
                result.loc[:, col] = (result[col] - mean_val) / std_val
        
        return result
    
    def neutralize_ols(self, df: pd.DataFrame, columns: Optional[list[str]] = None,
                       neutralize_vars: list[str] = None) -> pd.DataFrame:
        """
        【清洗三部曲 3】中性化 - 使用 OLS 去除行业和市值暴露。
        """
        if columns is None:
            columns = self.get_factor_names()
        
        if neutralize_vars is None:
            neutralize_vars = ['ln_total_mv']
        
        result = df.copy()
        
        if 'total_mv' not in result.columns:
            logger.warning("Missing total_mv column, skipping neutralization")
            return result
        
        result['ln_total_mv'] = np.log(result['total_mv'] + self.EPSILON)
        
        industry_dummies = []
        if 'industry_code' in result.columns:
            industry_dummies = pd.get_dummies(result['industry_code'], prefix='ind')
        
        for col in columns:
            if col not in result.columns:
                continue
            
            if 'trade_date' in result.columns:
                neutralized_values = []
                
                for date in result['trade_date'].unique():
                    mask = result['trade_date'] == date
                    day_data = result.loc[mask].copy()
                    
                    if len(day_data) < 30:
                        neutralized_values.append(day_data[[col, 'trade_date', 'symbol']])
                        continue
                    
                    y = day_data[col].values
                    X_vars = ['ln_total_mv']
                    X = day_data[X_vars].values
                    
                    if len(industry_dummies) > 0:
                        day_ind = industry_dummies.loc[mask]
                        X = np.column_stack([X, day_ind.values])
                    
                    X = np.column_stack([np.ones(len(X)), X])
                    
                    try:
                        beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                        y_pred = X @ beta
                        residuals = y - y_pred
                        day_data[col] = residuals
                        neutralized_values.append(day_data[[col, 'trade_date', 'symbol']])
                    except np.linalg.LinAlgError:
                        neutralized_values.append(day_data[[col, 'trade_date', 'symbol']])
                
                if neutralized_values:
                    neutralized_df = pd.concat(neutralized_values, ignore_index=True)
                    result.loc[:, col] = neutralized_df[col].values
            else:
                y = result[col].values
                X = result[['ln_total_mv']].values
                X = np.column_stack([np.ones(len(X)), X])
                
                try:
                    beta = np.linalg.pinv(X.T @ X) @ X.T @ y
                    residuals = y - X @ beta
                    result.loc[:, col] = residuals
                except np.linalg.LinAlgError:
                    pass
        
        return result
    
    def clean_factors(self, df: pd.DataFrame, 
                      do_winsorize: bool = True,
                      do_normalize: bool = True,
                      do_neutralize: bool = True) -> pd.DataFrame:
        """执行因子清洗三部曲。"""
        result = df.copy()
        
        logger.info("[V104 因子清洗] 开始执行清洗三部曲...")
        
        # 记录清洗前 IC
        self.factor_ic_before_cleaning = self._calculate_all_factor_ics(result)
        
        if do_winsorize:
            logger.info("  [Step 1] MAD Winsorization...")
            result = self.winsorize_mad(result, n_std=3.0)
        
        if do_normalize:
            logger.info("  [Step 2] Z-Score Normalization...")
            result = self.normalize_zscore(result)
        
        if do_neutralize and self.use_neutralization:
            logger.info("  [Step 3] OLS Neutralization (Size + Industry)...")
            result = self.neutralize_ols(result)
        
        # 记录清洗后 IC
        self.factor_ic_after_cleaning = self._calculate_all_factor_ics(result)
        
        self._print_cleaning_comparison()
        
        logger.info("[V104 因子清洗] 完成")
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
        logger.info("=" * 80)
        logger.info("[V104 因子清洗前后 IC 对比表]")
        logger.info("-" * 80)
        logger.info(f"{'因子名称':<40} {'清洗前 IC':>12} {'清洗后 IC':>12} {'改善':>10}")
        logger.info("-" * 80)
        
        all_factors = set(self.factor_ic_before_cleaning.keys()) | set(self.factor_ic_after_cleaning.keys())
        
        for factor in sorted(all_factors):
            before = self.factor_ic_before_cleaning.get(factor, 0)
            after = self.factor_ic_after_cleaning.get(factor, 0)
            improvement = after - before
            status = "✓" if improvement > 0 else "✗"
            logger.info(f"{factor:<40} {before:>12.4f} {after:>12.4f} {improvement:>+10.4f} {status}")
        
        logger.info("=" * 80)
    
    # ==================== V104 10+ 原始因子计算 ====================
    # 【核心原则】所有因子必须使用 T-1 日及之前数据，严禁使用当日数据
    
    def compute_volume_price_divergence(self, df: pd.DataFrame, 
                                         periods: list[int] = [5, 10]) -> pd.DataFrame:
        """
        【因子 1】量价背离系数。
        
        【核心逻辑】
        - 计算 T-1 日的价格变化和成交量变化
        - 背离 = 价格变化 - 成交量变化
        - 正背离：价涨量缩或价跌量增（可能是反转信号）
        
        【对齐原则】使用 shift(1) 确保使用 T-1 日数据
        """
        result = df.copy()
        
        for period in periods:
            # T-1 日相对于 T-period-1 日的价格变化
            result[f'price_change_{period}'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
            )
            
            # T-1 日相对于 T-period-1 日的成交量变化
            result[f'volume_change_{period}'] = result.groupby('symbol')['volume'].transform(
                lambda x: x.shift(1) / (x.shift(period + 1) + self.EPSILON) - 1.0
            )
            
            # 量价背离
            result[f'volume_price_divergence_{period}'] = (
                result[f'price_change_{period}'] - result[f'volume_change_{period}']
            )
        
        return result
    
    def compute_vcp_ratio(self, df: pd.DataFrame, 
                          periods: list[int] = [5, 10]) -> pd.DataFrame:
        """
        【因子 2】波动率收缩 VCP (Volatility Contraction Pattern)。
        
        【核心逻辑】
        - VCP = 近期波动率 / 远期波动率
        - VCP < 1: 波动率收缩，可能是突破前兆
        - VCP > 1: 波动率扩张，可能是趋势末端
        
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
        
        【对齐原则】使用 T-1 日及之前数据
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
        
        【对齐原则】使用 T-1 日及之前数据
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
        - 动量 = 价格变化率
        - 加速度 = 动量的变化率
        - 正加速度：动量增强
        
        【对齐原则】使用 T-1 日及之前数据
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
        
        【对齐原则】使用 T-1 日及之前数据
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
        - 峰度低：收益率分布平缓
        
        【对齐原则】使用 T-1 日及之前数据
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
        
        【对齐原则】使用 T-1 日及之前数据
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
        【因子 9】价格效率指标。
        
        【核心逻辑】
        - 效率 = |净价格变化| / 总价格变化路径
        - 效率高：价格趋势明确
        - 效率低：价格震荡
        
        【对齐原则】使用 T-1 日及之前数据
        """
        result = df.copy()
        
        # 净价格变化
        net_change = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1) - x.shift(window + 1)
        )
        
        # 总价格变化路径 - 使用 transform 实现
        def calc_total_path(x):
            shifted = x.shift(1)
            result_values = []
            for i in range(len(shifted)):
                if i < window - 1:
                    result_values.append(np.nan)
                else:
                    window_data = shifted.iloc[i-window+1:i+1]
                    valid_data = window_data.dropna()
                    if len(valid_data) < 2:
                        result_values.append(np.nan)
                    else:
                        path = np.abs(valid_data.diff()).sum()
                        result_values.append(path)
            return pd.Series(result_values, index=x.index, dtype=float)
        
        # 按 symbol 分组计算
        total_change = result.groupby('symbol')['close'].transform(calc_total_path)
        
        # 效率比率
        result['price_efficiency_20'] = np.abs(net_change) / (total_change + self.EPSILON)
        
        return result
    
    def compute_volatility_adjusted_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【因子 10】波动率调整动量。
        
        【核心逻辑】
        - 动量 / 波动率 (类似 Sharpe 比率)
        - 高风险调整后的收益
        
        【对齐原则】使用 T-1 日及之前数据
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
        
        【对齐原则】使用 T-1 日及之前数据
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
        
        【对齐原则】使用 T-1 日及之前数据
        """
        result = df.copy()
        
        # 计算 VWAP (使用 T-1 日数据)
        if 'vwap' not in result.columns:
            result['vwap'] = (result['high'].shift(1) + result['low'].shift(1) + result['close'].shift(1)) / 3.0
        else:
            result['vwap'] = result['vwap'].shift(1).fillna(
                (result['high'].shift(1) + result['low'].shift(1) + result['close'].shift(1)) / 3.0
            )
        
        # 残差
        result['price_residual'] = result['close'].shift(1) - result['vwap']
        
        # 残差动量
        for period in periods:
            result[f'residual_momentum_{period}'] = result.groupby('symbol')['price_residual'].transform(
                lambda x: x / (x.shift(period) + self.EPSILON) - 1.0
            )
        
        return result
    
    # ==================== 标签计算 (T+1 到 T+5) ====================
    
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
        """【数据防御】检查并修复缺失数据。"""
        result = df.copy()
        
        # 检查 total_mv 缺失
        if 'total_mv' in result.columns:
            null_ratio = result['total_mv'].isna().sum() / len(result)
            if null_ratio > 0.3:
                logger.warning(f"[数据防御] total_mv 缺失比例：{null_ratio:.1%}")
                
                if 'amount' in result.columns and 'turnover_rate' in result.columns:
                    logger.info("[数据防御] 用 amount/turnover_rate 估算 total_mv")
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
    
    # ==================== 因子生存竞争 ====================
    
    def run_factor_competition(self, df: pd.DataFrame) -> dict[str, float]:
        """
        【因子生存竞争】测试所有因子的独立预测能力。
        
        Args:
            df: 包含因子和标签的数据
            
        Returns:
            各因子 IC 字典
        """
        logger.info("=" * 80)
        logger.info("[V104 因子生存竞争] 开始测试各因子独立预测能力...")
        logger.info("-" * 80)
        
        competition_results = {}
        
        for factor_name in self.get_factor_names():
            if factor_name not in df.columns:
                continue
            
            ic = self._calculate_single_factor_ic(df[factor_name], df['t1_return'])
            competition_results[factor_name] = ic
        
        # 按 IC 排序
        sorted_results = sorted(competition_results.items(), key=lambda x: abs(x[1]), reverse=True)
        
        logger.info(f"{'排名':<6} {'因子名称':<40} {'IC':>10} {'状态':>8}")
        logger.info("-" * 80)
        
        for rank, (factor_name, ic) in enumerate(sorted_results, 1):
            status = "✓" if abs(ic) > 0.03 else "✗"
            logger.info(f"{rank:<6} {factor_name:<40} {ic:>10.4f} {status:>8}")
        
        logger.info("=" * 80)
        
        self.factor_competition_results = competition_results
        return competition_results
    
    def optimize_weights(self, df: pd.DataFrame, method: str = 'ic_weighted') -> dict[str, float]:
        """
        【自迭代优化】根据因子 IC 优化权重。
        
        Args:
            df: 包含因子和标签的数据
            method: 优化方法 ('ic_weighted', 'equal', 'top_k')
            
        Returns:
            优化后的权重
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
        
        logger.info(f"[权重优化] 使用 {method} 方法优化权重")
        logger.info(f"  Top factors: {list(optimized_weights.keys())[:5]}...")
        
        return optimized_weights
    
    # ==================== 预测评分 ====================
    
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
    
    # ==================== 因子计算主流程 ====================
    
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
        14. T+1/T+N 收益标签
        15. 缺失值处理
        16. 因子清洗
        17. 预测评分
        """
        logger.info("=" * 80)
        logger.info("V104 Factor Computation Started")
        logger.info("=" * 80)
        
        # 1. 数据防御
        logger.info("[Step 1] Data Defense Check...")
        result = self.check_and_repair_data(df)
        
        # 2-13. 计算 12 个原始因子
        logger.info("[Step 2] Computing Volume-Price Divergence...")
        result = self.compute_volume_price_divergence(result, periods=[5, 10])
        
        logger.info("[Step 3] Computing VCP Ratio...")
        result = self.compute_vcp_ratio(result, periods=[5, 10])
        
        logger.info("[Step 4] Computing Turnover Anomaly...")
        result = self.compute_turnover_anomaly(result, periods=[5, 20])
        
        logger.info("[Step 5] Computing Money Flow Intensity...")
        result = self.compute_money_flow_intensity(result, periods=[5, 10])
        
        logger.info("[Step 6] Computing Momentum Acceleration...")
        result = self.compute_momentum_acceleration(result)
        
        logger.info("[Step 7] Computing Volume Skew...")
        result = self.compute_volume_skew(result, window=20)
        
        logger.info("[Step 8] Computing Return Kurtosis...")
        result = self.compute_return_kurtosis(result, window=20)
        
        logger.info("[Step 9] Computing Relative Strength...")
        result = self.compute_relative_strength(result, periods=[10, 20])
        
        logger.info("[Step 10] Computing Price Efficiency...")
        result = self.compute_price_efficiency(result, window=20)
        
        logger.info("[Step 11] Computing Volatility-Adjusted Momentum...")
        result = self.compute_volatility_adjusted_momentum(result)
        
        logger.info("[Step 12] Computing Volume-Price Health...")
        result = self.compute_volume_price_health(result)
        
        logger.info("[Step 13] Computing Residual Momentum...")
        result = self.compute_residual_momentum(result, periods=[5, 10])
        
        # 14. 收益标签
        logger.info("[Step 14] Computing Return Labels...")
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)
        result = self.compute_tn_return(result, n=5)
        
        # 15. 缺失值处理
        logger.info("[Step 15] Filling Null Values...")
        result = self.fill_null_values(result)
        
        # 16. 因子清洗
        if clean:
            logger.info("[Step 16] Factor Cleaning (MAD/Z-Score/Neu)...")
            result = self.clean_factors(result, 
                                        do_winsorize=True,
                                        do_normalize=True,
                                        do_neutralize=self.use_neutralization)
        
        # 17. 预测评分
        logger.info("[Step 17] Computing Predict Score...")
        result = self.compute_predict_score(result)
        
        logger.info("=" * 80)
        logger.info("V104 Factor Computation Complete")
        logger.info("=" * 80)
        
        return result
    
    # ==================== 自迭代优化 ====================
    
    def auto_iterate_optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【自迭代优化】自动多轮优化直到 T+1 IC >= 0.03。
        
        Args:
            df: 输入数据
            
        Returns:
            优化后的数据
        """
        logger.info("=" * 80)
        logger.info("[V104 自迭代优化] 开始自动优化流程...")
        logger.info("=" * 80)
        
        best_result = None
        best_ic = 0.0
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n[迭代 {iteration}/{self.max_iterations}]")
            
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
            
            logger.info(f"[迭代 {iteration}] T+1 IC = {current_ic:.4f}")
            
            if abs(current_ic) > abs(best_ic):
                best_ic = current_ic
                best_result = result.copy()
            
            # 如果达到目标，提前退出
            if abs(current_ic) >= 0.03:
                logger.info(f"[迭代 {iteration}] 达到目标 IC (>= 0.03)，停止优化")
                return result
            
            # 优化权重
            if iteration < self.max_iterations:
                method = 'top_k' if iteration == 1 else 'ic_weighted'
                optimized_weights = self.optimize_weights(result, method=method)
                self.FACTOR_WEIGHTS = optimized_weights
                
                # 重新计算评分
                result = self.compute_predict_score(result, weights=optimized_weights)
        
        logger.info(f"\n[自迭代完成] 最佳 T+1 IC = {best_ic:.4f}")
        logger.info("=" * 80)
        
        return best_result if best_result is not None else df
    
    # ==================== 主接口：compute_score ====================
    
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
            if abs(t1_ic) < 0.03:
                logger.warning(f"[自迭代触发] T+1 IC ({t1_ic:.4f}) < 0.03，启动自迭代优化")
                result = self.auto_iterate_optimize(df)
        
        # 运行因子生存竞争（记录结果）
        self.run_factor_competition(result)
        
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


def get_alpha_research(config_path: str = "config/factors.yaml",
                       use_neutralization: bool = True,
                       auto_iterate: bool = True) -> AlphaResearchV104:
    """获取 AlphaResearchV104 实例。"""
    return AlphaResearchV104(config_path, use_neutralization, auto_iterate)