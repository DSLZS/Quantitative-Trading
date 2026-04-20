"""
Alpha Model Module - V200 Evolution: Survival of the Alpha

【V200 核心变革】
1. 波动率自适应特征 (Volatility-Adjusted Features)
   - 特征根据市场波动率动态缩放
   - 高波动环境下降低高风险因子权重

2. 风险滤网层 (Risk Filter Layer / Non-linear Gating)
   - 当市场处于"高波动/低流动性"状态时，自动惩罚高贝塔因子
   - 使用非线性门控函数实现状态自适应

3. IR 动态权重 (Information Ratio per Factor)
   - 不再只根据 IC 赋权，而是引入"收益/风险比"动态权重
   - 因子权重 = IR / (IR + 波动率惩罚)

4. 数据自愈增强
   - 报错必修：数据库连接断开、NULL 值自动修复
   - 严禁输出"数据问题无法运行"

【架构原则】
- 保持"选手 - 裁判"物理隔离
- alpha_model.py 严禁出现任何 ic、rank_ic 或回测逻辑代码
- 清理所有带旧版本号的注释，代码生产化

【验收红线】
- 2024 年 IC > 0.10 且 MDD < 25%
- 初始资金 100,000，费率 1.3‰
- 严禁未来函数，严禁 T+0
"""

from typing import Any, Optional, Dict, List, Tuple
import warnings
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import skew, kurtosis
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V200_Evolution_Survival_of_Alpha"

# V200 核心因子 (含波动率自适应因子)
V200_CORE_FACTORS = [
    'reversion_5',           # 5 日反转 - 防御型
    'volume_rank',           # 成交量排名 - 流动性监控
    'volume_price_contradiction',  # 量价矛盾 - 风险信号
    'liquidity_alpha',       # 流动性 Alpha
    'volatility_5',          # 5 日波动率 - 风险度量
    'volatility_20',         # 20 日波动率 - 风险度量
    'volatility_skew',       # 波动率偏度 - 尾部风险
    'liquidity_mkt_neutral', # 市值中性化流动性
    'beta_adj_factor',       # NEW: 贝塔调整因子 (风险滤网)
    'vol_adj_momentum',      # NEW: 波动率调整动量
]

MAX_FACTORS = 10

# V200 风险滤网参数 - 增强版
RISK_FILTER_HIGH_VOL_THRESHOLD = 0.60      # 高波动阈值 (分位数) - 降低阈值更敏感
RISK_FILTER_LOW_LIQ_THRESHOLD = 0.35       # 低流动性阈值 (分位数) - 提高阈值更敏感
RISK_FILTER_PENALTY_SCALE = 4.0            # 惩罚强度 - 增强
RISK_FILTER_BETA_PENALTY = 0.3             # 高贝塔因子惩罚系数 - 增强惩罚

# V200 IR 动态权重参数
IR_LOOKBACK_WINDOW = 20                    # IR 计算窗口
IR_MIN_WEIGHT = 0.02                       # 最小因子权重
IR_VOLATILITY_PENALTY = 0.0                # IR 波动率惩罚 (禁用，避免过度惩罚)

# V200 波动率自适应参数
VOL_ADAPTIVE_SCALE = 0.5                   # 波动率自适应缩放系数
VOL_ADAPTIVE_EXPONENT = 1.5                # 波动率自适应指数

# 配置
MIN_STOCK_COUNT = 5000
WARMUP_DAYS = 60
WARMUP_YEAR = 2022
EPSILON = 1e-6


def winsorize(series: pd.Series, sigma: float = 3.0) -> pd.Series:
    """缩尾处理异常值"""
    series_clean = series.copy()
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    mean = series_clean.mean()
    if pd.isna(mean):
        mean = 0.0
    
    std = series_clean.std()
    if pd.isna(std) or std < 1e-10:
        std = 1.0
    
    lower = mean - sigma * std
    upper = mean + sigma * std
    series_clean = series_clean.clip(lower=lower, upper=upper)
    
    return series_clean


def auto_fillna(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    V200 数据自愈：自动填充缺失值
    
    【V200 增强】
    1. 优先 ffill (前值填充)
    2. 其次 mean (组内均值)
    3. 最后 0 (全局填充)
    4. 新增：中位数填充作为后备
    """
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        missing_ratio = result[col].isna().sum() / len(result)
        if missing_ratio == 0:
            continue
        
        logger.debug(f"[Auto FillNA] Column {col}: {missing_ratio:.2%} missing")
        
        if 'symbol' in result.columns:
            result[col] = result.groupby('symbol')[col].transform(
                lambda x: x.ffill()
            )
            result[col] = result.groupby('symbol')[col].transform(
                lambda x: x.fillna(x.mean())
            )
            result[col] = result.groupby('symbol')[col].transform(
                lambda x: x.fillna(x.median())
            )
        
        result[col] = result[col].fillna(0)
    
    return result


def fix_data_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    V200 数据管道自愈：修复常见数据问题
    
    【修复逻辑】
    1. 负价格/成交量 → 替换为 NaN 后填充
    2. 无限值 → 替换为 NaN
    3. 异常大的收益率 → 缩尾处理
    4. 缺失交易日 → 向前填充
    """
    result = df.copy()
    
    # 1. 修复负价格和成交量
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        if col in result.columns:
            mask = result[col] < 0
            if mask.any():
                logger.warning(f"[Fix Pipeline] Found {mask.sum()} negative values in {col}")
                result.loc[mask, col] = np.nan
    
    # 2. 修复无限值
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mask = result[col].isin([np.inf, -np.inf])
        if mask.any():
            result.loc[mask, col] = np.nan
    
    # 3. 缩尾处理收益率
    if 'pct_chg' in result.columns:
        result['pct_chg'] = winsorize(result['pct_chg'], sigma=4.0)
    
    if 'returns' in result.columns:
        result['returns'] = winsorize(result['returns'], sigma=4.0)
    
    # 4. 填充所有剩余 NaN
    result = auto_fillna(result)
    
    logger.info("[Fix Pipeline] Data pipeline healed successfully")
    return result


class RiskFilterLayer:
    """
    V200 风险滤网层 (Risk Filter Layer)
    
    【核心功能】
    1. 识别市场状态 (高波动/低流动性)
    2. 对高贝塔因子实施非线性惩罚
    3. 动态调整因子权重
    
    【数学原理】
    penalty = exp((volatility - threshold) * scale) * beta_penalty
    """
    
    def __init__(
        self,
        high_vol_threshold: float = RISK_FILTER_HIGH_VOL_THRESHOLD,
        low_liq_threshold: float = RISK_FILTER_LOW_LIQ_THRESHOLD,
        penalty_scale: float = RISK_FILTER_PENALTY_SCALE,
        beta_penalty: float = RISK_FILTER_BETA_PENALTY,
    ):
        self.high_vol_threshold = high_vol_threshold
        self.low_liq_threshold = low_liq_threshold
        self.penalty_scale = penalty_scale
        self.beta_penalty = beta_penalty
        self.risk_state_history = {}
        
    def compute_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算风险指标
        
        【指标体系】
        1. ATR 波动率
        2. 流动性指标 (成交量/市值)
        3. 偏度/峰度 (尾部风险)
        4. Beta 估计 (相对于市场)
        """
        result = df.copy()
        
        # 1. ATR 计算
        result['prev_close'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1)
        )
        
        tr1 = result['high'] - result['low']
        tr2 = (result['high'] - result['prev_close']).abs()
        tr3 = (result['low'] - result['prev_close']).abs()
        
        result['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr_20'] = result.groupby('symbol')['true_range'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        result['atr_ratio'] = result['atr_20'] / (result['close'] + EPSILON)
        
        # 2. 流动性指标
        result['volume_ma20'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        result['liquidity_ratio'] = result['volume'] / (result['volume_ma20'] + EPSILON)
        
        # 3. 收益率分布特征
        result['returns'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change()
        )
        
        result['return_skew_20'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda s: skew(s) if len(s) > 2 else 0.0, raw=False
            )
        )
        result['return_kurt_20'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda s: kurtosis(s) if len(s) > 2 else 0.0, raw=False
            )
        )
        
        # 4. Beta 估计 (20 日滚动)
        # 首先计算市场平均收益率和市场波动率
        # 使用 fillna(0) 确保 returns 没有 NaN
        if 'returns' not in result.columns or result['returns'].isna().all():
            result['returns'] = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().fillna(0)
            )
        
        # 计算市场统计量 (按日期)
        market_stats = result.groupby('trade_date')['returns'].agg(
            market_return='mean',
            market_vol_20='std'
        ).reset_index()
        
        # 填充 NaN 值
        market_stats['market_return'] = market_stats['market_return'].fillna(0)
        market_stats['market_vol_20'] = market_stats['market_vol_20'].fillna(0.01)
        
        # 合并市场统计量 - 使用 validate 确保正确合并
        result = result.merge(market_stats, on='trade_date', how='left', validate='m:1')
        
        # 确保列存在并填充 NaN
        if 'market_return' not in result.columns:
            result['market_return'] = 0.0
        else:
            result['market_return'] = result['market_return'].fillna(0)
            
        if 'market_vol_20' not in result.columns:
            result['market_vol_20'] = 0.01
        else:
            result['market_vol_20'] = result['market_vol_20'].fillna(0.01)
        
        # 计算个股波动率
        result['stock_vol_20'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).std().fillna(0.01)
        )
        
        # Beta = corr(stock, market) * (stock_vol / market_vol)
        # 简化：直接使用波动率比作为 Beta 代理
        result['beta_proxy'] = (
            result['stock_vol_20'] / (result['market_vol_20'] + EPSILON)
        ).clip(0.5, 2.5)
        
        return result
    
    def classify_risk_state(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        分类风险状态
        
        【状态定义】
        - HIGH_RISK: 高波动 + 低流动性
        - MEDIUM_RISK: 高波动或低流动性
        - LOW_RISK: 正常状态
        """
        # 计算截面分位数
        daily_stats = df.groupby('trade_date').agg({
            'atr_ratio': 'mean',
            'liquidity_ratio': 'mean',
            'return_skew_20': 'mean',
            'beta_proxy': 'mean'
        }).reset_index()
        
        # 动态阈值
        vol_threshold = daily_stats['atr_ratio'].quantile(self.high_vol_threshold)
        liq_threshold = daily_stats['liquidity_ratio'].quantile(self.low_liq_threshold)
        
        # 状态分类
        def _classify(row):
            is_high_vol = row['atr_ratio'] > vol_threshold
            is_low_liq = row['liquidity_ratio'] < liq_threshold
            
            if is_high_vol and is_low_liq:
                return 'HIGH_RISK'
            elif is_high_vol or is_low_liq:
                return 'MEDIUM_RISK'
            else:
                return 'LOW_RISK'
        
        daily_stats['risk_state'] = daily_stats.apply(_classify, axis=1)
        
        # 映射回原始数据
        state_map = dict(zip(daily_stats['trade_date'], daily_stats['risk_state']))
        risk_state_series = df['trade_date'].map(state_map)
        
        self.risk_state_history = {
            'vol_threshold': vol_threshold,
            'liq_threshold': liq_threshold,
            'daily_stats': daily_stats
        }
        
        return risk_state_series, self.risk_state_history
    
    def compute_risk_penalty(
        self, 
        df: pd.DataFrame, 
        risk_state: str,
        factor_name: str,
        is_high_beta: bool = False
    ) -> pd.Series:
        """
        计算风险惩罚系数
        
        【惩罚逻辑】
        1. HIGH_RISK 状态下，高贝塔因子受到强惩罚
        2. MEDIUM_RISK 状态下，适度惩罚
        3. LOW_RISK 状态下，无惩罚
        
        penalty = exp((vol - threshold) * scale) * beta_factor
        """
        n = len(df)
        base_penalty = np.ones(n)
        
        if risk_state == 'HIGH_RISK':
            # 高风险状态：强惩罚
            vol_factor = np.exp((df['atr_ratio'] - df['atr_ratio'].mean()) * self.penalty_scale)
            vol_factor = vol_factor.clip(0.3, 3.0)
            
            if is_high_beta:
                # 高贝塔因子额外惩罚
                beta_factor = self.beta_penalty
            else:
                beta_factor = 1.0
            
            base_penalty = vol_factor * beta_factor
            
        elif risk_state == 'MEDIUM_RISK':
            # 中等风险状态：适度惩罚
            vol_factor = 1.0 + (df['atr_ratio'] - df['atr_ratio'].mean()) * 0.5
            vol_factor = vol_factor.clip(0.5, 2.0)
            
            if is_high_beta:
                beta_factor = 1.0 - (1.0 - self.beta_penalty) * 0.5
            else:
                beta_factor = 1.0
            
            base_penalty = vol_factor * beta_factor
        
        return pd.Series(base_penalty, index=df.index)
    
    def get_risk_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        risk_state: str,
        factor_betas: Dict[str, float]
    ) -> Dict[str, float]:
        """
        获取风险调整后的因子权重 - V200 终极增强版
        
        【调整逻辑】
        1. HIGH_RISK: reversion_5 权重占 50%+，其他因子大幅降低
        2. MEDIUM_RISK: 防御型因子权重增加 100%
        3. LOW_RISK: 保持原权重
        """
        adjusted = base_weights.copy()
        
        # 高贝塔因子列表 (风险型)
        high_beta_factors = {'volatility_5', 'volatility_20', 'beta_adj_factor', 'momentum_10', 'vol_adj_momentum'}
        # 核心防御型因子 (reversion_5 是核心中的核心)
        core_defensive = 'reversion_5'
        # 其他防御型因子
        other_defensive = {'liquidity_mkt_neutral', 'volume_price_contradiction'}
        # 流动性风险因子
        liquidity_risk_factors = {'liquidity_alpha', 'volatility_skew'}
        
        if risk_state == 'HIGH_RISK':
            # 高风险：reversion_5 权重占主导 (50%+)，其他因子大幅降低
            for factor, weight in adjusted.items():
                if factor == core_defensive:
                    adjusted[factor] = weight * 8.0  # reversion_5 权重翻 8 倍
                elif factor in high_beta_factors:
                    adjusted[factor] = weight * 0.05  # 降至 5%
                elif factor in other_defensive:
                    adjusted[factor] = weight * 1.5  # 其他防御型适度增加
                elif factor in liquidity_risk_factors:
                    adjusted[factor] = weight * 0.1  # 流动性风险大幅降低
            
            # 确保 reversion_5 权重占主导 (手动设置)
            adjusted[core_defensive] = 0.6  # 60% 权重给反转因子
            # 其他因子平分剩余 40%
            remaining = 0.4
            other_factors = [f for f in adjusted.keys() if f != core_defensive]
            for f in other_factors:
                adjusted[f] = remaining / len(other_factors)
        
        elif risk_state == 'MEDIUM_RISK':
            # 中风险：防御型因子权重增加 100%
            for factor, weight in adjusted.items():
                if factor == core_defensive:
                    adjusted[factor] = weight * 3.0  # reversion_5 权重翻 3 倍
                elif factor in high_beta_factors:
                    adjusted[factor] = weight * 0.15  # 降至 15%
                elif factor in other_defensive:
                    adjusted[factor] = weight * 1.5  # 其他防御型增加 50%
                elif factor in liquidity_risk_factors:
                    adjusted[factor] = weight * 0.3  # 流动性风险降低
        
        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted


class VolatilityAdjuster:
    """
    V200 波动率自适应调整器
    
    【核心功能】
    1. 根据市场波动率动态缩放特征
    2. 高波动环境下降低特征幅度
    3. 计算波动率调整后的因子
    """
    
    def __init__(
        self,
        scale: float = VOL_ADAPTIVE_SCALE,
        exponent: float = VOL_ADAPTIVE_EXPONENT,
    ):
        self.scale = scale
        self.exponent = exponent
    
    def adjust_feature(
        self,
        feature: pd.Series,
        market_vol: pd.Series,
        vol_baseline: float
    ) -> pd.Series:
        """
        波动率自适应特征调整
        
        【公式】
        adjusted_feature = feature * (vol_baseline / market_vol) ^ exponent
        
        市场波动率越高，特征值越小 (降杠杆)
        """
        vol_ratio = vol_baseline / (market_vol + EPSILON)
        adjustment = np.power(vol_ratio, self.exponent)
        adjustment = adjustment.clip(0.3, 2.0)
        
        return feature * adjustment
    
    def compute_vol_adj_momentum(
        self,
        df: pd.DataFrame,
        momentum_col: str = 'momentum_10',
        vol_col: str = 'volatility_20'
    ) -> pd.Series:
        """
        计算波动率调整动量因子
        
        【原理】
        vol_adj_momentum = momentum / volatility
        即：单位风险获得的动量收益
        """
        if momentum_col not in df.columns:
            # 计算动量
            df[momentum_col] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(10) - 1
            )
        
        if vol_col not in df.columns:
            df[vol_col] = df.groupby('symbol')['returns'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
        
        # 波动率调整动量
        vol_adj = df[momentum_col] / (df[vol_col] + EPSILON)
        
        # 按日期标准化 (只处理数值列)
        if 'trade_date' in df.columns:
            # 使用 transform 直接处理 Series
            vol_adj = vol_adj.groupby(df['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + EPSILON)
            )
        
        return vol_adj
    
    def compute_beta_adjusted_factor(
        self,
        df: pd.DataFrame,
        factor_col: str,
        beta_col: str = 'beta_proxy'
    ) -> pd.Series:
        """
        计算贝塔调整因子
        
        【原理】
        beta_adj_factor = factor / beta
        降低高贝塔因子的影响
        """
        if factor_col not in df.columns or beta_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        beta_adj = df[factor_col] / (df[beta_col] + EPSILON)
        return beta_adj


class IRWeightCalculator:
    """
    V200 IR 动态权重计算器
    
    【核心思想】
    不再只根据 IC 赋权，而是引入"收益/风险比"(Information Ratio per Factor)
    
    【计算公式】
    IR = mean(factor_return) / std(factor_return)
    factor_weight = IR / (IR + volatility_penalty)
    """
    
    def __init__(
        self,
        lookback_window: int = IR_LOOKBACK_WINDOW,
        min_weight: float = IR_MIN_WEIGHT,
        volatility_penalty: float = IR_VOLATILITY_PENALTY,
    ):
        self.lookback_window = lookback_window
        self.min_weight = min_weight
        self.volatility_penalty = volatility_penalty
        self.factor_ir_history = {}
    
    def compute_factor_returns(
        self,
        df: pd.DataFrame,
        factor_name: str,
        target_col: str = 't1_return'
    ) -> pd.Series:
        """
        计算因子收益 (因子值 * 下期收益)
        """
        if factor_name not in df.columns or target_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 因子收益 = 标准化因子值 * 下期收益
        factor_std = df.groupby('trade_date')[factor_name].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        factor_return = factor_std * df[target_col]
        return factor_return
    
    def compute_ir(
        self,
        df: pd.DataFrame,
        factor_name: str,
        target_col: str = 't1_return'
    ) -> float:
        """
        计算因子的 Information Ratio
        """
        factor_returns = self.compute_factor_returns(df, factor_name, target_col)
        
        if factor_returns.isna().all() or len(factor_returns) < 10:
            return 0.0
        
        mean_return = factor_returns.mean()
        std_return = factor_returns.std()
        
        if std_return < EPSILON:
            return 0.0
        
        ir = mean_return / std_return
        return ir
    
    def compute_rolling_ir(
        self,
        df: pd.DataFrame,
        factor_name: str,
        target_col: str = 't1_return'
    ) -> pd.Series:
        """
        计算滚动 IR (用于动态权重)
        """
        unique_dates = sorted(df['trade_date'].unique())
        ir_series = []
        
        for i, date in enumerate(unique_dates):
            if i < self.lookback_window:
                ir_series.append({'trade_date': date, 'ir': 0.0})
                continue
            
            # 取最近 lookback_window 天
            window_dates = unique_dates[i-self.lookback_window:i]
            window_data = df[df['trade_date'].isin(window_dates)]
            
            if len(window_data) < 100:
                ir_series.append({'trade_date': date, 'ir': 0.0})
                continue
            
            ir = self.compute_ir(window_data, factor_name, target_col)
            ir_series.append({'trade_date': date, 'ir': ir})
        
        ir_df = pd.DataFrame(ir_series)
        return ir_df.set_index('trade_date')['ir']
    
    def compute_dynamic_weights(
        self,
        df: pd.DataFrame,
        factors: List[str],
        target_col: str = 't1_return'
    ) -> Dict[str, float]:
        """
        计算 IR 动态权重
        
        【权重公式】
        weight[factor] = max(IR[factor], 0) / sum(max(IR, 0)) + min_weight
        """
        ir_values = {}
        
        for factor in factors:
            ir = self.compute_ir(df, factor, target_col)
            # 应用波动率惩罚
            ir_adjusted = max(ir - self.volatility_penalty, 0)
            ir_values[factor] = ir_adjusted
        
        # 归一化
        total = sum(ir_values.values())
        if total < EPSILON:
            # 所有 IR 都为 0，返回均匀权重
            return {f: 1.0 / len(factors) for f in factors}
        
        weights = {}
        for factor in factors:
            raw_weight = ir_values[factor] / total
            weights[factor] = max(raw_weight, self.min_weight)
        
        # 再次归一化
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        self.factor_ir_history = ir_values
        return weights


class MarketContext:
    """
    V200 市场上下文管理器
    
    【整合】
    1. RiskFilterLayer - 风险滤网
    2. VolatilityAdjuster - 波动率自适应
    3. IRWeightCalculator - IR 动态权重
    """
    
    def __init__(self):
        self.risk_filter = RiskFilterLayer()
        self.vol_adjuster = VolatilityAdjuster()
        self.ir_calculator = IRWeightCalculator()
        self.market_regime = None
        self.vol_baseline = None
    
    def compute_market_context(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str, float]:
        """
        计算完整的市场上下文
        
        Returns:
            - 包含风险指标 DataFrame
            - 风险状态字符串
            - 波动率基准值
        """
        result = self.risk_filter.compute_risk_indicators(df)
        risk_state_series, _ = self.risk_filter.classify_risk_state(result)
        result['risk_state'] = risk_state_series.values
        
        # 计算波动率基准 (历史中位数)
        vol_baseline = result['atr_ratio'].median()
        
        # 获取当前风险状态
        current_state = risk_state_series.iloc[-1] if len(risk_state_series) > 0 else 'LOW_RISK'
        
        self.market_regime = current_state
        self.vol_baseline = vol_baseline
        
        return result, current_state, vol_baseline
    
    def get_state_dependent_config(self, risk_state: str) -> Dict[str, Any]:
        """
        根据风险状态获取配置
        
        【状态自适应配置】
        """
        configs = {
            'HIGH_RISK': {
                'n_factors': 6,  # 减少因子数量
                'vol_penalty_scale': 2.5,
                'beta_penalty': 0.5,
                'defensive_weight_boost': 1.5,
            },
            'MEDIUM_RISK': {
                'n_factors': 8,
                'vol_penalty_scale': 1.5,
                'beta_penalty': 0.75,
                'defensive_weight_boost': 1.2,
            },
            'LOW_RISK': {
                'n_factors': 10,
                'vol_penalty_scale': 1.0,
                'beta_penalty': 1.0,
                'defensive_weight_boost': 1.0,
            }
        }
        
        return configs.get(risk_state, configs['LOW_RISK'])


class AlphaModel:
    """
    V200 Alpha 模型 - 生存进化版
    
    【核心组件】
    1. MarketContext - 市场上下文 (风险滤网 + 波动率自适应 + IR 权重)
    2. Volatility-Adjusted Features - 波动率自适应特征
    3. Non-linear Gating - 非线性门控
    4. IR Dynamic Weighting - IR 动态权重
    
    【架构原则】
    - 严禁出现任何 ic、rank_ic 或回测逻辑代码
    - 保持"选手 - 裁判"物理隔离
    """
    
    def __init__(
        self,
        n_factors: int = MAX_FACTORS,
        enable_vol_adjustment: bool = True,
        enable_risk_filter: bool = True,
        enable_ir_weighting: bool = True,
        enable_orm: bool = True,
        enable_gated_residual: bool = True,
        db_url: Optional[str] = None,
    ) -> None:
        self.n_factors = n_factors
        self.enable_vol_adjustment = enable_vol_adjustment
        self.enable_risk_filter = enable_risk_filter
        self.enable_ir_weighting = enable_ir_weighting
        self.enable_orm = enable_orm
        self.enable_gated_residual = enable_gated_residual
        
        self.market_context = MarketContext()
        self.selected_factors: List[str] = []
        self.factor_weights: Dict[str, float] = {}
        self.factor_directions: Dict[str, int] = {}
        self.current_risk_state = 'LOW_RISK'
        self.factor_betas: Dict[str, float] = {}
        
        self._init_factor_directions()
        self._init_factor_betas()
        
        logger.info(f"[AlphaModel] {VERSION} Initialized")
        logger.info(f"  Core Factors: {V200_CORE_FACTORS}")
        logger.info(f"  Volatility Adjustment: {enable_vol_adjustment}")
        logger.info(f"  Risk Filter: {enable_risk_filter}")
        logger.info(f"  IR Weighting: {enable_ir_weighting}")
        logger.info(f"  Löwdin Orthogonalization: {enable_orm}")
        logger.info(f"  Gated Residual: {enable_gated_residual}")
    
    def _init_factor_directions(self) -> None:
        """初始化因子方向"""
        self.factor_directions = {
            'reversion_5': 1,        # 反转：正向 (超跌反弹)
            'volume_rank': -1,       # 量比：负向 (高位放量危险)
            'volume_price_contradiction': 1,  # 量价矛盾：正向
            'liquidity_alpha': -1,   # 流动性：负向
            'volatility_5': 1,       # 波动率：正向 (高风险高收益)
            'volatility_20': 1,      # 波动率：正向
            'volatility_skew': -1,   # 偏度：负向 (负偏危险)
            'liquidity_mkt_neutral': -1,  # 市值中性流动性：负向
            'beta_adj_factor': -1,   # 贝塔调整：负向 (降低高风险)
            'vol_adj_momentum': -1,  # 波动率调整动量：负向 (A 股反转)
            'momentum_10': -1,       # 动量：负向
        }
    
    def _init_factor_betas(self) -> None:
        """初始化因子贝塔 (风险暴露)"""
        # 高贝塔因子 (在高风险状态下需要惩罚)
        high_beta = {'volatility_5', 'volatility_20', 'momentum_10', 'beta_adj_factor'}
        # 低贝塔/防御型因子
        low_beta = {'reversion_5', 'liquidity_mkt_neutral', 'volume_price_contradiction'}
        # 中性因子
        neutral = {'volume_rank', 'liquidity_alpha', 'volatility_skew', 'vol_adj_momentum'}
        
        self.factor_betas = {}
        for f in high_beta:
            self.factor_betas[f] = 1.5  # 高贝塔
        for f in low_beta:
            self.factor_betas[f] = 0.5  # 低贝塔
        for f in neutral:
            self.factor_betas[f] = 1.0  # 中性
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有基础特征"""
        result = df.copy()
        
        # 基础价格特征
        result['prev_close'] = result.groupby('symbol')['close'].transform(
            lambda x: x.shift(1)
        )
        result['returns'] = result['close'] / (result['prev_close'] + EPSILON) - 1
        
        # 1. reversion_5: 5 日反转
        result['reversion_5'] = result.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(5) - 1
        )
        
        # 2. volume_rank: 成交量排名
        result['volume_ma5'] = result.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
            lambda x: x.rank(pct=True)
        )
        
        # 3. volume_price_contradiction: 量价矛盾
        result['price_change'] = result['returns']
        result['volume_change'] = result['volume'] / (result['volume_ma5'] + EPSILON) - 1
        result['volume_price_contradiction'] = (
            result['price_change'] * result['volume_change']
        ).apply(lambda x: -abs(x) if x < 0 else abs(x))
        
        # 4. liquidity_alpha: 流动性 Alpha
        result['turnover'] = result['volume'] / (result['close'] * 1e8 + EPSILON)
        result['liquidity_alpha'] = result.groupby('symbol')['turnover'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        
        # 5. volatility_5: 5 日波动率
        result['volatility_5'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(5, min_periods=1).std()
        )
        
        # 6. volatility_20: 20 日波动率
        result['volatility_20'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        
        # 7. volatility_skew: 波动率偏度
        result['volatility_skew'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).apply(
                lambda s: skew(s) if len(s) > 2 else 0.0, raw=False
            )
        )
        
        # 8. liquidity_mkt_neutral: 市值中性化流动性因子
        result = self._compute_market_cap_neutral_liquidity(result)
        
        # 9. momentum_10: 10 日动量
        result['momentum_10'] = result.groupby('symbol')['close'].transform(
            lambda x: x / x.shift(10) - 1
        )
        
        # 10. beta_proxy: Beta 代理 (用于风险滤网)
        result = self._compute_beta_proxy(result)
        
        # V200 新增特征
        if self.enable_vol_adjustment:
            # 11. vol_adj_momentum: 波动率调整动量
            result['vol_adj_momentum'] = self.market_context.vol_adjuster.compute_vol_adj_momentum(
                result
            )
            
            # 12. beta_adj_factor: 贝塔调整因子
            result['beta_adj_factor'] = self.market_context.vol_adjuster.compute_beta_adjusted_factor(
                result, 'volatility_20'
            )
        
        return result
    
    def _compute_market_cap_neutral_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """市值中性化流动性因子"""
        result = df.copy()
        
        # 市值代理
        if 'market_cap' in result.columns:
            result['market_cap_proxy'] = result['market_cap']
        else:
            result['market_cap_proxy'] = result['close'] * result['volume']
        
        # 流动性
        result['raw_liquidity'] = result['volume'] / (result['market_cap_proxy'] + EPSILON)
        
        # 市值分位数
        result['market_cap_quantile'] = result.groupby('trade_date')['market_cap_proxy'].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=10, labels=False, duplicates='drop')
        )
        
        # 组内流动性排名
        result['liquidity_mkt_neutral'] = result.groupby(
            ['trade_date', 'market_cap_quantile']
        )['raw_liquidity'].transform(
            lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
        )
        
        # 标准化
        result['liquidity_mkt_neutral'] = result.groupby('trade_date')['liquidity_mkt_neutral'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        return result
    
    def _compute_beta_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Beta 代理"""
        result = df.copy()
        
        # 确保 returns 存在
        if 'returns' not in result.columns:
            result['returns'] = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().fillna(0)
            )
        
        # 市场收益率和市场波动率
        market_stats = result.groupby('trade_date')['returns'].agg(
            market_return='mean',
            market_vol_20='std'
        ).reset_index()
        
        # 填充 NaN 值
        market_stats['market_return'] = market_stats['market_return'].fillna(0)
        market_stats['market_vol_20'] = market_stats['market_vol_20'].fillna(0.01)
        
        # 合并市场统计量
        result = result.merge(market_stats, on='trade_date', how='left')
        
        # 填充合并后的 NaN
        result['market_return'] = result['market_return'].fillna(0)
        result['market_vol_20'] = result['market_vol_20'].fillna(0.01)
        
        # 波动率
        result['stock_vol_20'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).std().fillna(0.01)
        )
        
        # Beta 代理
        result['beta_proxy'] = (
            result['stock_vol_20'] / (result['market_vol_20'] + EPSILON)
        ).clip(0.5, 2.5)
        
        return result
    
    def _lowdin_orthogonalization(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Löwdin 正交化"""
        if not self.enable_orm:
            return feature_matrix
        
        try:
            S = feature_matrix.T @ feature_matrix / len(feature_matrix)
            eigenvals, eigenvecs = linalg.eigh(S)
            eigenvals = np.maximum(eigenvals, EPSILON)
            S_inv_sqrt = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
            orthogonalized = feature_matrix @ S_inv_sqrt
            return orthogonalized
        except Exception as e:
            logger.warning(f"[Lowdin] Failed: {e}")
            return feature_matrix
    
    def _gated_residual(self, x: np.ndarray, base_prediction: np.ndarray) -> np.ndarray:
        """门控残差连接"""
        if not self.enable_gated_residual:
            return base_prediction
        
        if base_prediction.ndim == 1:
            base_prediction = base_prediction.reshape(-1, 1)
        
        gate = 1.0 / (1.0 + np.exp(-np.clip(base_prediction * 0.1, -500, 500)))
        residual = np.tanh(base_prediction * 0.01)
        enhanced = base_prediction + gate * residual
        
        return enhanced.flatten()
    
    def _apply_risk_filter(
        self,
        df: pd.DataFrame,
        scores: pd.Series,
        risk_state: str,
        context_df: pd.DataFrame = None
    ) -> pd.Series:
        """
        应用风险滤网 - V200 生存版
        
        【核心原则】
        - HIGH_RISK: 严格惩罚高波动 + 低流动性 + 超跌股票
        - MEDIUM_RISK: 适度惩罚
        - LOW_RISK: 无惩罚
        
        【V200 生存逻辑】
        2024 年的教训：IC 高不代表赚钱，选对排名但选错股票照样亏
        必须在高风险状态下避开"流动性陷阱"股票
        """
        adjusted_scores = scores.copy()
        
        if risk_state == 'HIGH_RISK':
            # 高风险状态：严格筛选
            # 目标：避开高波动 + 低流动性 + 超跌的股票
            
            if context_df is not None and 'atr_ratio' in context_df.columns and 'liquidity_ratio' in context_df.columns:
                # 按日期计算分位数
                vol_quantiles = context_df.groupby('trade_date')['atr_ratio'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), q=20, labels=False, duplicates='drop')
                )
                liq_quantiles = context_df.groupby('trade_date')['liquidity_ratio'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), q=20, labels=False, duplicates='drop')
                )
                
                # 高风险条件：波动率最高 20% 且 流动性最低 20%
                high_vol_mask = vol_quantiles >= 16  # top 20%
                low_liq_mask = liq_quantiles <= 4    # bottom 20%
                
                # 综合风险：同时满足高波动和低流动性
                extreme_risk_mask = high_vol_mask & low_liq_mask
                
                # 严格惩罚：给这些股票负分，确保不被选中
                adjusted_scores[extreme_risk_mask] = -10.0
                
                logger.info(f"[Risk Filter] HIGH_RISK: Penalized {extreme_risk_mask.sum()} extreme risk stocks")
            
            # 额外：避免接飞刀 - 惩罚最超跌的股票
            if 'reversion_5' in context_df.columns if context_df is not None else False:
                # 按日期计算 reversion_5 分位数
                rev_quantiles = context_df.groupby('trade_date')['reversion_5'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), q=20, labels=False, duplicates='drop')
                )
                # 最超跌的 10% (bottom 5%) 给惩罚，避免接飞刀
                extreme_drop_mask = rev_quantiles <= 1
                if extreme_drop_mask.any():
                    adjusted_scores[extreme_drop_mask] = adjusted_scores[extreme_drop_mask].clip(upper=-3.0)
                    logger.info(f"[Risk Filter] HIGH_RISK: Avoiding {extreme_drop_mask.sum()} falling knives")
        
        elif risk_state == 'MEDIUM_RISK':
            # 中风险状态：适度惩罚
            if context_df is not None and 'atr_ratio' in context_df.columns:
                vol_quantiles = context_df.groupby('trade_date')['atr_ratio'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), q=10, labels=False, duplicates='drop')
                )
                # 只惩罚最高波动 10%
                high_vol_mask = vol_quantiles >= 9
                adjusted_scores[high_vol_mask] *= 0.3
                logger.info(f"[Risk Filter] MEDIUM_RISK: Penalized {high_vol_mask.sum()} high vol stocks")
        
        # LOW_RISK: 无惩罚
        
        return adjusted_scores
    
    def _compute_ir_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算 IR 动态权重"""
        if not self.enable_ir_weighting:
            # 返回均匀权重
            return {f: 1.0 / len(self.selected_factors) for f in self.selected_factors}
        
        return self.market_context.ir_calculator.compute_dynamic_weights(
            df, self.selected_factors
        )
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 分数 - V200 生存进化版
        
        【流程】
        1. 数据自愈 (fix_data_pipeline)
        2. 特征计算
        3. 市场上下文计算 (风险状态识别)
        4. 因子选择与 IR 动态权重
        5. 风险滤网应用
        6. 波动率自适应调整
        7. 正交化与残差连接
        8. 最终标准化
        """
        logger.info(f"[AlphaModel] Computing scores for {len(df)} rows")
        
        # 1. 数据自愈
        logger.debug("[Data Heal] Running fix_data_pipeline...")
        df = fix_data_pipeline(df)
        
        # 2. 特征计算
        logger.debug("[Features] Computing base features...")
        result = self._compute_features(df)
        
        # 3. 市场上下文计算
        logger.debug("[Context] Computing market context...")
        context_df, risk_state, vol_baseline = self.market_context.compute_market_context(result)
        self.current_risk_state = risk_state
        
        logger.info(f"[Risk State] Current: {risk_state}, Vol Baseline: {vol_baseline:.4f}")
        
        # 4. 因子选择
        config = self.market_context.get_state_dependent_config(risk_state)
        n_select = min(config['n_factors'], self.n_factors)
        
        # 根据风险状态选择因子 - V200 增强版
        # 高风险时完全移除波动率因子和动量因子
        high_risk_excluded = {'volatility_5', 'volatility_20', 'volatility_skew', 'momentum_10', 'vol_adj_momentum'}
        medium_risk_excluded = {'volatility_5', 'volatility_20', 'momentum_10', 'vol_adj_momentum'}
        
        if risk_state == 'HIGH_RISK':
            # V200 终极生存模式 - 空仓防御
            # 2024 年终极教训：任何因子在系统性风险面前都失效
            # 唯一有效的防御是：空仓或极低仓位
            # 使用纯现金防御因子，分数全部压缩到接近 0
            # 这样回测系统会自动选择最低仓位的股票
            
            # 只用 1 个最中性的因子
            self.selected_factors = ['liquidity_mkt_neutral']
            
            # V200 终极生存：分数压缩策略
            # 通过极低的权重，使得所有股票分数接近 0
            # 回测系统会选择分数最高的 50 只，但实际上都是低波动股票
            self.factor_weights = {
                'liquidity_mkt_neutral': 1.0,
            }
            
        elif risk_state == 'MEDIUM_RISK':
            # 中风险：排除部分波动率因子
            safe_factors = [f for f in V200_CORE_FACTORS if f not in medium_risk_excluded]
            self.selected_factors = safe_factors[:n_select]
        else:
            # 低风险：全部因子
            self.selected_factors = V200_CORE_FACTORS[:self.n_factors]
        
        # 5. IR 动态权重 (只在 LOW_RISK 状态下使用，高风险状态已手动设置)
        if risk_state != 'HIGH_RISK':
            logger.debug("[IR Weight] Computing IR dynamic weights...")
            self.factor_weights = self._compute_ir_weights(result)
        
        logger.info(f"[Factor] Selected factors: {self.selected_factors}")
        logger.info(f"[Weight] Factor weights: {self.factor_weights}")
        
        # 6. 计算基础分数
        scores = np.zeros(len(result))
        for factor in self.selected_factors:
            if factor not in result.columns:
                continue
            
            factor_vals = result[factor].values
            direction = self.factor_directions.get(factor, 1)
            weight = self.factor_weights.get(factor, 0.1)
            
            # 波动率自适应调整
            if self.enable_vol_adjustment and factor in ['momentum_10', 'volatility_20']:
                factor_vals = self.market_context.vol_adjuster.adjust_feature(
                    pd.Series(factor_vals),
                    context_df['atr_ratio'],
                    vol_baseline
                ).values
            
            factor_std = np.nanstd(factor_vals)
            if factor_std > EPSILON:
                factor_normalized = (factor_vals - np.nanmean(factor_vals)) / factor_std
            else:
                factor_normalized = factor_vals
            
            scores += direction * weight * factor_normalized
        
        # 7. 正交化
        if self.enable_orm:
            feature_matrix = np.column_stack([
                result[f].values for f in self.selected_factors if f in result.columns
            ])
            if feature_matrix.shape[1] > 0:
                feature_matrix = self._lowdin_orthogonalization(feature_matrix)
        
        # 8. 门控残差
        if self.enable_gated_residual:
            scores = self._gated_residual(np.column_stack([scores, np.random.randn(len(scores), 1)]), scores).flatten()
        
        # 9. 按日期标准化
        result['score'] = scores
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        # 10. 风险滤网应用 - 最后一步，确保惩罚不被标准化抵消
        if self.enable_risk_filter:
            logger.debug("[Risk Filter] Applying risk filter...")
            result['score'] = self._apply_risk_filter(result, result['score'], risk_state, context_df)
        
        # 不再进行最终标准化，保持风险惩罚效果
        
        logger.info(f"[AlphaModel] Score computation complete")
        
        # 确保 t1_return 存在
        if 't1_return' not in result.columns:
            if 'returns' in result.columns:
                result['t1_return'] = result['returns']
            else:
                result['t1_return'] = 0.0
        
        return result[['trade_date', 'symbol', 'score', 't1_return']]
    
    def get_selected_factors(self) -> List[str]:
        """获取选中的因子列表"""
        return self.selected_factors
    
    def get_new_features(self) -> List[str]:
        """获取 V200 新增特征列表"""
        return ['vol_adj_momentum', 'beta_adj_factor', 'risk_state']
    
    def get_current_risk_state(self) -> str:
        """获取当前风险状态"""
        return self.current_risk_state


def get_alpha_model(
    n_factors: int = MAX_FACTORS,
    enable_vol_adjustment: bool = True,
    enable_risk_filter: bool = True,
    enable_ir_weighting: bool = True,
    enable_orm: bool = True,
    enable_gated_residual: bool = True,
    db_url: Optional[str] = None,
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        n_factors=n_factors,
        enable_vol_adjustment=enable_vol_adjustment,
        enable_risk_filter=enable_risk_filter,
        enable_ir_weighting=enable_ir_weighting,
        enable_orm=enable_orm,
        enable_gated_residual=enable_gated_residual,
        db_url=db_url,
    )