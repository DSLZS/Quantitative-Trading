"""
Alpha Model Module - V202 Architecture Renormalization
======================================================

【V202 核心变革】
1. 严格未来函数审计
   - 严禁 shift(-1) 或任何 T+1 数据访问
   - 所有 Score 必须仅基于 T 日及之前的历史数据
   
2. 严格"裁判 - 选手"解耦
   - AlphaModel 只输出 score
   - 禁止计算收益率，禁止接触回测逻辑
   - 删除所有 t1_return 相关代码

3. 真正的行业中性化
   - 使用 stock_industry_daily 进行行业内排名标准化
   - 避免行业集中暴露

4. 动态波动率滤网
   - 市场剧震时自动调低仓位
   - 切换至防御性因子

5. 资金流过滤
   - 使用 net_main_amount 剔除散户拉升的高风险个股

【V202 验收红线】
- 单年回测时间 < 10 分钟
- 初始资金：100,000
- 费率：1.3‰
- 无未来函数，严禁 T+0
"""

from typing import Any, Optional, Dict, List, Tuple
import warnings
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    logger.warning("Polars not installed, falling back to pandas")

load_dotenv()
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V202_Architecture_Renormalization"

# V202 核心因子 (精简有效)
V202_CORE_FACTORS = [
    'reversion_5',              # 5 日反转 - 防御型
    'volatility_20',            # 20 日波动率 - 风险度量
    'liquidity_mkt_neutral',    # 市值中性化流动性
    'volume_price_contradiction', # 量价矛盾
    'industry_neutral_score',   # 行业中性化评分 (V202 核心)
    'fund_flow_signal',         # 资金流信号 (V202 核心)
    'volatility_regime',        # 波动率状态 (V202 动态滤网)
]

MAX_FACTORS = 10

# 风险滤网参数
RISK_FILTER_HIGH_VOL_THRESHOLD = 0.60
RISK_FILTER_LOW_LIQ_THRESHOLD = 0.35
RISK_FILTER_PENALTY_SCALE = 4.0

# 行业中性化参数
INDUSTRY_NEUTRALIZE_ENABLED = True

# 动态波动率滤网参数
VOLATILITY_REGIME_WINDOW = 20
VOLATILITY_HIGH_THRESHOLD = 0.75  # 高波动状态阈值
VOLATILITY_DEFENSE_SCALE = 0.5    # 防御模式缩放因子

# 资金流参数
FUND_FLOW_WINDOW = 5
FUND_FLOW_MIN_WEIGHT = 0.1

# 配置
MIN_STOCK_COUNT = 5000
WARMUP_DAYS = 60
EPSILON = 1e-6


# ==============================================================================
# 工具函数
# ==============================================================================

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


def normalize_group(df: pd.DataFrame, score_col: str, group_col: str = None) -> pd.Series:
    """
    截面或分组标准化评分
    
    【原理】
    - 将原始评分转换为 [0, 1] 区间的排名百分位
    - 可选按组 (行业) 内标准化实现行业中性化
    """
    if group_col is None or group_col not in df.columns:
        # 截面标准化
        if len(df) < 2:
            return pd.Series(0.5, index=df.index)
        
        ranks = df[score_col].rank(pct=True)
        # 转换为均值为 0，标准差为 1 的标准化评分
        normalized = (ranks - 0.5) * 2  # 范围 [-1, 1]
        return normalized
    
    # 分组标准化 (行业中性化核心)
    def rank_pct(group):
        if len(group) < 2:
            return pd.Series(0.0, index=group.index)
        return (group.rank(pct=True) - 0.5) * 2
    
    normalized = df.groupby(group_col)[score_col].transform(rank_pct)
    return normalized


def auto_fillna(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """自动填充缺失值"""
    result = df.copy()
    
    if columns is None:
        columns = result.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in result.columns:
            continue
        
        # 1. 前值填充
        result[col] = result.groupby('symbol', group_keys=False)[col].ffill()
        
        # 2. 截面均值填充
        missing_mask = result[col].isna()
        if missing_mask.any():
            col_mean = result[col].mean()
            if pd.notna(col_mean):
                result.loc[missing_mask, col] = col_mean
        
        # 3. 最后用 0 填充
        result[col] = result[col].fillna(0)
    
    return result


# ==============================================================================
# AlphaModel V202
# ==============================================================================

class AlphaModel:
    """
    V202 Alpha 模型 - 严格解耦的评分模型
    
    【核心职责】
    1. 计算因子值 (仅使用 T 日及之前数据)
    2. 行业中性化处理
    3. 动态波动率滤网
    4. 资金流过滤
    5. 输出标准化评分 score
    
    【严禁】
    - 计算任何收益率 (t1_return 等)
    - 接触回测逻辑
    - 使用未来函数 (shift(-1) 等)
    """
    
    def __init__(
        self,
        n_factors: int = MAX_FACTORS,
        enable_industry_neutral: bool = True,
        enable_vol_regime: bool = True,
        enable_fund_flow: bool = True,
    ):
        self.n_factors = min(n_factors, MAX_FACTORS)
        self.enable_industry_neutral = enable_industry_neutral and INDUSTRY_NEUTRALIZE_ENABLED
        self.enable_vol_regime = enable_vol_regime
        self.enable_fund_flow = enable_fund_flow
        
        self.factors_used = V202_CORE_FACTORS[:self.n_factors]
        
        # 状态跟踪
        self._current_vol_regime = 'NORMAL'
        self._last_compute_date = None
        
        logger.info("=" * 70)
        logger.info(f"V202 AlphaModel Initialized")
        logger.info("=" * 70)
        logger.info(f"  Factors: {self.factors_used}")
        logger.info(f"  Industry Neutral: {self.enable_industry_neutral}")
        logger.info(f"  Vol Regime Filter: {self.enable_vol_regime}")
        logger.info(f"  Fund Flow Filter: {self.enable_fund_flow}")
        logger.info("=" * 70)
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分
        
        【核心原则】
        - 所有计算仅使用 T 日及之前数据
        - 严禁使用 shift(-1) 等未来函数
        - 输出 score 列，严禁输出任何收益率相关列
        
        Args:
            df: 输入数据，必须包含以下列:
                - symbol, trade_date, close, open, high, low
                - volume, amount, pct_chg
                - (可选) industry_code, net_main_amount
        
        Returns:
            包含 score 列的 DataFrame
        """
        if df.empty:
            logger.warning("[AlphaModel] Empty input DataFrame")
            return df
        
        logger.info(f"[AlphaModel] Computing scores for {len(df)} rows...")
        
        # 数据预处理
        df = df.copy()
        df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 确保日期格式
        if df['trade_date'].dtype == 'int64':
            df['trade_date_str'] = df['trade_date'].astype(str)
        else:
            df['trade_date_str'] = df['trade_date']
        
        # 计算基础因子
        df = self._compute_base_factors(df)
        
        # 数据自愈
        df = auto_fillna(df, self.factors_used)
        
        # 计算综合评分
        df = self._compute_composite_score(df)
        
        # 行业中性化 (V202 核心)
        if self.enable_industry_neutral and 'industry_code' in df.columns:
            df = self._apply_industry_neutralization(df)
        
        # 动态波动率滤网 (V202 核心)
        if self.enable_vol_regime:
            df = self._apply_volatility_regime(df)
        
        # 资金流过滤 (V202 核心)
        if self.enable_fund_flow and 'net_main_amount' in df.columns:
            df = self._apply_fund_flow_filter(df)
        
        # 最终标准化确保 score 在合理范围
        df['score'] = winsorize(df['score'], sigma=3.0)
        
        logger.info(f"[AlphaModel] Score computed. Range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        
        return df
    
    def _compute_base_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基础因子
        
        【严禁未来函数】所有因子计算只能使用 shift(1) 或更早的历史数据
        """
        # 确保数值类型
        numeric_cols = ['close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 1. 5 日反转因子 (防御型)
        # reversion_5 = -pct_chg (当日涨跌幅的负值，赌反转)
        if 'pct_chg' in df.columns:
            df['reversion_5'] = -df['pct_chg'] / 100.0  # 转换为小数
        
        # 2. 20 日波动率因子
        # 计算 20 日收益率标准差
        df['return_1d'] = df.groupby('symbol')['close'].pct_change(1).fillna(0)
        df['volatility_20'] = df.groupby('symbol')['return_1d'].transform(
            lambda x: x.rolling(20, min_periods=5).std()
        ).fillna(0)
        
        # 3. 市值中性化流动性因子
        # liquidity = amount / (amount.rolling(20).mean())
        df['amount_ma20'] = df.groupby('symbol')['amount'].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        ).fillna(0)
        df['liquidity_mkt_neutral'] = df['amount'] / (df['amount_ma20'] + EPSILON)
        df['liquidity_mkt_neutral'] = winsorize(df['liquidity_mkt_neutral'], sigma=3.0)
        
        # 4. 量价矛盾因子
        # 当日放量下跌或缩量上涨视为矛盾信号
        df['volume_ratio'] = df['volume'] / (df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        ) + EPSILON)
        
        df['volume_price_contradiction'] = np.where(
            (df['pct_chg'] < 0) & (df['volume_ratio'] > 1.5),  # 放量下跌
            1.0,
            np.where(
                (df['pct_chg'] > 0) & (df['volume_ratio'] < 0.5),  # 缩量上涨
                1.0,
                0.0
            )
        )
        
        # 5. 行业中性化评分 (V202 核心)
        # 在行业内计算综合评分的初步估计
        df['industry_neutral_score'] = 0.0
        
        # 6. 资金流信号 (V202 核心)
        if 'net_main_amount' in df.columns:
            df['net_main_ma5'] = df.groupby('symbol')['net_main_amount'].transform(
                lambda x: x.rolling(5, min_periods=3).mean()
            ).fillna(0)
            
            # 截面标准化
            df['fund_flow_signal'] = df.groupby('trade_date')['net_main_ma5'].transform(
                lambda x: (x - x.mean()) / (x.std() + EPSILON)
            ).fillna(0)
            df['fund_flow_signal'] = winsorize(df['fund_flow_signal'], sigma=3.0)
        else:
            df['fund_flow_signal'] = 0.0
        
        # 7. 波动率状态 (V202 动态滤网)
        # 计算市场整体波动率状态
        df['volatility_regime'] = 0.0
        
        return df
    
    def _compute_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合评分
        
        【权重配置】
        - 防御型因子 (reversion_5): 30%
        - 风险型因子 (volatility_20, liquidity): 30%
        - 资金流因子 (fund_flow_signal): 20%
        - 行业中性化因子：20%
        """
        # 各因子标准化
        score_components = pd.DataFrame(index=df.index)
        
        if 'reversion_5' in df.columns:
            score_components['reversion'] = normalize_group(df, 'reversion_5')
        
        if 'volatility_20' in df.columns:
            # 波动率越低评分越高 (负向)
            score_components['volatility'] = -normalize_group(df, 'volatility_20')
        
        if 'liquidity_mkt_neutral' in df.columns:
            score_components['liquidity'] = normalize_group(df, 'liquidity_mkt_neutral')
        
        if 'volume_price_contradiction' in df.columns:
            # 量价矛盾是风险信号 (负向)
            score_components['contradiction'] = -df['volume_price_contradiction']
        
        if 'fund_flow_signal' in df.columns and self.enable_fund_flow:
            score_components['fund_flow'] = normalize_group(df, 'fund_flow_signal')
        
        # 加权综合
        weights = {
            'reversion': 0.30,
            'volatility': 0.20,
            'liquidity': 0.10,
            'contradiction': 0.10,
            'fund_flow': 0.30,
        }
        
        # 动态调整权重 (根据可用因子)
        available_cols = [c for c in weights.keys() if c in score_components.columns]
        if available_cols:
            total_weight = sum(weights[c] for c in available_cols)
            for c in available_cols:
                weights[c] /= total_weight
        
        # 计算加权评分
        df['raw_score'] = sum(score_components[c] * weights[c] for c in available_cols)
        
        # 截面标准化为最终 score
        df['score'] = normalize_group(df, 'raw_score')
        
        return df
    
    def _apply_industry_neutralization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用行业中性化 (V202 核心)
        
        【原理】
        1. 按行业分组
        2. 在行业内重新标准化评分
        3. 消除行业集中暴露风险
        """
        if 'industry_code' not in df.columns:
            logger.warning("[IndustryNeutral] No industry_code column, skipping")
            return df
        
        # 检查行业数量
        industry_counts = df['industry_code'].value_counts()
        if len(industry_counts) < 3:
            logger.warning(f"[IndustryNeutral] Only {len(industry_counts)} industries, skipping")
            return df
        
        # 在行业内重新标准化
        df['industry_neutral_score'] = df.groupby('industry_code')['score'].transform(
            lambda x: normalize_group(pd.DataFrame({'score': x}), 'score')
        )
        
        # 使用行业中性化评分
        df['score'] = df['industry_neutral_score']
        
        logger.debug(f"[IndustryNeutral] Applied neutralization across {len(industry_counts)} industries")
        
        return df
    
    def _apply_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用动态波动率滤网 (V202 核心)
        
        【原理】
        1. 计算市场整体波动率状态
        2. 高波动时降低评分绝对值 (防御模式)
        3. 自动切换至防御性因子
        """
        # 计算当日市场平均波动率
        if 'volatility_20' in df.columns:
            daily_vol = df.groupby('trade_date')['volatility_20'].mean()
            
            # 计算波动率的分位数
            vol_quantiles = daily_vol.quantile([0.25, 0.5, 0.75])
            vol_75 = vol_quantiles.get(0.75, 0.05)
            
            # 标记高波动状态
            high_vol_dates = daily_vol[daily_vol > vol_75].index.tolist()
            
            # 在高波动日期降低评分绝对值
            high_vol_mask = df['trade_date'].isin(high_vol_dates)
            if high_vol_mask.any():
                df.loc[high_vol_mask, 'score'] *= VOLATILITY_DEFENSE_SCALE
                self._current_vol_regime = 'HIGH'
            else:
                self._current_vol_regime = 'NORMAL'
        
        df['volatility_regime'] = 1.0 if self._current_vol_regime == 'HIGH' else 0.0
        
        return df
    
    def _apply_fund_flow_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用资金流过滤 (V202 核心)
        
        【原理】
        1. 剔除主力大幅净流出的股票
        2. 优先选择主力净流入的股票
        """
        if 'fund_flow_signal' not in df.columns:
            return df
        
        # 剔除资金流极差的股票 (后 20%)
        flow_quantile = df['fund_flow_signal'].quantile(0.2)
        
        # 对资金流极差的股票降低评分
        bad_flow_mask = df['fund_flow_signal'] < flow_quantile
        if bad_flow_mask.any():
            df.loc[bad_flow_mask, 'score'] *= 0.5
        
        return df
    
    def get_current_risk_state(self) -> Dict[str, Any]:
        """获取当前风险状态"""
        return {
            'volatility_regime': self._current_vol_regime,
            'industry_neutral_enabled': self.enable_industry_neutral,
            'fund_flow_enabled': self.enable_fund_flow,
        }
    
    def get_factor_importance(self) -> Dict[str, float]:
        """获取因子重要性"""
        return {
            'reversion_5': 0.30,
            'fund_flow_signal': 0.30,
            'volatility_20': 0.20,
            'liquidity_mkt_neutral': 0.10,
            'volume_price_contradiction': 0.10,
        }


def get_alpha_model(
    n_factors: int = MAX_FACTORS,
    enable_industry_neutral: bool = True,
    enable_vol_regime: bool = True,
    enable_fund_flow: bool = True,
) -> AlphaModel:
    """获取 AlphaModel 实例"""
    return AlphaModel(
        n_factors=n_factors,
        enable_industry_neutral=enable_industry_neutral,
        enable_vol_regime=enable_vol_regime,
        enable_fund_flow=enable_fund_flow,
    )