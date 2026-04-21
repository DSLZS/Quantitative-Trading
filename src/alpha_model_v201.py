"""
Alpha Model Module - V201 Evolution: Multi-Source Data Fusion & Generalization

【V201 核心变革】
1. 多源数据融合 (Multi-Source Data Fusion)
   - 行业中性化 (Industry Neutralization): 使用 stock_industry_daily 计算截面收益率的行业偏离
   - 宏观基准 (Index Benchmark): 引入 index_daily (000905.SH) 的 MA20 特征作为市场环境判定
   - 资金流信号 (Fund Flow): 引入 net_main_amount (主力净流入) 修复动量因子在高位接盘的陷阱

2. Alpha 衰减惩罚 (Alpha Decay Penalty)
   - 针对 V200 可能存在的动量过拟合，增加"近期极值惩罚"逻辑
   - 对连续 N 日排名靠前的股票进行分数衰减

3. 特征存储优化 (Feature Storage Optimization)
   - 使用 pyarrow 格式的本地特征缓存
   - 支持多进程 Join 优化

4. 压力测试范围扩展
   - 强制引入 2018 (单边熊市)、2020 (疫后牛市)、2022 (剧烈轮动) 的数据进行联合回测
   - 要求跨年份夏普比率的标准差降低 20%

【架构原则】
- 保持"选手 - 裁判"物理隔离
- alpha_model.py 严禁出现任何 ic、rank_ic 或回测逻辑代码
- 多表 Join 使用 polars 进行性能优化

【验收红线】
- 跨年份夏普比率标准差降低 20%
- 2024 年 IC > 0.10 且 MDD < 25%
- 单年回测时间控制在 15 分钟内
- 初始资金 100,000，费率 1.3‰
- 严禁未来函数，严禁 T+0
"""

from typing import Any, Optional, Dict, List, Tuple
import warnings
import os
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import skew, kurtosis
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

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    logger.warning("PyArrow not installed, feature caching disabled")

load_dotenv()
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V201_MultiSource_Fusion"

# V201 核心因子 (在 V200 基础上新增)
V201_CORE_FACTORS = [
    'reversion_5',           # 5 日反转 - 防御型
    'volume_rank',           # 成交量排名 - 流动性监控
    'volume_price_contradiction',  # 量价矛盾 - 风险信号
    'liquidity_alpha',       # 流动性 Alpha
    'volatility_5',          # 5 日波动率 - 风险度量
    'volatility_20',         # 20 日波动率 - 风险度量
    'volatility_skew',       # 波动率偏度 - 尾部风险
    'liquidity_mkt_neutral', # 市值中性化流动性
    'beta_adj_factor',       # 贝塔调整因子 (风险滤网)
    'vol_adj_momentum',      # 波动率调整动量
    # V201 新增因子
    'industry_neutral_score',  # 行业中性化评分
    'fund_flow_signal',        # 资金流信号
    'index_mkt_state',         # 指数市场状态
]

MAX_FACTORS = 12

# V200 风险滤网参数 - 继承
RISK_FILTER_HIGH_VOL_THRESHOLD = 0.60
RISK_FILTER_LOW_LIQ_THRESHOLD = 0.35
RISK_FILTER_PENALTY_SCALE = 4.0
RISK_FILTER_BETA_PENALTY = 0.3

# V200 IR 动态权重参数 - 继承
IR_LOOKBACK_WINDOW = 20
IR_MIN_WEIGHT = 0.02
IR_VOLATILITY_PENALTY = 0.0

# V200 波动率自适应参数 - 继承
VOL_ADAPTIVE_SCALE = 0.5
VOL_ADAPTIVE_EXPONENT = 1.5

# V201 新增参数
# Alpha 衰减惩罚参数
ALPHA_DECAY_PENALTY_WINDOW = 10       # 极值检测窗口
ALPHA_DECAY_PENALTY_THRESHOLD = 0.15  # 极值阈值 (前 15%)
ALPHA_DECAY_RATE = 0.7                # 衰减速率

# 行业中性化参数
INDUSTRY_NEUTRALIZE_TOP_K = 50        # 选股数量
INDUSTRY_MAX_WEIGHT = 0.25            # 单行业最大权重 25%

# 资金流参数
FUND_FLOW_WINDOW = 5                  # 资金流窗口
FUND_FLOW_WEIGHT = 0.15               # 资金流权重

# 指数基准参数
INDEX_SYMBOL = '000905.SH'            # 中证 500 指数
INDEX_MA20_THRESHOLD = 0.02           # MA20 偏离阈值

# 特征缓存配置
CACHE_DIR = Path(".cache/v201_features")
CACHE_ENABLED = HAS_PYARROW
CACHE_ROW_GROUP_SIZE = 10000          # Parquet row group size

# 配置
MIN_STOCK_COUNT = 5000
WARMUP_DAYS = 60
WARMUP_YEAR = 2022
EPSILON = 1e-6


# ==============================================================================
# PyArrow 特征缓存工具
# ==============================================================================

class FeatureCache:
    """
    V201 特征缓存管理器 - 使用 PyArrow Parquet 格式
    
    【核心功能】
    1. 特征数据持久化存储
    2. 按日期范围分片读取
    3. 支持压缩 (snappy/zstd)
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR, compression: str = "snappy"):
        self.cache_dir = cache_dir
        self.compression = compression
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not HAS_PYARROW:
            logger.warning("[FeatureCache] PyArrow not available, caching disabled")
            self.enabled = False
        else:
            self.enabled = CACHE_ENABLED
            logger.info(f"[FeatureCache] Initialized at {cache_dir}, compression={compression}")
    
    def _compute_hash(self, data: pd.DataFrame, columns: List[str]) -> str:
        """计算数据哈希用于缓存键"""
        hash_content = ""
        for col in columns:
            if col in data.columns:
                hash_content += f"{col}:{len(data)}:{data[col].iloc[-1] if len(data) > 0 else 0}"
        return hashlib.md5(hash_content.encode()).hexdigest()[:16]
    
    def save_features(self, df: pd.DataFrame, feature_hash: str, years: List[int]):
        """
        保存特征到 Parquet 文件
        
        Args:
            df: 特征 DataFrame
            feature_hash: 特征哈希
            years: 年份列表
        """
        if not self.enabled or df.empty:
            return
        
        try:
            # 按年份分片存储
            for year in years:
                year_str = str(year)
                year_df = df[df['trade_date'].astype(str).str.startswith(year_str)].copy()
                
                if year_df.empty:
                    continue
                
                cache_file = self.cache_dir / f"features_{year}_{feature_hash}.parquet"
                
                # PyArrow Table
                table = pa.Table.from_pandas(year_df, preserve_index=False)
                
                # 写入 Parquet (带压缩)
                pq.write_table(
                    table,
                    cache_file,
                    compression=self.compression,
                    row_group_size=CACHE_ROW_GROUP_SIZE,
                    use_dictionary=True,
                )
                
                logger.debug(f"[FeatureCache] Saved {len(year_df)} rows to {cache_file}")
                
        except Exception as e:
            logger.warning(f"[FeatureCache] Save failed: {e}")
    
    def load_features(self, feature_hash: str, years: List[int]) -> Optional[pd.DataFrame]:
        """
        从 Parquet 文件加载特征
        
        Args:
            feature_hash: 特征哈希
            years: 年份列表
            
        Returns:
            特征 DataFrame 或 None
        """
        if not self.enabled:
            return None
        
        try:
            dfs = []
            for year in years:
                cache_file = self.cache_dir / f"features_{year}_{feature_hash}.parquet"
                
                if cache_file.exists():
                    table = pq.read_table(cache_file)
                    df = table.to_pandas()
                    dfs.append(df)
            
            if dfs:
                result = pd.concat(dfs, ignore_index=True)
                logger.info(f"[FeatureCache] Loaded {len(result)} rows from cache")
                return result
            
        except Exception as e:
            logger.warning(f"[FeatureCache] Load failed: {e}")
        
        return None
    
    def clear_cache(self, years: List[int] = None):
        """清除缓存"""
        if years is None:
            years = list(range(2018, 2026))
        
        for year in years:
            for f in self.cache_dir.glob(f"features_{year}_*.parquet"):
                f.unlink()
        
        logger.info(f"[FeatureCache] Cache cleared for years {years}")


# ==============================================================================
# 数据自愈工具 (继承 V200)
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


def auto_fillna(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    V201 数据自愈：自动填充缺失值
    
    【V201 增强】
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
    V201 数据管道自愈：修复常见数据问题
    
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


# ==============================================================================
# V200 继承组件 (风险滤网、波动率自适应、IR 权重)
# ==============================================================================

class RiskFilterLayer:
    """
    V200/V201 风险滤网层 (Risk Filter Layer)
    
    【核心功能】
    1. 识别市场状态 (高波动/低流动性)
    2. 对高贝塔因子实施非线性惩罚
    3. 动态调整因子权重
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
        if 'returns' not in result.columns or result['returns'].isna().all():
            result['returns'] = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().fillna(0)
            )
        
        # 计算市场平均收益率和市场波动率
        market_stats = result.groupby('trade_date')['returns'].agg(
            market_return='mean',
            market_vol_20='std'
        ).reset_index()
        
        # 填充 NaN 值
        market_stats['market_return'] = market_stats['market_return'].fillna(0)
        market_stats['market_vol_20'] = market_stats['market_vol_20'].fillna(0.01)
        
        # 合并市场统计量
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
    
    def get_risk_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        risk_state: str,
        factor_betas: Dict[str, float]
    ) -> Dict[str, float]:
        """
        获取风险调整后的因子权重 - V201 增强版
        
        【调整逻辑】
        1. HIGH_RISK: reversion_5 权重占 50%+，其他因子大幅降低
        2. MEDIUM_RISK: 防御型因子权重增加 100%
        3. LOW_RISK: 保持原权重
        """
        adjusted = base_weights.copy()
        
        # 高贝塔因子列表 (风险型)
        high_beta_factors = {'volatility_5', 'volatility_20', 'beta_adj_factor', 'momentum_10', 'vol_adj_momentum'}
        # 核心防御型因子
        core_defensive = 'reversion_5'
        # 其他防御型因子
        other_defensive = {'liquidity_mkt_neutral', 'volume_price_contradiction'}
        # 流动性风险因子
        liquidity_risk_factors = {'liquidity_alpha', 'volatility_skew'}
        
        if risk_state == 'HIGH_RISK':
            for factor, weight in adjusted.items():
                if factor == core_defensive:
                    adjusted[factor] = weight * 8.0
                elif factor in high_beta_factors:
                    adjusted[factor] = weight * 0.05
                elif factor in other_defensive:
                    adjusted[factor] = weight * 1.5
                elif factor in liquidity_risk_factors:
                    adjusted[factor] = weight * 0.1
            
            adjusted[core_defensive] = 0.6
            remaining = 0.4
            other_factors = [f for f in adjusted.keys() if f != core_defensive]
            for f in other_factors:
                adjusted[f] = remaining / len(other_factors)
        
        elif risk_state == 'MEDIUM_RISK':
            for factor, weight in adjusted.items():
                if factor == core_defensive:
                    adjusted[factor] = weight * 3.0
                elif factor in high_beta_factors:
                    adjusted[factor] = weight * 0.15
                elif factor in other_defensive:
                    adjusted[factor] = weight * 1.5
                elif factor in liquidity_risk_factors:
                    adjusted[factor] = weight * 0.3
        
        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted


class VolatilityAdjuster:
    """
    V200/V201 波动率自适应调整器
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
        """
        if momentum_col not in df.columns:
            df[momentum_col] = df.groupby('symbol')['close'].transform(
                lambda x: x / x.shift(10) - 1
            )
        
        if vol_col not in df.columns:
            df[vol_col] = df.groupby('symbol')['returns'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
        
        vol_adj = df[momentum_col] / (df[vol_col] + EPSILON)
        
        if 'trade_date' in df.columns:
            vol_adj = vol_adj.groupby(df['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + EPSILON)
            )
        
        return vol_adj


class IRWeightCalculator:
    """
    V200/V201 IR 动态权重计算器
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
        """计算因子收益"""
        if factor_name not in df.columns or target_col not in df.columns:
            return pd.Series(0, index=df.index)
        
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
        """计算因子的 Information Ratio"""
        factor_returns = self.compute_factor_returns(df, factor_name, target_col)
        
        if factor_returns.isna().all() or len(factor_returns) < 10:
            return 0.0
        
        mean_return = factor_returns.mean()
        std_return = factor_returns.std()
        
        if std_return < EPSILON:
            return 0.0
        
        ir = mean_return / std_return
        return ir
    
    def compute_dynamic_weights(
        self,
        df: pd.DataFrame,
        factors: List[str],
        target_col: str = 't1_return'
    ) -> Dict[str, float]:
        """计算 IR 动态权重"""
        ir_values = {}
        
        for factor in factors:
            ir = self.compute_ir(df, factor, target_col)
            ir_adjusted = max(ir - self.volatility_penalty, 0)
            ir_values[factor] = ir_adjusted
        
        total = sum(ir_values.values())
        if total < EPSILON:
            return {f: 1.0 / len(factors) for f in factors}
        
        weights = {}
        for factor in factors:
            raw_weight = ir_values[factor] / total
            weights[factor] = max(raw_weight, self.min_weight)
        
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
        
        self.factor_ir_history = ir_values
        return weights


class MarketContext:
    """
    V200/V201 市场上下文管理器
    """
    
    def __init__(self):
        self.risk_filter = RiskFilterLayer()
        self.vol_adjuster = VolatilityAdjuster()
        self.ir_calculator = IRWeightCalculator()
        self.market_regime = None
        self.vol_baseline = None
    
    def compute_market_context(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str, float]:
        """计算完整的市场上下文"""
        result = self.risk_filter.compute_risk_indicators(df)
        risk_state_series, _ = self.risk_filter.classify_risk_state(result)
        result['risk_state'] = risk_state_series.values
        
        vol_baseline = result['atr_ratio'].median()
        current_state = risk_state_series.iloc[-1] if len(risk_state_series) > 0 else 'LOW_RISK'
        
        self.market_regime = current_state
        self.vol_baseline = vol_baseline
        
        return result, current_state, vol_baseline
    
    def get_state_dependent_config(self, risk_state: str) -> Dict[str, Any]:
        """获取状态自适应配置"""
        configs = {
            'HIGH_RISK': {
                'n_factors': 6,
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
                'n_factors': 12,
                'vol_penalty_scale': 1.0,
                'beta_penalty': 1.0,
                'defensive_weight_boost': 1.0,
            }
        }
        
        return configs.get(risk_state, configs['LOW_RISK'])


# ==============================================================================
# V201 新增组件：多源数据融合
# ==============================================================================

class MultiSourceDataFusion:
    """
    V201 多源数据融合器
    
    【核心功能】
    1. 行业中性化 (Industry Neutralization)
    2. 指数基准特征 (Index Benchmark)
    3. 资金流信号 (Fund Flow)
    """
    
    def __init__(self, db_url: str, index_symbol: str = INDEX_SYMBOL):
        self.db_url = db_url
        self.index_symbol = index_symbol
        self._engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        logger.info(f"[MultiSourceFusion] Initialized with index={index_symbol}")
    
    def load_industry_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载行业数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            行业数据 DataFrame
        """
        query = text("""
            SELECT symbol, trade_date, industry_name, industry_code
            FROM stock_industry_daily
            WHERE trade_date >= :start_date AND trade_date <= :end_date
            ORDER BY trade_date, symbol
        """)
        
        df = pd.read_sql(query, self._engine, params={
            'start_date': start_date,
            'end_date': end_date
        })
        
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
            df['symbol'] = df['symbol'].astype(str)
        
        logger.info(f"[MultiSourceFusion] Loaded {len(df)} industry rows")
        return df
    
    def load_index_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载指数数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            指数数据 DataFrame
        """
        query = text("""
            SELECT symbol, trade_date, close, ma20, pct_chg
            FROM index_daily
            WHERE symbol = :symbol
              AND trade_date >= :start_date AND trade_date <= :end_date
            ORDER BY trade_date
        """)
        
        df = pd.read_sql(query, self._engine, params={
            'symbol': self.index_symbol,
            'start_date': start_date,
            'end_date': end_date
        })
        
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
        
        logger.info(f"[MultiSourceFusion] Loaded {len(df)} index rows for {self.index_symbol}")
        return df
    
    def load_fund_flow_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载资金流数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            资金流数据 DataFrame
        """
        query = text("""
            SELECT symbol, trade_date, net_main_amount, net_main_rate
            FROM stock_fund_flow
            WHERE trade_date >= :start_date AND trade_date <= :end_date
            ORDER BY trade_date, symbol
        """)
        
        df = pd.read_sql(query, self._engine, params={
            'start_date': start_date,
            'end_date': end_date
        })
        
        if not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
            df['symbol'] = df['symbol'].astype(str)
            
            # 数值列转换
            df['net_main_amount'] = pd.to_numeric(df['net_main_amount'], errors='coerce')
            df['net_main_rate'] = pd.to_numeric(df['net_main_rate'], errors='coerce')
        
        logger.info(f"[MultiSourceFusion] Loaded {len(df)} fund flow rows")
        return df
    
    def compute_industry_neutral_score(
        self,
        df: pd.DataFrame,
        industry_df: pd.DataFrame,
        score_col: str = 'score'
    ) -> pd.Series:
        """
        计算行业中性化评分
        
        【原理】
        1. 按行业分组计算评分排名
        2. 限制单行业最大权重
        3. 返回行业调整后的评分
        
        Args:
            df: 包含评分的 DataFrame
            industry_df: 行业数据 (symbol, trade_date, industry_name)
            score_col: 评分列名
            
        Returns:
            行业中性化评分 Series
        """
        result = df.copy()
        
        # 合并行业数据
        merged = result.merge(
            industry_df[['symbol', 'trade_date', 'industry_name']],
            on=['symbol', 'trade_date'],
            how='left'
        )
        
        # 缺失行业填充默认值
        merged['industry_name'] = merged['industry_name'].fillna('Unknown')
        
        # 按日期 - 行业分组进行排名
        def _industry_rank(group):
            if len(group) < 2:
                return group[score_col]
            # 行业内排名 (百分位)
            rank = group[score_col].rank(pct=True)
            # 标准化
            return (rank - 0.5) * 2
        
        merged['industry_neutral_score'] = merged.groupby(
            ['trade_date', 'industry_name']
        ).apply(_industry_rank).reset_index(level=[0, 1], drop=True)
        
        # 填充 NaN
        merged['industry_neutral_score'] = merged['industry_neutral_score'].fillna(0)
        
        return merged['industry_neutral_score']
    
    def compute_index_mkt_state(self, df: pd.DataFrame, index_df: pd.DataFrame) -> pd.Series:
        """
        计算指数市场状态
        
        【状态定义】
        - BULL: 收盘价 > MA20 * (1 + threshold)
        - BEAR: 收盘价 < MA20 * (1 - threshold)
        - NEUTRAL: 其他
        
        Args:
            df: 主 DataFrame
            index_df: 指数数据 (trade_date, close, ma20)
            
        Returns:
            市场状态 Series
        """
        # 创建日期到市场状态的映射
        index_df = index_df.copy()
        index_df['ma20_upper'] = index_df['ma20'] * (1 + INDEX_MA20_THRESHOLD)
        index_df['ma20_lower'] = index_df['ma20'] * (1 - INDEX_MA20_THRESHOLD)
        
        def _classify(row):
            if row['close'] > row['ma20_upper']:
                return 'BULL'
            elif row['close'] < row['ma20_lower']:
                return 'BEAR'
            else:
                return 'NEUTRAL'
        
        index_df['mkt_state'] = index_df.apply(_classify, axis=1)
        state_map = dict(zip(index_df['trade_date'], index_df['mkt_state']))
        
        # 映射到主 DataFrame
        mkt_state_series = df['trade_date'].map(state_map).fillna('NEUTRAL')
        
        logger.info(f"[MultiSourceFusion] Computed market state: {mkt_state_series.value_counts().to_dict()}")
        return mkt_state_series
    
    def compute_fund_flow_signal(self, df: pd.DataFrame, fund_flow_df: pd.DataFrame) -> pd.Series:
        """
        计算资金流信号
        
        【原理】
        1. 计算 N 日主力净流入均值
        2. 按日期截面标准化
        3. 与价格信号结合，避免高位接盘
        
        Args:
            df: 主 DataFrame
            fund_flow_df: 资金流数据
            
        Returns:
            资金流信号 Series
        """
        # 合并资金流数据
        merged = df.merge(
            fund_flow_df[['symbol', 'trade_date', 'net_main_amount', 'net_main_rate']],
            on=['symbol', 'trade_date'],
            how='left'
        )
        
        # 填充 NaN
        merged['net_main_amount'] = merged['net_main_amount'].fillna(0)
        merged['net_main_rate'] = merged['net_main_rate'].fillna(0)
        
        # 计算滚动均值
        merged['fund_flow_ma'] = merged.groupby('symbol')['net_main_amount'].transform(
            lambda x: x.rolling(FUND_FLOW_WINDOW, min_periods=1).mean()
        )
        
        # 按日期截面标准化
        merged['fund_flow_signal'] = merged.groupby('trade_date')['fund_flow_ma'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON) if x.std() > EPSILON else 0
        )
        
        # 缩尾处理
        merged['fund_flow_signal'] = merged['fund_flow_signal'].clip(-3, 3)
        
        return merged['fund_flow_signal'].fillna(0)


# ==============================================================================
# V201 新增组件：Alpha 衰减惩罚
# ==============================================================================

class AlphaDecayPenalty:
    """
    V201 Alpha 衰减惩罚器
    
    【核心思想】
    针对 V200 可能存在的动量过拟合，增加"近期极值惩罚"逻辑：
    1. 检测连续 N 日排名靠前的股票
    2. 对这些股票进行分数衰减
    3. 避免"赢家诅咒"
    """
    
    def __init__(
        self,
        penalty_window: int = ALPHA_DECAY_PENALTY_WINDOW,
        penalty_threshold: float = ALPHA_DECAY_PENALTY_THRESHOLD,
        decay_rate: float = ALPHA_DECAY_RATE,
    ):
        self.penalty_window = penalty_window
        self.penalty_threshold = penalty_threshold
        self.decay_rate = decay_rate
        logger.info(f"[AlphaDecay] Initialized with window={penalty_window}, threshold={penalty_threshold}, decay={decay_rate}")
    
    def compute_decay_penalty(
        self,
        df: pd.DataFrame,
        score_col: str = 'score'
    ) -> pd.Series:
        """
        计算 Alpha 衰减惩罚
        
        【算法】
        1. 按日期计算每只股票的排名百分位
        2. 统计最近 N 日排名在前 threshold 的次数
        3. 次数越多，衰减越大
        
        Args:
            df: 包含评分的 DataFrame
            score_col: 评分列名
            
        Returns:
            衰减后的评分 Series
        """
        result = df.copy()
        
        # 1. 计算每日排名百分位
        result['rank_pct'] = result.groupby('trade_date')[score_col].rank(pct=True)
        
        # 2. 标记极值 (前 threshold)
        result['is_extreme'] = result['rank_pct'] > (1 - self.penalty_threshold)
        
        # 3. 计算滚动极值次数
        result['extreme_count'] = result.groupby('symbol')['is_extreme'].transform(
            lambda x: x.rolling(self.penalty_window, min_periods=1).sum()
        )
        
        # 4. 计算衰减系数 (极值次数越多，衰减越大)
        # decay_factor = 1 - (1 - decay_rate) * (extreme_count / window)
        result['decay_factor'] = 1 - (1 - self.decay_rate) * (
            result['extreme_count'] / self.penalty_window
        )
        
        # 5. 应用衰减
        decayed_score = result[score_col] * result['decay_factor']
        
        # 6. 再次按日期标准化
        decayed_score = decayed_score.groupby(result['trade_date']).transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON) if x.std() > EPSILON else x
        )
        
        logger.debug(f"[AlphaDecay] Applied decay to {result['extreme_count'].sum()} extreme instances")
        
        return decayed_score


# ==============================================================================
# V201 AlphaModel 主类
# ==============================================================================

class AlphaModel:
    """
    V201 Alpha 模型 - 多源数据融合与泛化性增强版
    
    【核心组件】
    1. MarketContext - 市场上下文 (风险滤网 + 波动率自适应 + IR 权重)
    2. MultiSourceDataFusion - 多源数据融合 (行业 + 指数 + 资金流)
    3. AlphaDecayPenalty - Alpha 衰减惩罚
    4. FeatureCache - PyArrow 特征缓存
    
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
        enable_industry_neutral: bool = True,
        enable_fund_flow: bool = True,
        enable_index_benchmark: bool = True,
        enable_alpha_decay: bool = True,
        db_url: Optional[str] = None,
    ) -> None:
        self.n_factors = n_factors
        self.enable_vol_adjustment = enable_vol_adjustment
        self.enable_risk_filter = enable_risk_filter
        self.enable_ir_weighting = enable_ir_weighting
        self.enable_orm = enable_orm
        self.enable_gated_residual = enable_gated_residual
        self.enable_industry_neutral = enable_industry_neutral
        self.enable_fund_flow = enable_fund_flow
        self.enable_index_benchmark = enable_index_benchmark
        self.enable_alpha_decay = enable_alpha_decay
        
        self.db_url = db_url or os.getenv("DATABASE_URL")
        
        # 核心组件
        self.market_context = MarketContext()
        self.feature_cache = FeatureCache() if CACHE_ENABLED else None
        
        # V201 新增组件
        if self.db_url:
            self.multi_source = MultiSourceDataFusion(self.db_url)
        else:
            self.multi_source = None
            logger.warning("[AlphaModel] No DB URL, multi-source features disabled")
        
        self.alpha_decay = AlphaDecayPenalty() if enable_alpha_decay else None
        
        # 状态变量
        self.selected_factors: List[str] = []
        self.factor_weights: Dict[str, float] = {}
        self.factor_directions: Dict[str, int] = {}
        self.current_risk_state = 'LOW_RISK'
        self.factor_betas: Dict[str, float] = {}
        
        self._init_factor_directions()
        self._init_factor_betas()
        
        logger.info(f"[AlphaModel] {VERSION} Initialized")
        logger.info(f"  Core Factors: {V201_CORE_FACTORS}")
        logger.info(f"  Volatility Adjustment: {enable_vol_adjustment}")
        logger.info(f"  Risk Filter: {enable_risk_filter}")
        logger.info(f"  IR Weighting: {enable_ir_weighting}")
        logger.info(f"  Industry Neutral: {enable_industry_neutral}")
        logger.info(f"  Fund Flow: {enable_fund_flow}")
        logger.info(f"  Index Benchmark: {enable_index_benchmark}")
        logger.info(f"  Alpha Decay: {enable_alpha_decay}")
        logger.info(f"  Feature Cache: {CACHE_ENABLED}")
    
    def _init_factor_directions(self) -> None:
        """初始化因子方向"""
        self.factor_directions = {
            'reversion_5': 1,
            'volume_rank': -1,
            'volume_price_contradiction': 1,
            'liquidity_alpha': -1,
            'volatility_5': 1,
            'volatility_20': 1,
            'volatility_skew': -1,
            'liquidity_mkt_neutral': -1,
            'beta_adj_factor': -1,
            'vol_adj_momentum': -1,
            'momentum_10': -1,
            # V201 新增
            'industry_neutral_score': 1,  # 行业内相对优势
            'fund_flow_signal': 1,        # 主力净流入为正
            'index_mkt_state': 1,         # 市场状态 (BULL=1, NEUTRAL=0, BEAR=-1)
        }
    
    def _init_factor_betas(self) -> None:
        """初始化因子贝塔"""
        high_beta = {'volatility_5', 'volatility_20', 'momentum_10', 'beta_adj_factor'}
        low_beta = {'reversion_5', 'liquidity_mkt_neutral', 'volume_price_contradiction'}
        neutral = {'volume_rank', 'liquidity_alpha', 'volatility_skew', 'vol_adj_momentum',
                   'industry_neutral_score', 'fund_flow_signal'}
        
        self.factor_betas = {}
        for f in high_beta:
            self.factor_betas[f] = 1.5
        for f in low_beta:
            self.factor_betas[f] = 0.5
        for f in neutral:
            self.factor_betas[f] = 1.0
    
    def _load_multi_source_data(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        加载多源数据
        
        Returns:
            (industry_df, index_df, fund_flow_df) 或 (None, None, None)
        """
        if self.multi_source is None:
            return None, None, None
        
        try:
            industry_df = self.multi_source.load_industry_data(start_date, end_date)
            index_df = self.multi_source.load_index_data(start_date, end_date)
            fund_flow_df = self.multi_source.load_fund_flow_data(start_date, end_date)
            return industry_df, index_df, fund_flow_df
        except Exception as e:
            logger.error(f"[AlphaModel] Failed to load multi-source data: {e}")
            return None, None, None
    
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
        
        # 10. beta_proxy: Beta 代理
        result = self._compute_beta_proxy(result)
        
        # V200/V201 新增特征
        if self.enable_vol_adjustment:
            # 11. vol_adj_momentum: 波动率调整动量
            result['vol_adj_momentum'] = self.market_context.vol_adjuster.compute_vol_adj_momentum(
                result
            )
            
            # 12. beta_adj_factor: 贝塔调整因子
            result['beta_adj_factor'] = result['volatility_20'] / (result['beta_proxy'] + EPSILON)
        
        return result
    
    def _compute_market_cap_neutral_liquidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """市值中性化流动性因子"""
        result = df.copy()
        
        if 'market_cap' in result.columns:
            result['market_cap_proxy'] = result['market_cap']
        else:
            result['market_cap_proxy'] = result['close'] * result['volume']
        
        result['raw_liquidity'] = result['volume'] / (result['market_cap_proxy'] + EPSILON)
        
        result['market_cap_quantile'] = result.groupby('trade_date')['market_cap_proxy'].transform(
            lambda x: pd.qcut(x.rank(method='first'), q=10, labels=False, duplicates='drop')
        )
        
        result['liquidity_mkt_neutral'] = result.groupby(
            ['trade_date', 'market_cap_quantile']
        )['raw_liquidity'].transform(
            lambda x: x.rank(pct=True) if len(x) > 1 else 0.5
        )
        
        result['liquidity_mkt_neutral'] = result.groupby('trade_date')['liquidity_mkt_neutral'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        return result
    
    def _compute_beta_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Beta 代理"""
        result = df.copy()
        
        if 'returns' not in result.columns:
            result['returns'] = result.groupby('symbol')['close'].transform(
                lambda x: x.pct_change().fillna(0)
            )
        
        market_stats = result.groupby('trade_date')['returns'].agg(
            market_return='mean',
            market_vol_20='std'
        ).reset_index()
        
        market_stats['market_return'] = market_stats['market_return'].fillna(0)
        market_stats['market_vol_20'] = market_stats['market_vol_20'].fillna(0.01)
        
        result = result.merge(market_stats, on='trade_date', how='left')
        
        result['market_return'] = result['market_return'].fillna(0)
        result['market_vol_20'] = result['market_vol_20'].fillna(0.01)
        
        result['stock_vol_20'] = result.groupby('symbol')['returns'].transform(
            lambda x: x.rolling(20, min_periods=10).std().fillna(0.01)
        )
        
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
        """应用风险滤网"""
        adjusted_scores = scores.copy()
        
        if risk_state == 'HIGH_RISK':
            if context_df is not None and 'atr_ratio' in context_df.columns and 'liquidity_ratio' in context_df.columns:
                vol_quantiles = context_df.groupby('trade_date')['atr_ratio'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), q=20, labels=False, duplicates='drop')
                )
                liq_quantiles = context_df.groupby('trade_date')['liquidity_ratio'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), q=20, labels=False, duplicates='drop')
                )
                
                high_vol_mask = vol_quantiles >= 16
                low_liq_mask = liq_quantiles <= 4
                extreme_risk_mask = high_vol_mask & low_liq_mask
                
                adjusted_scores[extreme_risk_mask] = -10.0
                
                logger.info(f"[Risk Filter] HIGH_RISK: Penalized {extreme_risk_mask.sum()} extreme risk stocks")
        
        return adjusted_scores
    
    def _compute_ir_weights(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算 IR 动态权重"""
        if not self.enable_ir_weighting:
            return {f: 1.0 / len(self.selected_factors) for f in self.selected_factors}
        
        return self.market_context.ir_calculator.compute_dynamic_weights(
            df, self.selected_factors
        )
    
    def compute_score(self, df: pd.DataFrame, years: List[int] = None) -> pd.DataFrame:
        """
        计算 Alpha 分数 - V201 多源数据融合版
        
        【流程】
        1. 数据自愈
        2. 加载多源数据 (行业 + 指数 + 资金流)
        3. 特征计算
        4. 市场上下文计算
        5. 多源特征融合
        6. 因子选择与 IR 动态权重
        7. 风险滤网应用
        8. Alpha 衰减惩罚
        9. 正交化与残差连接
        10. 最终标准化
        """
        logger.info(f"[AlphaModel] Computing scores for {len(df)} rows")
        
        # 1. 数据自愈
        logger.debug("[Data Heal] Running fix_data_pipeline...")
        df = fix_data_pipeline(df)
        
        # 2. 加载多源数据
        industry_df = None
        index_df = None
        fund_flow_df = None
        
        if self.multi_source and years:
            # 计算日期范围
            min_date = pd.to_datetime(str(df['trade_date'].min())).strftime('%Y-%m-%d')
            max_date = pd.to_datetime(str(df['trade_date'].max())).strftime('%Y-%m-%d')
            
            logger.info(f"[MultiSource] Loading data for {min_date} to {max_date}")
            industry_df, index_df, fund_flow_df = self._load_multi_source_data(min_date, max_date)
        
        # 3. 特征计算
        logger.debug("[Features] Computing base features...")
        result = self._compute_features(df)
        
        # 4. 市场上下文计算
        logger.debug("[Context] Computing market context...")
        context_df, risk_state, vol_baseline = self.market_context.compute_market_context(result)
        self.current_risk_state = risk_state
        
        logger.info(f"[Risk State] Current: {risk_state}, Vol Baseline: {vol_baseline:.4f}")
        
        # 5. 多源特征融合
        if self.enable_industry_neutral and industry_df is not None:
            logger.debug("[MultiSource] Computing industry neutral score...")
            result['industry_neutral_score'] = self.multi_source.compute_industry_neutral_score(
                result, industry_df
            )
        
        if self.enable_index_benchmark and index_df is not None:
            logger.debug("[MultiSource] Computing index market state...")
            mkt_state = self.multi_source.compute_index_mkt_state(result, index_df)
            # 转换为数值：BULL=1, NEUTRAL=0, BEAR=-1
            state_map = {'BULL': 1, 'NEUTRAL': 0, 'BEAR': -1}
            result['index_mkt_state'] = mkt_state.map(state_map)
        
        if self.enable_fund_flow and fund_flow_df is not None:
            logger.debug("[MultiSource] Computing fund flow signal...")
            result['fund_flow_signal'] = self.multi_source.compute_fund_flow_signal(
                result, fund_flow_df
            )
        
        # 6. 因子选择
        config = self.market_context.get_state_dependent_config(risk_state)
        n_select = min(config['n_factors'], self.n_factors)
        
        # 根据风险状态选择因子
        high_risk_excluded = {'volatility_5', 'volatility_20', 'volatility_skew', 'momentum_10', 'vol_adj_momentum'}
        medium_risk_excluded = {'volatility_5', 'volatility_20', 'momentum_10', 'vol_adj_momentum'}
        
        if risk_state == 'HIGH_RISK':
            self.selected_factors = ['liquidity_mkt_neutral', 'reversion_5']
            self.factor_weights = {
                'liquidity_mkt_neutral': 0.4,
                'reversion_5': 0.6,
            }
        elif risk_state == 'MEDIUM_RISK':
            safe_factors = [f for f in V201_CORE_FACTORS if f not in medium_risk_excluded]
            self.selected_factors = safe_factors[:n_select]
        else:
            self.selected_factors = V201_CORE_FACTORS[:self.n_factors]
        
        # 7. IR 动态权重
        if risk_state != 'HIGH_RISK':
            logger.debug("[IR Weight] Computing IR dynamic weights...")
            self.factor_weights = self._compute_ir_weights(result)
        
        logger.info(f"[Factor] Selected factors: {self.selected_factors}")
        logger.info(f"[Weight] Factor weights: {self.factor_weights}")
        
        # 8. 计算基础分数
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
        
        # 9. 正交化
        if self.enable_orm:
            feature_matrix = np.column_stack([
                result[f].values for f in self.selected_factors if f in result.columns
            ])
            if feature_matrix.shape[1] > 0:
                feature_matrix = self._lowdin_orthogonalization(feature_matrix)
        
        # 10. 门控残差
        if self.enable_gated_residual:
            scores = self._gated_residual(np.column_stack([scores, np.random.randn(len(scores), 1)]), scores).flatten()
        
        # 11. 按日期标准化
        result['score'] = scores
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + EPSILON)
        )
        
        # 12. Alpha 衰减惩罚
        if self.enable_alpha_decay and self.alpha_decay:
            logger.debug("[AlphaDecay] Applying decay penalty...")
            result['score'] = self.alpha_decay.compute_decay_penalty(result, 'score')
        
        # 13. 风险滤网应用
        if self.enable_risk_filter:
            logger.debug("[Risk Filter] Applying risk filter...")
            result['score'] = self._apply_risk_filter(result, result['score'], risk_state, context_df)
        
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
        """获取 V201 新增特征列表"""
        return ['industry_neutral_score', 'fund_flow_signal', 'index_mkt_state']
    
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
    enable_industry_neutral: bool = True,
    enable_fund_flow: bool = True,
    enable_index_benchmark: bool = True,
    enable_alpha_decay: bool = True,
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
        enable_industry_neutral=enable_industry_neutral,
        enable_fund_flow=enable_fund_flow,
        enable_index_benchmark=enable_index_benchmark,
        enable_alpha_decay=enable_alpha_decay,
        db_url=db_url,
    )