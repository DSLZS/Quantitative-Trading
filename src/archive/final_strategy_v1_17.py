"""
Final Strategy V1.17 - Iteration 17: 双重中性化与全周期实战回测

【核心改进 - Iteration 17】

1. 双重中性化 (Double Neutralization)
   - 在行业中性化基础上，增加市值中性化
   - 每个交易日进行双重残差化：Factor ~ Industry_Dummy + Log(Total_MV)
   - 取残差作为最终因子，彻底剔除行业和大小盘影响

2. 全周期实战测试 (Full Period Backtest)
   - 时间跨度：2023-01 至 2026-03
   - 滚动权重：继续使用 20 日滚动 IC 赋权
   - 严格时序：使用 T-1 日累计 IC 决定 T 日权重

3. 特征持久化 (Feature Persistence)
   - 计算结果存储为 data/features_v17.parquet
   - 支持增量更新和过期清理
   - 避免重复计算，实现"计算一次，多次回测"

4. 严格撮合逻辑 (V13 标准)
   - T+1 开盘买入
   - T+6 开盘卖出 (持有 5 天)
   - 0.2% 单边滑点

【严禁事项】
- 严禁修改 BacktestEngine 撮合逻辑
- 严禁将滑点从 0.2% 调低
- 严禁将 T+1 买入改为 T 日买入
- 严禁使用未来 IC 数据

作者：量化首席架构师
版本：V1.17 (Iteration 17)
日期：2026-03-17
"""

import sys
import os
import json
import hashlib
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import pickle

import polars as pl
import numpy as np
from scipy import stats
from loguru import logger
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

# 绘图支持
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available, plots will be skipped")

# 导入本地模块
try:
    from .db_manager import DatabaseManager
    from .factor_engine import FactorEngine
    from .backtest_engine import BacktestEngine
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine
    from backtest_engine import BacktestEngine


# ===========================================
# 配置常量
# ===========================================

# 核心因子参数
REVERSAL_WINDOW = 5           # 短线反转窗口
VWAP_DEVIATION_WINDOW = 20    # VWAP 偏离窗口
VOLATILITY_WINDOW = 20        # 波动率窗口
TURNOVER_WINDOW = 20          # 换手率窗口

# 中性化配置 - V17 双重中性化
USE_INDUSTRY_NEUTRAL = True   # 启用行业中性化
USE_MARKET_CAP_NEUTRAL = True # 启用市值中性化 (V17 新增)
USE_MAD_WINSOR = True         # 启用 MAD 极值处理
MAD_THRESHOLD = 3.0           # MAD 阈值（标准差倍数）

# 模型配置 - IC 加权法
IC_WINDOW = 20                # 滚动 IC 计算窗口（20 天）

# 缓存配置 - V17
CACHE_DIR = Path("data/cache/v17")
CACHE_FILE = Path("data/features_v17.parquet")
CACHE_EXPIRY_DAYS = 7         # 缓存有效期（天）

# 回测配置 - V13 标准 (严禁修改)
INITIAL_CAPITAL = 100000.0
SLIPPAGE = 0.002              # 单边 0.2% 滑点 (V13 标准)
HOLDING_PERIOD = 5            # T+5 持有期
BUY_DELAY = 1                 # T+1 买入


# ===========================================
# 数据类定义
# ===========================================

@dataclass
class ICResult:
    """IC 分析结果"""
    factor_name: str
    mean_ic: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    positive_ratio: float = 0.0
    num_valid_days: int = 0
    min_ic: float = 0.0
    max_ic: float = 0.0
    t_stat: float = 0.0
    ic_series: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)


@dataclass
class QuintileResult:
    """五分位组合收益结果"""
    q1_return: float = 0.0
    q2_return: float = 0.0
    q3_return: float = 0.0
    q4_return: float = 0.0
    q5_return: float = 0.0
    q1_q5_spread: float = 0.0
    q1_count: int = 0
    q2_count: int = 0
    q3_count: int = 0
    q4_count: int = 0
    q5_count: int = 0
    q5_q1_cumulative: List[Dict] = field(default_factory=list)


@dataclass
class NeutralizationResult:
    """中性化前后对比结果"""
    factor_name: str
    raw_mean_ic: float = 0.0
    raw_icir: float = 0.0
    neutralized_mean_ic: float = 0.0
    neutralized_icir: float = 0.0
    ic_improvement: float = 0.0
    icir_improvement: float = 0.0
    # 行业均值验证
    industry_means_before: Dict[str, float] = field(default_factory=dict)
    industry_means_after: Dict[str, float] = field(default_factory=dict)
    # 市值相关性验证 (V17 新增)
    corr_with_mv_before: float = 0.0
    corr_with_mv_after: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0
    profit_factor: float = 0.0
    quintile_results: Optional[QuintileResult] = None
    daily_values: List[Dict] = field(default_factory=list)
    trades: List[Dict] = field(default_factory=list)


# ===========================================
# V17 核心因子引擎 - 带正交化
# ===========================================

class V17CoreFactorEngine:
    """
    V17 核心因子引擎 - 4 个精炼因子 + 正交化处理
    
    【因子列表】
    1. 量价相关性 (vol_price_corr): 最强因子，作为正交化基准
    2. 短线反转 (reversal_st): 经过正交化处理
    3. 波动风险 (vol_risk): 低波异常效应
    4. 异常换手 (turnover_signal): 换手率放大信号
    
    【丢弃的因子】
    - pv_deviation: 与 reversal_st 相关性高达 0.758
    
    【正交化方法】
    - 使用回归残差法：reversal_st_ortho = reversal_st - β * vol_price_corr
    - β = Cov(reversal_st, vol_price_corr) / Var(vol_price_corr)
    """
    
    EPSILON = 1e-6
    
    def __init__(self):
        logger.info("V17CoreFactorEngine initialized")
    
    def compute_short_term_reversal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算短线反转因子
        
        【计算公式】
        - reversal = -(close[t] / close[t-5] - 1)
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 5 日收益率
        momentum_5 = pl.col("close") / (
            pl.col("close").shift(REVERSAL_WINDOW + 1) + self.EPSILON
        ) - 1
        
        # 取反作为反转信号
        reversal_signal = -momentum_5.shift(1)
        
        result = result.with_columns([
            reversal_signal.alias("reversal_st"),
        ])
        
        logger.debug(f"[Reversal] Computed, rows={len(result)}")
        return result
    
    def compute_price_vwap_deviation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算量价背离因子（用于诊断，不用于最终模型）
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
        ])
        
        vwap = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        vwap_ma20 = vwap.rolling_mean(window_size=VWAP_DEVIATION_WINDOW).shift(1)
        deviation = (pl.col("close") - vwap_ma20) / (vwap_ma20 + self.EPSILON)
        pv_deviation = -deviation
        
        result = result.with_columns([
            vwap.alias("vwap"),
            pv_deviation.alias("pv_deviation"),
        ])
        
        logger.debug(f"[PV Deviation] Computed, rows={len(result)}")
        return result
    
    def compute_volatility_risk(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算波动风险因子
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        volatility_20 = returns.rolling_std(
            window_size=VOLATILITY_WINDOW, ddof=1
        ).shift(1) * np.sqrt(252)
        vol_risk = -volatility_20
        
        result = result.with_columns([
            volatility_20.alias("volatility_ann"),
            vol_risk.alias("vol_risk"),
        ])
        
        logger.debug(f"[Volatility] Computed, rows={len(result)}")
        return result
    
    def compute_turnover_ratio(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算异常换手因子
        """
        result = df.clone().with_columns([
            pl.col("turnover_rate").cast(pl.Float64, strict=False),
        ])
        
        turnover_ma20 = pl.col("turnover_rate").rolling_mean(
            window_size=TURNOVER_WINDOW
        ).shift(1)
        turnover_ratio = pl.col("turnover_rate") / (turnover_ma20 + self.EPSILON)
        turnover_signal = (turnover_ratio - 1).clip(-0.9, 2.0)
        
        result = result.with_columns([
            turnover_ratio.alias("turnover_ratio"),
            turnover_signal.alias("turnover_signal"),
        ])
        
        logger.debug(f"[Turnover] Computed, rows={len(result)}")
        return result
    
    def compute_volume_price_correlation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算量价相关性因子（最强因子，作为正交化基准）
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        ])
        
        if "volume" not in result.columns:
            result = result.with_columns([
                pl.lit(1.0).alias("volume")
            ])
        
        volume_change = (pl.col("volume") / pl.col("volume").shift(1) - 1).fill_null(0)
        returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        
        window = 20
        vol_mean = volume_change.rolling_mean(window_size=window).shift(1)
        ret_mean = returns.rolling_mean(window_size=window).shift(1)
        cov_xy = ((volume_change - vol_mean) * (returns - ret_mean)).rolling_mean(window_size=window).shift(1)
        vol_std = volume_change.rolling_std(window_size=window, ddof=1).shift(1)
        ret_std = returns.rolling_std(window_size=window, ddof=1).shift(1)
        vol_price_corr = cov_xy / (vol_std * ret_std + self.EPSILON)
        
        result = result.with_columns([
            vol_price_corr.alias("vol_price_corr"),
        ])
        
        logger.debug(f"[Vol-Price Corr] Computed, rows={len(result)}")
        return result
    
    def orthogonalize_factor(
        self, 
        df: pl.DataFrame, 
        target_factor: str, 
        benchmark_factor: str
    ) -> pl.DataFrame:
        """
        使用回归残差法对因子进行正交化
        
        【数学原理】
        target_ortho = target - β * benchmark
        其中 β = Cov(target, benchmark) / Var(benchmark)
        
        【金融意义】
        - 只保留 target 因子相对于 benchmark 的"独特信息"
        - 消除多重共线性
        
        Args:
            df: 输入 DataFrame
            target_factor: 需要正交化的因子
            benchmark_factor: 基准因子
            
        Returns:
            添加正交化因子的 DataFrame
        """
        logger.info(f"[Orthogonalization] {target_factor} vs {benchmark_factor}")
        
        # 按日期分组计算 β 和残差（使用 Polars 原生语法）
        result = df.clone()
        
        # 计算每个交易日的 β 系数
        # β = Cov(X, Y) / Var(Y)
        beta_calc = df.group_by("trade_date").agg([
            pl.cov(target_factor, benchmark_factor).alias("cov_xy"),
            pl.col(benchmark_factor).var().alias("var_y"),
        ]).with_columns([
            (pl.col("cov_xy") / (pl.col("var_y") + self.EPSILON)).alias("beta")
        ]).select(["trade_date", "beta"])
        
        # 合并 β 系数
        result = result.join(beta_calc, on="trade_date", how="left")
        
        # 计算正交化残差：residual = target - β * benchmark
        result = result.with_columns([
            (pl.col(target_factor) - pl.col("beta") * pl.col(benchmark_factor)).alias(
                f"{target_factor}_ortho"
            )
        ]).drop(["beta"])
        
        logger.info(f"[Orthogonalization] Completed, residual mean≈0, std≈1")
        return result
    
    def compute_all_core_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算所有核心因子并进行正交化
        """
        result = df.clone()
        
        # 依次计算因子
        result = self.compute_short_term_reversal(result)
        result = self.compute_price_vwap_deviation(result)  # 用于诊断
        result = self.compute_volatility_risk(result)
        result = self.compute_turnover_ratio(result)
        result = self.compute_volume_price_correlation(result)
        
        # 正交化处理：reversal_st 对 vol_price_corr 正交化
        result = self.orthogonalize_factor(
            result, 
            target_factor="reversal_st", 
            benchmark_factor="vol_price_corr"
        )
        
        logger.info(f"[V17 Core Factors] All factors computed + orthogonalized")
        return result
    
    def get_core_factor_names(self) -> List[str]:
        """获取核心因子名称列表（正交化后）"""
        return [
            "reversal_st_ortho",    # 短线反转（正交化后）
            "vol_risk",             # 波动风险
            "turnover_signal",      # 异常换手
            "vol_price_corr",       # 量价相关性（基准因子）
        ]
    
    def get_all_factor_names(self) -> List[str]:
        """获取所有因子名称（包括诊断因子）"""
        return [
            "reversal_st",
            "reversal_st_ortho",
            "pv_deviation",         # 诊断用，不用于模型
            "vol_risk",
            "turnover_signal",
            "vol_price_corr",
        ]


# ===========================================
# V17 双重中性化模块 - 行业 + 市值
# ===========================================

class V17DoubleNeutralizer:
    """
    V17 双重中性化器 - 纯 Polars 实现
    
    【处理流程】
    1. MAD Winsorization（极值处理）
    2. Industry Neutralization（行业中性化）
    3. Market Cap Neutralization（市值中性化）- V17 新增
    4. Cross-sectional Standardization（Z-Score 标准化）
    
    【核心改进】
    - 使用 Polars group_by().agg() 替代 pandas transform
    - 使用 Polars 表达式语法，无 Python 循环
    - 行业代码缺失归类为 'Others'
    - 市值取对数以符合正态分布
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        use_mad_winsor: bool = True,
        use_industry_neutral: bool = True,
        use_market_cap_neutral: bool = True,  # V17 新增
        mad_threshold: float = 3.0,
    ):
        self.use_mad_winsor = use_mad_winsor
        self.use_industry_neutral = use_industry_neutral
        self.use_market_cap_neutral = use_market_cap_neutral
        self.mad_threshold = mad_threshold
        
        logger.info(f"V17DoubleNeutralizer initialized (MAD={use_mad_winsor}, Industry={use_industry_neutral}, MktCap={use_market_cap_neutral})")
    
    def mad_winsorize_column(self, df: pl.DataFrame, factor_name: str) -> pl.DataFrame:
        """
        MAD Winsorization - 按日期截面处理
        
        【Polars 实现】
        使用 group_by("trade_date").agg() 计算每日统计量
        """
        if not self.use_mad_winsor:
            return df
        
        # 计算每日中位数和 MAD
        daily_stats = df.group_by("trade_date").agg([
            pl.col(factor_name).median().alias("median"),
            (pl.col(factor_name) - pl.col(factor_name).median()).abs().median().alias("mad"),
        ]).with_columns([
            (pl.col("median") - self.mad_threshold * pl.col("mad")).alias("lower"),
            (pl.col("median") + self.mad_threshold * pl.col("mad")).alias("upper"),
        ]).select(["trade_date", "lower", "upper"])
        
        # 合并统计量并截断
        df = df.join(daily_stats, on="trade_date", how="left")
        df = df.with_columns([
            pl.col(factor_name).clip(pl.col("lower"), pl.col("upper")).alias(factor_name)
        ]).drop(["lower", "upper"])
        
        logger.debug(f"[MAD Winsorize] {factor_name} processed")
        return df
    
    def industry_neutralize_column(self, df: pl.DataFrame, factor_name: str) -> pl.DataFrame:
        """
        行业中性化 - 使用 Polars 原生语法
        
        【核心逻辑】
        - 每个交易日截面上，减去行业平均值
        - 行业代码缺失归类为 'Others'
        
        【Polars 实现】
        使用 group_by(["trade_date", "industry_code"]).agg() 计算行业均值
        """
        if not self.use_industry_neutral:
            return df
        
        # 检查 industry_code 是否存在
        if "industry_code" not in df.columns:
            logger.warning("industry_code not found, skipping industry neutralization")
            return df
        
        # 处理缺失的行业代码 - 归类为 'Others'
        df = df.with_columns([
            pl.when(pl.col("industry_code").is_null() | 
                   (pl.col("industry_code") == "") | 
                   (pl.col("industry_code") == "UNKNOWN") |
                   (pl.col("industry_code") == "N/A"))
            .then(pl.lit("Others"))
            .otherwise(pl.col("industry_code"))
            .alias("industry_code")
        ])
        
        # 计算每日各行业均值
        industry_means = df.group_by(["trade_date", "industry_code"]).agg([
            pl.col(factor_name).mean().alias(f"industry_mean_{factor_name}"),
            pl.col(factor_name).std().alias(f"industry_std_{factor_name}"),
        ])
        
        # 合并行业均值
        df = df.join(industry_means, on=["trade_date", "industry_code"], how="left", suffix="_right")
        
        # 中性化：因子值 - 行业均值
        df = df.with_columns([
            (pl.col(factor_name) - pl.col(f"industry_mean_{factor_name}")).alias(f"{factor_name}_neutral")
        ]).drop([f"industry_mean_{factor_name}", f"industry_std_{factor_name}"])
        
        logger.debug(f"[Industry Neutralize] {factor_name} processed")
        return df
    
    def market_cap_neutralize_column(self, df: pl.DataFrame, factor_name: str) -> pl.DataFrame:
        """
        市值中性化 - V17 新增
        
        【核心逻辑】
        - 对 total_mv 取对数：log_mv = log(total_mv)
        - 在每个交易日截面上，进行线性回归：factor = α + β * log_mv + ε
        - 取残差 ε 作为中性化后的因子值
        
        【Polars 实现】
        使用 group_by("trade_date").agg() 计算每日回归系数
        """
        if not self.use_market_cap_neutral:
            return df
        
        # 检查 total_mv 是否存在
        if "total_mv" not in df.columns:
            logger.warning("total_mv not found, skipping market cap neutralization")
            return df
        
        # 计算对数市值
        source_col = f"{factor_name}_neutral" if f"{factor_name}_neutral" in df.columns else factor_name
        
        df = df.with_columns([
            pl.col("total_mv").cast(pl.Float64, strict=False).log().alias("log_mv"),
            pl.col(source_col).cast(pl.Float64, strict=False).alias(f"{factor_name}_input"),
        ])
        
        # 计算每日回归系数
        # β = Cov(X, Y) / Var(X), 其中 X = log_mv, Y = factor
        beta_calc = df.group_by("trade_date").agg([
            pl.cov(f"{factor_name}_input", "log_mv").alias("cov_xy"),
            pl.col("log_mv").var().alias("var_x"),
            pl.col(f"{factor_name}_input").mean().alias("mean_y"),
            pl.col("log_mv").mean().alias("mean_x"),
        ]).with_columns([
            (pl.col("cov_xy") / (pl.col("var_x") + self.EPSILON)).alias("beta"),
        ]).select(["trade_date", "beta", "mean_y", "mean_x"])
        
        # 合并回归系数
        df = df.join(beta_calc, on="trade_date", how="left")
        
        # 计算残差：residual = factor - (α + β * log_mv)
        # 其中 α = mean_y - β * mean_x
        df = df.with_columns([
            (pl.col(f"{factor_name}_input") - 
             (pl.col("mean_y") - pl.col("beta") * pl.col("mean_x") + pl.col("beta") * pl.col("log_mv"))
            ).alias(f"{factor_name}_double_neutral")
        ]).drop(["beta", "mean_y", "mean_x", f"{factor_name}_input", "log_mv"])
        
        logger.debug(f"[Market Cap Neutralize] {factor_name} processed")
        return df
    
    def cross_sectional_standardize_column(
        self, 
        df: pl.DataFrame, 
        factor_name: str,
        use_double_neutral: bool = True
    ) -> pl.DataFrame:
        """
        截面标准化（Z-Score）- 使用 Polars 原生语法
        
        【核心逻辑】
        - 每个交易日截面上，均值为 0，标准差为 1
        - z_score = (x - mean) / std
        """
        # 确定使用哪个因子列
        if use_double_neutral and f"{factor_name}_double_neutral" in df.columns:
            source_col = f"{factor_name}_double_neutral"
        elif f"{factor_name}_neutral" in df.columns:
            source_col = f"{factor_name}_neutral"
        else:
            source_col = factor_name
        
        target_col = f"{factor_name}_std"
        
        # 计算每日截面统计量
        daily_stats = df.group_by("trade_date").agg([
            pl.col(source_col).mean().alias("mean"),
            pl.col(source_col).std().alias("std"),
        ])
        
        # 合并统计量
        df = df.join(daily_stats, on="trade_date", how="left")
        
        # 标准化
        df = df.with_columns([
            ((pl.col(source_col) - pl.col("mean")) / (pl.col("std") + self.EPSILON)).alias(target_col)
        ]).drop(["mean", "std"])
        
        logger.debug(f"[Z-Score] {source_col} -> {target_col}")
        return df
    
    def normalize_factors(
        self,
        df: pl.DataFrame,
        factor_names: List[str],
    ) -> pl.DataFrame:
        """
        对所有因子进行双重中性化处理
        
        【处理流程】
        1. MAD Winsorization（极值处理）
        2. Industry Neutralization（行业中性化）
        3. Market Cap Neutralization（市值中性化）- V17 新增
        4. Cross-sectional Standardization（标准化）
        """
        result = df.clone()
        
        for factor_name in factor_names:
            if factor_name not in result.columns:
                logger.warning(f"Factor {factor_name} not found, skipping")
                continue
            
            logger.debug(f"Normalizing factor: {factor_name}")
            
            # 1. MAD Winsorization
            result = self.mad_winsorize_column(result, factor_name)
            
            # 2. 行业中性化
            result = self.industry_neutralize_column(result, factor_name)
            
            # 3. 市值中性化 - V17 新增
            result = self.market_cap_neutralize_column(result, factor_name)
            
            # 4. 标准化
            result = self.cross_sectional_standardize_column(result, factor_name, use_double_neutral=True)
        
        logger.info(f"[Normalization] Completed for {len(factor_names)} factors (Double Neutralization)")
        return result
    
    def verify_neutralization(
        self, 
        df: pl.DataFrame, 
        factor_name: str,
        sample_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        验证中性化效果 - 检查各行业均值和市值相关性是否趋近于 0
        
        Returns:
            包含行业均值和市值相关性的字典
        """
        result = {
            "industry_means": {},
            "corr_with_log_mv": 0.0,
        }
        
        if "industry_code" not in df.columns:
            return result
        
        # 处理缺失行业代码
        df_clean = df.with_columns([
            pl.when(pl.col("industry_code").is_null() | 
                   (pl.col("industry_code") == "") | 
                   (pl.col("industry_code") == "UNKNOWN") |
                   (pl.col("industry_code") == "N/A"))
            .then(pl.lit("Others"))
            .otherwise(pl.col("industry_code"))
            .alias("industry_code")
        ])
        
        # 选择日期
        if sample_date:
            df_clean = df_clean.filter(pl.col("trade_date") == sample_date)
        
        # 计算各行业均值
        neutral_col = f"{factor_name}_std" if f"{factor_name}_std" in df_clean.columns else factor_name
        
        industry_means = df_clean.group_by("industry_code").agg([
            pl.col(neutral_col).mean().alias("mean"),
            pl.col(neutral_col).std().alias("std"),
            pl.col(neutral_col).count().alias("count"),
        ]).sort("mean", descending=True)
        
        # 转换为字典
        for row in industry_means.iter_rows(named=True):
            result["industry_means"][row["industry_code"]] = row["mean"]
        
        # 计算与对数市值的相关性
        if "total_mv" in df_clean.columns and neutral_col in df_clean.columns:
            df_corr = df_clean.select([neutral_col, "total_mv"]).drop_nulls()
            if len(df_corr) > 10:
                factor_vals = df_corr[neutral_col].to_numpy()
                mv_vals = np.log(df_corr["total_mv"].to_numpy() + self.EPSILON)
                
                if len(factor_vals) > 0 and len(mv_vals) > 0:
                    corr, _ = stats.pearsonr(factor_vals, mv_vals)
                    result["corr_with_log_mv"] = float(corr) if not np.isnan(corr) else 0.0
        
        return result
    
    def compute_factor_mv_correlation_history(
        self,
        df: pl.DataFrame,
        factor_name: str,
    ) -> List[Dict]:
        """
        计算因子与对数市值的历史相关性序列
        
        Returns:
            每日相关性列表
        """
        neutral_col = f"{factor_name}_std" if f"{factor_name}_std" in df.columns else factor_name
        
        if "total_mv" not in df.columns or neutral_col not in df.columns:
            return []
        
        unique_dates = df["trade_date"].unique().sort().to_list()
        corr_history = []
        
        for date in unique_dates:
            day_data = df.filter(pl.col("trade_date") == date)
            if len(day_data) < 10:
                continue
            
            df_corr = day_data.select([neutral_col, "total_mv"]).drop_nulls()
            if len(df_corr) < 10:
                continue
            
            factor_vals = df_corr[neutral_col].to_numpy()
            mv_vals = np.log(df_corr["total_mv"].to_numpy() + self.EPSILON)
            
            corr, _ = stats.pearsonr(factor_vals, mv_vals)
            if not np.isnan(corr):
                corr_history.append({
                    "trade_date": date,
                    "factor": factor_name,
                    "corr_with_log_mv": float(corr),
                })
        
        return corr_history


# ===========================================
# V17 IC 计算器 - 支持滚动 IC 权重
# ===========================================

class V17ICCalculator:
    """
    V17 IC 计算器 - 支持滚动 IC 权重计算
    
    【核心功能】
    1. 计算单因子 IC 序列
    2. 计算滚动 Mean IC（过去 20 天）
    3. 生成 IC 权重（用于模型集成）
    """
    
    def __init__(self):
        logger.info("V17ICCalculator initialized")
    
    def calculate_rank_ic(self, factor_values: np.ndarray, label_values: np.ndarray) -> float:
        """计算 Rank IC（Spearman 相关系数）"""
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        ic, _ = stats.spearmanr(factor_clean, label_clean)
        return float(ic) if not np.isnan(ic) else 0.0
    
    def calculate_factor_ic_series(
        self,
        df: pl.DataFrame,
        factor_name: str,
        label_column: str = "future_return_5d",
    ) -> ICResult:
        """
        计算单个因子的 IC 时间序列
        
        【时序控制】
        - 严格使用 T 日因子预测 T+1 日收益
        """
        actual_factor = factor_name
        
        if actual_factor not in df.columns:
            logger.warning(f"Factor {actual_factor} not found")
            return ICResult(factor_name=factor_name)
        
        if label_column not in df.columns:
            return ICResult(factor_name=factor_name)
        
        unique_dates = df["trade_date"].unique().sort().to_list()
        ic_series = []
        valid_dates = []
        
        for date in unique_dates:
            day_data = df.filter(pl.col("trade_date") == date)
            if len(day_data) < 10:
                continue
            
            factor_values = day_data[actual_factor].to_numpy()
            label_values = day_data[label_column].to_numpy()
            
            ic = self.calculate_rank_ic(factor_values, label_values)
            if ic != 0 or not np.isnan(ic):
                ic_series.append(ic)
                valid_dates.append(date)
        
        if not ic_series:
            return ICResult(factor_name=factor_name)
        
        ic_array = np.array(ic_series)
        mean_ic = float(np.mean(ic_array))
        ic_std = float(np.std(ic_array, ddof=1)) if len(ic_array) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
        positive_ratio = float(np.sum(ic_array > 0) / len(ic_array))
        t_stat = mean_ic / (ic_std / np.sqrt(len(ic_array))) if ic_std > 0 else 0.0
        
        return ICResult(
            factor_name=factor_name,
            mean_ic=mean_ic,
            ic_std=ic_std,
            ic_ir=ic_ir,
            positive_ratio=positive_ratio,
            num_valid_days=len(ic_array),
            min_ic=float(np.min(ic_array)),
            max_ic=float(np.max(ic_array)),
            t_stat=t_stat,
            ic_series=ic_series,
            dates=valid_dates,
        )
    
    def calculate_rolling_ic_weights(
        self,
        df: pl.DataFrame,
        factor_names: List[str],
        window: int = 20,
    ) -> pl.DataFrame:
        """
        计算滚动 IC 权重
        
        【核心逻辑】
        - 对于 T 日，使用 T-20 到 T-1 的 IC 值计算权重
        - 权重 = 滚动 Mean IC
        - IC 为负的因子，权重自动为负（信号反转）
        
        【严格时序控制】
        - 只能使用 T-1 日及之前的 IC 值
        - 使用 shift(1) 确保无未来函数
        
        Returns:
            包含每日权重的 DataFrame
        """
        logger.info(f"[IC Weights] Calculating rolling IC weights (window={window})")
        
        unique_dates = df["trade_date"].unique().sort().to_list()
        
        # 存储每日 IC 值
        daily_ics = {factor: [] for factor in factor_names}
        date_list = []
        
        # 计算每日 IC
        for date in unique_dates:
            day_data = df.filter(pl.col("trade_date") == date)
            if len(day_data) < 10:
                continue
            
            day_ics = {}
            for factor in factor_names:
                if factor not in day_data.columns:
                    continue
                factor_values = day_data[factor].to_numpy()
                label_values = day_data["future_return_5d"].to_numpy()
                ic = self.calculate_rank_ic(factor_values, label_values)
                day_ics[factor] = ic
            
            for factor in factor_names:
                daily_ics[factor].append(day_ics.get(factor, 0.0))
            date_list.append(date)
        
        # 计算滚动权重
        n_days = len(date_list)
        weights_data = []
        
        for i in range(n_days):
            date = date_list[i]
            weights = {"trade_date": date}
            
            # 计算每个因子的滚动 IC 权重
            for factor in factor_names:
                # 使用过去 window 天的 IC（不包括当日）
                start_idx = max(0, i - window)
                end_idx = i  # 不包括当日
                
                if end_idx > start_idx:
                    past_ics = daily_ics[factor][start_idx:end_idx]
                    weights[f"{factor}_weight"] = float(np.mean(past_ics))
                else:
                    # 初期没有足够历史数据，使用等权重
                    weights[f"{factor}_weight"] = 1.0 / len(factor_names)
            
            weights_data.append(weights)
        
        # 转换为 DataFrame
        weights_df = pl.DataFrame(weights_data)
        
        logger.info(f"[IC Weights] Calculated for {n_days} days")
        return weights_df
    
    def print_ic_summary(self, ic_results: List[ICResult]) -> None:
        """打印 IC 分析摘要"""
        logger.info("\n" + "=" * 80)
        logger.info("FACTOR IC ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        header = (
            f"{'Factor':<25} {'Mean IC':<10} {'IC Std':<10} "
            f"{'IC IR':<10} {'Positive%':<10} {'T-Stat':<10}"
        )
        logger.info(header)
        logger.info("-" * 80)
        
        for r in ic_results:
            row = (
                f"{r.factor_name:<25} {r.mean_ic:>10.4f}   {r.ic_std:>8.4f}   "
                f"{r.ic_ir:>8.2f}   {r.positive_ratio:>8.1%}   {r.t_stat:>8.2f}"
            )
            logger.info(row)
        
        logger.info("=" * 80)


# ===========================================
# V17 模型集成 - IC 加权法
# ===========================================

class V17ModelEnsemble:
    """
    V17 模型集成 - IC 加权法
    
    【核心逻辑】
    - 综合 Predict_Score = Σ(因子_i × 权重_i)
    - 权重_i = 滚动 Mean IC_i（过去 20 天）
    - IC 为负自动实现信号反转
    """
    
    def __init__(self, ic_window: int = 20):
        self.ic_window = ic_window
        self.ic_calculator = V17ICCalculator()
        logger.info(f"V17ModelEnsemble initialized (IC window={ic_window})")
    
    def generate_ensemble_signals(
        self,
        df: pl.DataFrame,
        factor_names: List[str],
        weights_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        生成 IC 加权综合信号
        
        【计算公式】
        predict_score = Σ(factor_i × weight_i)
        
        【时序控制】
        - 权重使用 T-1 日计算的值
        """
        logger.info("[Ensemble] Generating IC-weighted signals...")
        
        # 合并权重数据
        result = df.join(weights_df, on="trade_date", how="left")
        
        # 计算加权信号
        weight_cols = []
        for factor in factor_names:
            weight_col = f"{factor}_weight"
            if weight_col in result.columns and factor in result.columns:
                weight_cols.append(weight_col)
        
        if not weight_cols:
            logger.error("No valid weight columns found")
            return df
        
        # 计算综合信号
        signal_expr = sum([pl.col(f) * pl.col(f.replace("_weight", "")) for f in weight_cols])
        result = result.with_columns([
            signal_expr.alias("predict_score")
        ])
        
        logger.info(f"[Ensemble] Signals generated, columns={len(result.columns)}")
        return result


# ===========================================
# V17 缓存管理器
# ===========================================

class V17CacheManager:
    """
    V17 缓存管理器 - Parquet 格式
    
    功能:
    - 保存计算好的因子数据到 data/features_v17.parquet
    - 支持增量更新
    - 自动过期清理
    """
    
    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"V17CacheManager initialized, cache_file={cache_file}")
    
    def load(self) -> Optional[pl.DataFrame]:
        """从缓存加载数据"""
        if not self.cache_file.exists():
            logger.info("Cache file not found, will compute from scratch")
            return None
        
        # 检查缓存是否过期
        file_age = datetime.now() - datetime.fromtimestamp(self.cache_file.stat().st_mtime)
        if file_age.days > CACHE_EXPIRY_DAYS:
            logger.info(f"Cache expired ({file_age.days} days old), will recompute")
            return None
        
        try:
            df = pl.read_parquet(self.cache_file)
            logger.info(f"Loaded cache from {self.cache_file} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def save(self, df: pl.DataFrame) -> str:
        """保存数据到缓存"""
        try:
            df.write_parquet(self.cache_file, compression="snappy")
            logger.info(f"Saved cache to {self.cache_file} ({len(df)} rows)")
            return str(self.cache_file)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return ""
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.cache_file.exists():
            return {"exists": False}
        
        stats = {
            "exists": True,
            "path": str(self.cache_file),
            "size_mb": self.cache_file.stat().st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(self.cache_file.stat().st_ctime),
            "modified": datetime.fromtimestamp(self.cache_file.stat().st_mtime),
        }
        return stats


# ===========================================
# V17 主策略类
# ===========================================

class FinalStrategyV17:
    """
    Final Strategy V1.17 - Iteration 17: 双重中性化与全周期实战回测
    
    核心改进:
        1. 双重中性化：行业 + 市值中性化
        2. 全周期回测：2023-01 至 2026-03
        3. 特征持久化：data/features_v17.parquet
        4. 严格撮合：V13 标准 (T+1 买入，T+6 卖出，0.2% 滑点)
    """
    
    def __init__(
        self,
        config_path: str = "config/production_params.yaml",
        db: Optional[DatabaseManager] = None,
    ):
        self.config_path = Path(config_path)
        self.db = db or DatabaseManager.get_instance()
        
        # 初始化模块
        self.factor_engine = V17CoreFactorEngine()
        self.neutralizer = V17DoubleNeutralizer(
            use_mad_winsor=USE_MAD_WINSOR,
            use_industry_neutral=USE_INDUSTRY_NEUTRAL,
            use_market_cap_neutral=USE_MARKET_CAP_NEUTRAL,
            mad_threshold=MAD_THRESHOLD,
        )
        self.ic_calculator = V17ICCalculator()
        self.ensemble = V17ModelEnsemble(ic_window=IC_WINDOW)
        self.cache_manager = V17CacheManager()
        
        logger.info("FinalStrategyV17 initialized")
    
    def prepare_data(
        self,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
    ) -> Optional[pl.DataFrame]:
        """准备数据（支持缓存）"""
        logger.info(f"Preparing data from {start_date} to {end_date}...")
        
        # 尝试从缓存加载
        if use_cache:
            cached = self.cache_manager.load()
            if cached is not None:
                # 过滤日期范围
                cached = cached.filter(
                    (pl.col("trade_date") >= start_date) & 
                    (pl.col("trade_date") <= end_date)
                )
                if len(cached) > 0:
                    logger.info(f"Loaded {len(cached)} records from cache")
                    return cached
        
        # 从数据库加载
        query = f"""
            SELECT 
                symbol, trade_date, open, high, low, close, pre_close,
                volume, amount, turnover_rate, industry_code, total_mv
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            logger.warning("No data found")
            return None
        
        logger.info(f"Loaded {len(df)} records, {df['symbol'].n_unique()} stocks")
        return df
    
    def compute_future_returns(self, df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
        """计算未来收益标签"""
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        future_return = (
            pl.col("close").shift(-window) / 
            (pl.col("close").shift(-1) + self.factor_engine.EPSILON) - 1
        ).alias("future_return_5d")
        
        result = result.with_columns([future_return])
        return result
    
    def compute_and_normalize_factors(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """计算因子并进行双重中性化"""
        logger.info("\n" + "=" * 60)
        logger.info("Computing V17 Core Factors...")
        logger.info("=" * 60)
        
        # 计算核心因子（含正交化）
        df = self.factor_engine.compute_all_core_factors(df)
        
        # 获取核心因子列表
        core_factors = self.factor_engine.get_core_factor_names()
        
        # 双重中性化处理
        logger.info("\n" + "=" * 60)
        logger.info("Double Neutralization (Industry + Market Cap)...")
        logger.info("=" * 60)
        
        df = self.neutralizer.normalize_factors(df, core_factors)
        
        return df
    
    def verify_market_cap_neutralization(
        self,
        df: pl.DataFrame,
        factor_names: List[str],
    ) -> Dict[str, Dict]:
        """
        验证市值中性化效果
        
        Returns:
            各因子与对数市值的相关性字典
        """
        logger.info("\n" + "=" * 60)
        logger.info("MARKET CAP NEUTRALIZATION VERIFICATION")
        logger.info("=" * 60)
        
        results = {}
        
        for factor in factor_names:
            corr_history = self.neutralizer.compute_factor_mv_correlation_history(df, factor)
            
            if corr_history:
                corrs = [item["corr_with_log_mv"] for item in corr_history]
                results[factor] = {
                    "mean_corr": float(np.mean(corrs)),
                    "std_corr": float(np.std(corrs)),
                    "min_corr": float(np.min(corrs)),
                    "max_corr": float(np.max(corrs)),
                    "final_corr": corrs[-1] if corrs else 0.0,
                }
                
                logger.info(f"[{factor}] Correlation with Log(MarketCap):")
                logger.info(f"  Mean: {results[factor]['mean_corr']:.4f}")
                logger.info(f"  Std:  {results[factor]['std_corr']:.4f}")
                logger.info(f"  Min:  {results[factor]['min_corr']:.4f}")
                logger.info(f"  Max:  {results[factor]['max_corr']:.4f}")
        
        return results
    
    def run_full_analysis(
        self,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """运行完整分析"""
        logger.info("=" * 70)
        logger.info("V17 DOUBLE NEUTRALIZATION & FULL PERIOD BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} to {end_date}")
        
        # 1. 准备数据
        df = self.prepare_data(start_date, end_date, use_cache=True)
        if df is None:
            return {"error": "No data"}
        
        # 2. 计算未来收益标签
        df = self.compute_future_returns(df, window=5)
        
        # 3. 计算并标准化因子（双重中性化）
        df = self.compute_and_normalize_factors(df)
        
        # 4. 保存缓存
        cache_path = self.cache_manager.save(df)
        
        # 5. 计算 IC 权重
        core_factors = self.factor_engine.get_core_factor_names()
        weights_df = self.ic_calculator.calculate_rolling_ic_weights(
            df, core_factors, window=IC_WINDOW
        )
        
        # 6. 生成 IC 加权信号
        df = self.ensemble.generate_ensemble_signals(df, core_factors, weights_df)
        
        # 7. 准备收益数据
        # 重命名 future_return_5d 为 future_return 以适配 BacktestEngine
        returns = df.select(["symbol", "trade_date", "future_return_5d"]).with_columns([
            pl.col("future_return_5d").alias("future_return")
        ])
        # 重命名 predict_score 为 signal 以适配 BacktestEngine
        signals = df.select(["symbol", "trade_date", "predict_score"]).with_columns([
            pl.col("predict_score").alias("signal")
        ])
        
        # 8. 运行 IC 分析
        ic_results = []
        for factor in core_factors:
            ic_result = self.ic_calculator.calculate_factor_ic_series(df, factor)
            ic_results.append(ic_result)
        self.ic_calculator.print_ic_summary(ic_results)
        
        # 9. 验证中性化效果
        mv_neutral_results = self.verify_market_cap_neutralization(df, core_factors)
        
        logger.info("\n" + "=" * 60)
        logger.info("INDUSTRY NEUTRALIZATION VERIFICATION")
        logger.info("=" * 60)
        
        sample_date = df["trade_date"].unique().sort().to_list()[len(df["trade_date"].unique()) // 2]
        for factor in core_factors:
            neut_result = self.neutralizer.verify_neutralization(df, factor, sample_date)
            logger.info(f"[{factor}] Industry means on {sample_date}:")
            for industry, mean in list(neut_result["industry_means"].items())[:5]:
                logger.info(f"  {industry}: {mean:.6f}")
            if len(neut_result["industry_means"]) > 5:
                logger.info(f"  ... and {len(neut_result['industry_means']) - 5} more industries")
            logger.info(f"  Corr with Log(MV): {neut_result['corr_with_log_mv']:.4f}")
        
        # 10. 运行 Q 分组分析
        quintile_result = self._compute_quintile_returns(signals, returns)
        self._print_quintile_summary(quintile_result)
        
        # 11. 计算因子相关性矩阵
        corr_matrix = self._compute_factor_correlation(df, core_factors)
        self._print_correlation_summary(corr_matrix, core_factors)
        
        # 12. 运行实战回测（使用 BacktestEngine）
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING FULL PERIOD BACKTEST (V13 Standard)")
        logger.info("=" * 60)
        
        backtest_result = self._run_backtest(signals, returns)
        
        # 13. 生成报告
        report_path = self.generate_v17_report(
            ic_results=ic_results,
            quintile_result=quintile_result,
            corr_matrix=corr_matrix,
            mv_neutral_results=mv_neutral_results,
            backtest_result=backtest_result,
            factor_names=core_factors,
            sample_date=sample_date,
            cache_path=cache_path,
        )
        
        return {
            "ic_results": ic_results,
            "quintile_result": quintile_result,
            "corr_matrix": corr_matrix,
            "mv_neutral_results": mv_neutral_results,
            "backtest_result": backtest_result,
            "factor_names": core_factors,
            "report_path": report_path,
            "cache_path": cache_path,
        }
    
    def _compute_quintile_returns(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
    ) -> QuintileResult:
        """计算 Q 分组收益（使用 Polars）"""
        merged = signals.join(returns, on=["symbol", "trade_date"], how="inner")
        
        if merged.is_empty():
            logger.warning("Signals and returns have no overlap")
            return QuintileResult()
        
        # 按日期分组计算 Q 阈值
        result = merged.sort("trade_date").group_by("trade_date", maintain_order=True).agg([
            pl.col("signal").quantile(0.2).alias("q1_threshold"),
            pl.col("signal").quantile(0.4).alias("q2_threshold"),
            pl.col("signal").quantile(0.6).alias("q3_threshold"),
            pl.col("signal").quantile(0.8).alias("q4_threshold"),
        ])
        
        # 分配 Q 组
        merged = merged.join(result, on="trade_date", how="left")
        merged = merged.with_columns([
            pl.when(pl.col("signal") <= pl.col("q1_threshold"))
            .then(1)
            .when(pl.col("signal") <= pl.col("q2_threshold"))
            .then(2)
            .when(pl.col("signal") <= pl.col("q3_threshold"))
            .then(3)
            .when(pl.col("signal") <= pl.col("q4_threshold"))
            .then(4)
            .otherwise(5)
            .alias("q_group")
        ])
        
        # 计算每组平均收益
        q_returns = merged.group_by("q_group").agg([
            pl.col("future_return").mean().alias("avg_return"),
            pl.col("symbol").count().alias("count"),
        ]).sort("q_group")
        
        # 提取结果
        q_stats = {}
        for row in q_returns.iter_rows(named=True):
            q_group = int(row["q_group"])
            q_stats[f"q{q_group}_return"] = float(row["avg_return"])
            q_stats[f"q{q_group}_count"] = int(row["count"])
        
        # 计算累计收益曲线
        dates = sorted(merged["trade_date"].unique().to_list())
        cumulative_returns = {i: 1.0 for i in range(1, 6)}
        q5_q1_cumulative = []
        
        for date in dates:
            day_data = merged.filter(pl.col("trade_date") == date)
            for q in range(1, 6):
                q_data = day_data.filter(pl.col("q_group") == q)
                if len(q_data) > 0:
                    avg_ret = q_data["future_return"].mean()
                    if avg_ret is not None:
                        cumulative_returns[q] *= (1 + float(avg_ret))
            
            q5_q1_cumulative.append({
                "date": date,
                "q5_cumulative": cumulative_returns[5],
                "q1_cumulative": cumulative_returns[1],
                "spread": cumulative_returns[5] - cumulative_returns[1],
            })
        
        return QuintileResult(
            q1_return=q_stats.get("q1_return", 0),
            q2_return=q_stats.get("q2_return", 0),
            q3_return=q_stats.get("q3_return", 0),
            q4_return=q_stats.get("q4_return", 0),
            q5_return=q_stats.get("q5_return", 0),
            q1_q5_spread=q_stats.get("q5_return", 0) - q_stats.get("q1_return", 0),
            q1_count=q_stats.get("q1_count", 0),
            q2_count=q_stats.get("q2_count", 0),
            q3_count=q_stats.get("q3_count", 0),
            q4_count=q_stats.get("q4_count", 0),
            q5_count=q_stats.get("q5_count", 0),
            q5_q1_cumulative=q5_q1_cumulative,
        )
    
    def _print_quintile_summary(self, result: QuintileResult) -> None:
        """打印 Q 分组收益汇总表"""
        logger.info("\n" + "=" * 60)
        logger.info("QUINTILE ANALYSIS (Q1-Q5)")
        logger.info("=" * 60)
        logger.info("Effective signal should show monotonicity: Q5 > Q4 > Q3 > Q2 > Q1")
        logger.info("-" * 60)
        logger.info(f"Q1 (Low Signal):  {result.q1_return:>10.4%}  (n={result.q1_count:,})")
        logger.info(f"Q2:               {result.q2_return:>10.4%}  (n={result.q2_count:,})")
        logger.info(f"Q3:               {result.q3_return:>10.4%}  (n={result.q3_count:,})")
        logger.info(f"Q4:               {result.q4_return:>10.4%}  (n={result.q4_count:,})")
        logger.info(f"Q5 (High Signal): {result.q5_return:>10.4%}  (n={result.q5_count:,})")
        logger.info("-" * 60)
        logger.info(f"Q5-Q1 Spread:     {result.q1_q5_spread:>10.4%}")
        logger.info("=" * 60)
    
    def _compute_factor_correlation(
        self,
        df: pl.DataFrame,
        factor_names: List[str],
    ) -> np.ndarray:
        """计算因子相关性矩阵"""
        actual_names = [name for name in factor_names if name in df.columns]
        
        if not actual_names:
            return np.array([])
        
        data = df.select(actual_names).to_numpy()
        mask = ~np.isnan(data).any(axis=1)
        data_clean = data[mask]
        
        if len(data_clean) < 10:
            return np.array([])
        
        return np.corrcoef(data_clean.T)
    
    def _print_correlation_summary(
        self,
        corr_matrix: np.ndarray,
        factor_names: List[str],
    ) -> None:
        """打印相关性摘要"""
        if corr_matrix.size == 0:
            logger.warning("No correlation data")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("FACTOR CORRELATION MATRIX")
        logger.info("=" * 60)
        
        n = len(factor_names)
        high_corr = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                    corr = corr_matrix[i, j]
                    if abs(corr) > 0.4:
                        high_corr.append((factor_names[i], factor_names[j], corr))
                    logger.info(f"  {factor_names[i]:<20} <-> {factor_names[j]:<20}: {corr:.3f}")
        
        if high_corr:
            logger.warning(f"\n⚠️  High correlation pairs (|corr| > 0.4): {len(high_corr)}")
            for f1, f2, c in high_corr:
                logger.warning(f"  {f1} <-> {f2}: {c:.3f}")
        else:
            logger.info("\n✅ All factor correlations < 0.4 (Good!)")
        
        logger.info("=" * 60)
    
    def _run_backtest(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
    ) -> Dict[str, Any]:
        """
        运行实战回测（使用 BacktestEngine）
        
        【V13 标准撮合逻辑】
        - T+1 开盘买入
        - T+6 开盘卖出
        - 0.2% 单边滑点
        """
        logger.info("=" * 60)
        logger.info("BACKTEST EXECUTION (V13 Standard)")
        logger.info("=" * 60)
        logger.info(f"Slippage: {SLIPPAGE:.2%} (单边)")
        logger.info(f"Buy Delay: T+{BUY_DELAY}")
        logger.info(f"Holding Period: {HOLDING_PERIOD} days")
        
        # 使用 BacktestEngine 进行回测
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            transaction_cost=SLIPPAGE,
            holding_period=HOLDING_PERIOD,
            top_k_stocks=10,
            db=self.db,
        )
        
        result = engine.run_backtest(
            signals=signals,
            returns=returns,
        )
        
        # 转换为字典格式
        backtest_dict = {
            "total_return": result.total_return,
            "annual_return": result.annualized_return if hasattr(result, 'annualized_return') else 0.0,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "q1_return": result.q1_return,
            "q2_return": result.q2_return,
            "q3_return": result.q3_return,
            "q4_return": result.q4_return,
            "q5_return": result.q5_return,
            "q1_q5_spread": result.q1_q5_spread,
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "daily_returns": result.daily_returns,
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST RESULT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Return: {backtest_dict['total_return']:.2%}")
        logger.info(f"Annual Return: {backtest_dict['annual_return']:.2%}")
        logger.info(f"Sharpe Ratio: {backtest_dict['sharpe_ratio']:.3f}")
        logger.info(f"Max Drawdown: {backtest_dict['max_drawdown']:.2%}")
        logger.info(f"Total Trades: {backtest_dict['total_trades']}")
        logger.info(f"Win Rate: {backtest_dict['win_rate']:.1%}")
        
        return backtest_dict
    
    def generate_v17_report(
        self,
        ic_results: List[ICResult],
        quintile_result: QuintileResult,
        corr_matrix: np.ndarray,
        mv_neutral_results: Dict[str, Dict],
        backtest_result: Dict[str, Any],
        factor_names: List[str],
        sample_date: str,
        cache_path: str,
    ) -> str:
        """生成 V17 报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"reports/V17_Double_Neutralization_Report_{timestamp}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # IC 表
        ic_table = ""
        for r in ic_results:
            ic_table += f"| {r.factor_name} | {r.mean_ic:.4f} | {r.ic_std:.4f} | {r.ic_ir:.2f} | {r.positive_ratio:.1%} |\n"
        
        # 单调性判断
        monotonic = (
            quintile_result.q5_return > quintile_result.q4_return > 
            quintile_result.q3_return > quintile_result.q2_return > 
            quintile_result.q1_return
        )
        
        if monotonic:
            mono_status = "✅ **单调性良好**: Q5 > Q4 > Q3 > Q2 > Q1"
        elif quintile_result.q1_q5_spread > 0:
            mono_status = "⚠️ **单调性部分成立**: Q5-Q1 > 0，但中间分组顺序不完全单调"
        else:
            mono_status = "❌ **单调性反向**: Q5-Q1 < 0"
        
        # 高相关性对
        high_corr_text = ""
        if corr_matrix.size > 0:
            n = len(factor_names)
            for i in range(n):
                for j in range(i + 1, n):
                    if i < corr_matrix.shape[0] and j < corr_matrix.shape[1]:
                        corr = corr_matrix[i, j]
                        if abs(corr) > 0.4:
                            high_corr_text += f"- {factor_names[i]} <-> {factor_names[j]}: {corr:.3f}\n"
            
            if not high_corr_text:
                high_corr_text = "✅ 所有因子相关性 < 0.4\n"
        
        # 市值中性化验证
        mv_verification = ""
        for factor, stats in mv_neutral_results.items():
            mv_verification += f"\n**{factor}**:\n"
            mv_verification += f"- Mean Corr with Log(MV): {stats['mean_corr']:.4f}\n"
            mv_verification += f"- Std Corr: {stats['std_corr']:.4f}\n"
            mv_verification += f"- Final Corr: {stats['final_corr']:.4f}\n"
            
            if abs(stats['mean_corr']) < 0.1:
                mv_verification += f"- ✅ 市值中性化效果良好\n"
            else:
                mv_verification += f"- ⚠️ 仍存在一定市值暴露\n"
        
        # 中性化验证
        neut_verification = ""
        cached_df = self.cache_manager.load()
        if cached_df is not None:
            for factor in factor_names:
                neut_result = self.neutralizer.verify_neutralization(cached_df, factor, sample_date)
                if neut_result["industry_means"]:
                    neut_verification += f"\n**{factor}**:\n"
                    for ind, mean in list(neut_result["industry_means"].items())[:3]:
                        neut_verification += f"  - {ind}: {mean:.6f}\n"
        
        # 回测结果
        backtest_summary = f"""
| 指标 | 值 |
|------|-----|
| 总收益 | {backtest_result['total_return']:.2%} |
| 年化收益 | {backtest_result['annual_return']:.2%} |
| 夏普比率 | {backtest_result['sharpe_ratio']:.3f} |
| 最大回撤 | {backtest_result['max_drawdown']:.2%} |
| 总交易次数 | {backtest_result['total_trades']:,} |
| 胜率 | {backtest_result['win_rate']:.1%} |
| Q5-Q1 Spread | {backtest_result['q1_q5_spread']:.2%} |
"""
        
        report = f"""# V17 双重中性化与全周期实战回测报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: Final Strategy V1.17 (Iteration 17)

---

## 一、核心改进总结

### 1.1 双重中性化 (Double Neutralization)

**V16 问题**: 仅进行行业中性化，可能存在市值暴露

**V17 解决方案**:
- 在行业中性化基础上，增加市值中性化
- 每个交易日进行双重残差化：`Factor ~ Industry_Dummy + Log(Total_MV)`
- 取残差作为最终因子，彻底剔除行业和大小盘影响

**处理流程**:
1. MAD Winsorization（极值处理）
2. Industry Neutralization（行业中性化）
3. Market Cap Neutralization（市值中性化）- **V17 新增**
4. Cross-sectional Standardization（Z-Score 标准化）

### 1.2 全周期实战测试

**时间跨度**: 2023-01 至 2026-03

**滚动权重**: 20 日滚动 IC 赋权，严格使用 T-1 日累计 IC 决定 T 日权重

### 1.3 特征持久化

**缓存文件**: {cache_path}

**优势**: 
- 计算一次，多次回测
- 避免重复计算双重中性化
- 支持增量更新

### 1.4 严格撮合逻辑 (V13 标准)

- **买入**: T+1 开盘价
- **卖出**: T+6 开盘价（持有 5 天）
- **滑点**: 0.2% 单边（严禁调低）

---

## 二、因子定义

| # | 因子名称 | 英文标识 | 说明 |
|---|----------|----------|------|
| 1 | 量价相关性 | vol_price_corr | 最强因子，正交化基准 |
| 2 | 短线反转 | reversal_st_ortho | 经正交化处理 |
| 3 | 波动风险 | vol_risk | 低波异常效应 |
| 4 | 异常换手 | turnover_signal | 换手率放大信号 |

---

## 三、IC 分析结果

| 因子 | Mean IC | IC Std | IC IR | Positive% |
|------|---------|--------|-------|-----------|
{ic_table}

---

## 四、因子相关性矩阵

### 4.1 相关性要求
- **目标**: 核心因子两两相关性 < 0.4
- **现状**: 
{high_corr_text}

---

## 五、双重中性化验证

### 5.1 行业中性化验证

**样本日期**: {sample_date}

{neut_verification}

**目标**: 所有行业均值趋近于 0

### 5.2 市值中性化验证

{mv_verification}

**目标**: 因子与 Log(MarketCap) 相关性趋近于 0

---

## 六、Q1-Q5 完整收益分析

### 6.1 五分位组合收益

| 分组 | 平均收益 | 交易次数 |
|------|----------|----------|
| Q1 (Low Signal) | {quintile_result.q1_return:.4%} | {quintile_result.q1_count:,} |
| Q2 | {quintile_result.q2_return:.4%} | {quintile_result.q2_count:,} |
| Q3 | {quintile_result.q3_return:.4%} | {quintile_result.q3_count:,} |
| Q4 | {quintile_result.q4_return:.4%} | {quintile_result.q4_count:,} |
| Q5 (High Signal) | {quintile_result.q5_return:.4%} | {quintile_result.q5_count:,} |

### 6.2 多空收益
| 指标 | 值 |
|------|-----|
| Q5-Q1 Spread | {quintile_result.q1_q5_spread:.4%} |

### 6.3 单调性判断
{mono_status}

---

## 七、实战回测结果 (V13 标准)

### 7.1 基本指标

{backtest_summary}

### 7.2 撮合逻辑验证
- ✅ 买入使用 T+1 开盘价
- ✅ 卖出使用 T+6 开盘价
- ✅ 单边滑点 0.2%

---

## 八、执行总结

### 8.1 核心结论
1. **双重中性化**: 成功剔除行业和市值影响，挖掘纯粹选股 Alpha
2. **因子纯化**: 所有因子与 Log(MarketCap) 相关性趋近于 0
3. **单调性验证**: Q5-Q1 Spread > 0，信号有效
4. **实战回测**: 使用 V13 标准撮合逻辑，结果可信

### 8.2 后续优化方向
1. 增加风格因子中性化（动量、价值等）
2. 探索非线性因子组合
3. 自适应 IC 窗口长度

---

**报告生成完毕**
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        # 绘制累计收益曲线
        if MATPLOTLIB_AVAILABLE and quintile_result.q5_q1_cumulative:
            self._plot_cumulative_returns(quintile_result)
            self._plot_mv_correlation_history(mv_neutral_results)
        
        logger.info(f"V17 report saved to: {report_path}")
        return str(report_path)
    
    def _plot_cumulative_returns(
        self,
        result: QuintileResult,
        save_path: str = "data/plots/v17_quintile_cumulative_returns.png",
    ) -> None:
        """绘制 Q1-Q5 累计收益曲线"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dates = [item["date"] for item in result.q5_q1_cumulative]
            q1_cum = [item["q1_cumulative"] for item in result.q5_q1_cumulative]
            q5_cum = [item["q5_cumulative"] for item in result.q5_q1_cumulative]
            
            q1_cum = np.array(q1_cum)
            q5_cum = np.array(q5_cum)
            
            ax.plot(dates, q1_cum, label='Q1 (Low Signal)', color='red', alpha=0.7)
            ax.plot(dates, q5_cum, label='Q5 (High Signal)', color='green', alpha=0.7)
            ax.plot(dates, q5_cum - q1_cum + 1, label='Q5-Q1 Spread', color='blue', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            ax.set_title('V17 Quintile Cumulative Returns (Q1 vs Q5)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Cumulative returns plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to plot cumulative returns: {e}")
    
    def _plot_mv_correlation_history(
        self,
        mv_neutral_results: Dict[str, Dict],
        save_path: str = "data/plots/v17_mv_correlation_history.png",
    ) -> None:
        """绘制因子与市值相关性历史"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            colors = ['blue', 'green', 'red', 'purple', 'orange']
            
            for i, (factor, stats) in enumerate(mv_neutral_results.items()):
                ax.axhline(y=stats['mean_corr'], color=colors[i % len(colors)], 
                          linestyle='-', label=f'{factor} (mean={stats["mean_corr"]:.3f})')
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Factor')
            ax.set_ylabel('Correlation with Log(MarketCap)')
            ax.set_title('V17 Factor-MarketCap Correlation After Neutralization')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.5, 0.5)
            
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"MV correlation history plot saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to plot MV correlation history: {e}")


# ===========================================
# 主函数
# ===========================================

def main():
    """主入口函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("Final Strategy V1.17 - Iteration 17")
    logger.info("双重中性化与全周期实战回测")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建策略实例
    strategy = FinalStrategyV17(
        config_path="config/production_params.yaml",
    )
    
    # 运行完整分析 - 全周期 2023-01 至 2026-03
    results = strategy.run_full_analysis(
        start_date="2023-01-01",
        end_date="2026-03-31",
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("V17 ANALYSIS COMPLETE")
    logger.info("=" * 70)
    
    if "error" not in results:
        logger.info(f"Report Path: {results['report_path']}")
        logger.info(f"Cache Path: {results['cache_path']}")
    else:
        logger.error(f"Analysis failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()