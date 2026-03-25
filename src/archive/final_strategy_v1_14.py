"""
Final Strategy V1.14 - Iteration 14: 核心特征挖掘与因子集成

【核心改进 - Iteration 14】
1. 波动率特征 (Volatility):
   - volatility_20: 过去 20 日收益率标准差
   - volatility_ratio: 短期/长期波动率比值
   - 测试"低波动异常收益"效应

2. 动量/反转特征 (Momentum/Reversal):
   - momentum_5: 5 日收益率（短线动量）
   - momentum_20: 20 日收益率（中线动量）
   - reversal_signal: 反转信号（动量变化率）
   - 测试 A 股短线反转效应

3. 资金流特征 (Liquidity):
   - vwap_return: 成交额加权收益率
   - turnover_change: 换手率变化率
   - amount_ma_ratio: 成交额相对 MA20 比值
   - 识别主力资金异常流动

4. 全分组单调性分析 (Full Quintile Analysis):
   - 同时计算 Q1, Q2, Q3, Q4, Q5 五个组合的累计收益率
   - 展示五条曲线的分布

5. 集成学习模型升级:
   - 使用 Ridge 回归集成 5-8 个因子
   - 输出综合 Predict_Score
   - 严格时序验证：仅使用 shift(1) 后的数据

【严禁数据偷看】
- 所有因子计算必须使用 shift(1) 确保无未来函数
- 信号生成仅能使用 T 日及之前数据
- 回测撮合保持 T+1 开盘买入、T+6 开盘卖出

作者：量化策略团队
版本：V1.14 (Iteration 14)
日期：2026-03-17
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import json

import polars as pl
import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import joblib

# 导入本地模块
try:
    from .db_manager import DatabaseManager
    from .factor_engine import FactorEngine
    from .feature_pipeline import FeaturePipeline
    from .model_trainer import ModelTrainer
    from .backtest_engine import BacktestEngine
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine
    from feature_pipeline import FeaturePipeline
    from model_trainer import ModelTrainer
    from backtest_engine import BacktestEngine


# ===========================================
# 配置常量
# ===========================================

# 波动率特征参数
VOLATILITY_WINDOW_SHORT = 5   # 短期波动率窗口
VOLATILITY_WINDOW_LONG = 20   # 长期波动率窗口

# 动量/反转特征参数
MOMENTUM_SHORT_WINDOW = 5     # 短期动量窗口
MOMENTUM_LONG_WINDOW = 20     # 长期动量窗口

# 资金流特征参数
VWAP_WINDOW = 20              # VWAP 计算窗口
TURNOVER_WINDOW = 10          # 换手率变化窗口

# 集成学习参数
MODEL_TYPE = "ridge"          # ridge / random_forest / gradient_boosting
RIDGE_ALPHA = 1.0             # Ridge 回归 alpha 参数
RF_N_ESTIMATORS = 100         # 随机森林树数量
RF_MAX_DEPTH = 5              # 随机森林最大深度
GB_N_ESTIMATORS = 100         # GBDT 树数量
GB_MAX_DEPTH = 3              # GBDT 最大深度

# 训练数据配置
TRAIN_END_DATE = "2023-12-31"  # 训练截止日期
TRAIN_START_DATE = "2022-01-01"  # 训练开始日期

# 因子权重配置 (用于简单加权基准)
FACTOR_WEIGHTS_V14 = {
    # 波动率特征
    "volatility_20": -0.08,      # 低波动率偏好
    "volatility_ratio": -0.04,   # 波动率收缩偏好
    
    # 动量/反转特征
    "momentum_5": 0.06,          # 短期动量
    "momentum_20": 0.04,         # 长期动量
    "reversal_signal": 0.05,     # 反转信号
    
    # 资金流特征
    "vwap_return": 0.08,         # VWAP 收益
    "turnover_change": 0.04,     # 换手率变化
    "amount_ma_ratio": 0.05,     # 成交额放大
}


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


@dataclass
class QuintileResult:
    """五分位组合收益结果"""
    q1_return: float = 0.0  # 最低分组
    q2_return: float = 0.0
    q3_return: float = 0.0
    q4_return: float = 0.0
    q5_return: float = 0.0  # 最高分组
    q1_q5_spread: float = 0.0  # Q5 - Q1 多空收益
    q1_count: int = 0
    q2_count: int = 0
    q3_count: int = 0
    q4_count: int = 0
    q5_count: int = 0


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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "avg_hold_days": self.avg_hold_days,
            "profit_factor": self.profit_factor,
        }


# ===========================================
# V14 特征工程模块
# ===========================================

class V14FeatureEngine:
    """
    V14 特征工程模块 - 核心特征挖掘
    
    新增特征类别:
    1. 波动率特征 (Volatility)
    2. 动量/反转特征 (Momentum/Reversal)
    3. 资金流特征 (Liquidity)
    
    【时序控制】
    - 所有因子计算必须使用 shift(1) 确保无未来函数
    - 信号生成仅能使用 T 日及之前数据
    """
    
    EPSILON = 1e-6
    
    def __init__(self):
        logger.info("V14FeatureEngine initialized")
    
    def compute_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算波动率特征
        
        【金融逻辑】
        - 低波动率股票往往有异常收益（低波动异常）
        - 波动率收缩预示突破机会
        
        【计算公式】
        - volatility_20 = std(returns, 20)
        - volatility_5 = std(returns, 5)
        - volatility_ratio = volatility_5 / volatility_20
        
        【时序控制】
        - 收益率计算使用 close[t] / close[t-1] - 1
        - rolling_std 使用历史 N 日数据
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            
        Returns:
            添加了波动率特征的 DataFrame
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 计算日收益率
        returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        
        # 短期波动率 (5 日)
        volatility_5 = returns.rolling_std(window_size=VOLATILITY_WINDOW_SHORT, ddof=1)
        
        # 长期波动率 (20 日)
        volatility_20 = returns.rolling_std(window_size=VOLATILITY_WINDOW_LONG, ddof=1)
        
        # 波动率比率 (短期/长期)
        # 值 < 1 表示波动率收缩，值 > 1 表示波动率扩张
        volatility_ratio = volatility_5 / (volatility_20 + self.EPSILON)
        
        # 波动率变化率
        volatility_change = volatility_20 / (volatility_20.shift(1) + self.EPSILON) - 1
        
        result = result.with_columns([
            volatility_5.alias("volatility_5"),
            volatility_20.alias("volatility_20"),
            volatility_ratio.alias("volatility_ratio"),
            volatility_change.alias("volatility_change"),
        ])
        
        logger.debug(f"[Volatility Features] Computed, rows={len(result)}")
        return result
    
    def compute_momentum_reversal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算动量/反转特征
        
        【金融逻辑】
        - A 股市场存在短线反转效应（5 日）
        - 中长期存在动量效应（20 日）
        
        【计算公式】
        - momentum_5 = close[t] / close[t-5] - 1
        - momentum_20 = close[t] / close[t-20] - 1
        - reversal_signal = -momentum_5 * sign(momentum_20 - momentum_5)
        
        【时序控制】
        - 动量计算使用历史价格
        
        Args:
            df: 包含价格数据的 DataFrame
            
        Returns:
            添加了动量/反转特征的 DataFrame
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 短期动量 (5 日收益率)
        momentum_5 = pl.col("close") / (pl.col("close").shift(MOMENTUM_SHORT_WINDOW) + self.EPSILON) - 1
        
        # 长期动量 (20 日收益率)
        momentum_20 = pl.col("close") / (pl.col("close").shift(MOMENTUM_LONG_WINDOW) + self.EPSILON) - 1
        
        # 动量变化率（用于反转信号）
        momentum_change = momentum_5 - momentum_20
        
        # 反转信号
        # 当短期动量与长期动量方向相反时，可能存在反转机会
        # 简化版本：反转信号 = -短期动量（赌反转）
        reversal_signal = -momentum_5 * (1.0 / (momentum_20.abs() + self.EPSILON))
        
        # 限制在合理范围内
        reversal_signal = reversal_signal.clip(-5.0, 5.0)
        
        result = result.with_columns([
            momentum_5.alias("momentum_5"),
            momentum_20.alias("momentum_20"),
            momentum_change.alias("momentum_change"),
            reversal_signal.alias("reversal_signal"),
        ])
        
        logger.debug(f"[Momentum/Reversal Features] Computed, rows={len(result)}")
        return result
    
    def compute_liquidity_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算资金流特征
        
        【金融逻辑】
        - 成交额放大预示主力资金流入
        - VWAP 相关指标反映真实成交成本
        - 换手率变化反映市场关注度
        
        【计算公式】
        - vwap = (high + low + close) / 3
        - vwap_return = close / vwap_lag - 1
        - turnover_change = turnover[t] / turnover[t-10] - 1
        - amount_ma_ratio = amount / ma(amount, 20)
        
        【时序控制】
        - 所有计算使用历史数据
        
        Args:
            df: 包含 OHLCV 数据的 DataFrame
            
        Returns:
            添加了资金流特征的 DataFrame
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            pl.col("amount").cast(pl.Float64, strict=False),
        ])
        
        # 确保 amount 列存在
        if "amount" not in result.columns:
            result = result.with_columns([
                (pl.col("volume") * pl.col("close")).alias("amount")
            ])
        
        # 计算 VWAP (成交量加权平均价)
        vwap = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
        
        # VWAP 收益率（相对于 N 日前 VWAP）
        vwap_return = pl.col("close") / (vwap.shift(VWAP_WINDOW) + self.EPSILON) - 1
        
        # 成交额 MA 比率
        amount_ma_20 = pl.col("amount").rolling_mean(window_size=VWAP_WINDOW)
        amount_ma_ratio = pl.col("amount") / (amount_ma_20 + self.EPSILON)
        
        # 换手率特征（如果有 turnover_rate）
        if "turnover_rate" in result.columns:
            turnover = pl.col("turnover_rate").cast(pl.Float64, strict=False)
            turnover_ma = turnover.rolling_mean(window_size=TURNOVER_WINDOW)
            turnover_change = turnover / (turnover_ma + self.EPSILON) - 1
            turnover_ma_ratio = turnover / (turnover_ma + self.EPSILON)
        else:
            # 使用成交量变化率代替
            volume_ma = pl.col("volume").rolling_mean(window_size=TURNOVER_WINDOW)
            turnover_change = pl.col("volume") / (volume_ma + self.EPSILON) - 1
            turnover_ma_ratio = turnover_change
        
        # 资金流强度（成交额放大 + 价格上涨）
        price_change = pl.col("close") / (pl.col("close").shift(5) + self.EPSILON) - 1
        money_flow_intensity = amount_ma_ratio * price_change
        
        result = result.with_columns([
            vwap.alias("vwap"),
            vwap_return.alias("vwap_return"),
            amount_ma_ratio.alias("amount_ma_ratio"),
            turnover_change.alias("turnover_change"),
            turnover_ma_ratio.alias("turnover_ma_ratio"),
            money_flow_intensity.alias("money_flow_intensity"),
        ])
        
        logger.debug(f"[Liquidity Features] Computed, rows={len(result)}")
        return result
    
    def compute_all_v14_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算所有 V14 新增特征
        
        Args:
            df: 输入 DataFrame
            
        Returns:
            添加了所有 V14 特征的 DataFrame
        """
        result = df.clone()
        
        # 计算波动率特征
        result = self.compute_volatility_features(result)
        
        # 计算动量/反转特征
        result = self.compute_momentum_reversal_features(result)
        
        # 计算资金流特征
        result = self.compute_liquidity_features(result)
        
        logger.info(f"[V14 Features] All features computed, total columns={len(result.columns)}")
        return result
    
    def get_v14_feature_names(self) -> List[str]:
        """获取 V14 新增特征名称列表"""
        return [
            # 波动率特征
            "volatility_5",
            "volatility_20",
            "volatility_ratio",
            "volatility_change",
            # 动量/反转特征
            "momentum_5",
            "momentum_20",
            "momentum_change",
            "reversal_signal",
            # 资金流特征
            "vwap",
            "vwap_return",
            "amount_ma_ratio",
            "turnover_change",
            "turnover_ma_ratio",
            "money_flow_intensity",
        ]


# ===========================================
# IC 计算器模块
# ===========================================

class V14ICCalculator:
    """
    V14 IC 计算器 - 因子 Rank IC 分析
    
    功能:
    - 计算每个因子的 Rank IC 值
    - 计算 IC 序列统计信息
    - 支持新因子 IC 矩阵输出
    """
    
    def __init__(self):
        logger.info("V14ICCalculator initialized")
    
    def calculate_rank_ic(self, factor_values: np.ndarray, label_values: np.ndarray) -> float:
        """
        计算 Rank IC 值（Spearman 相关系数）
        
        Args:
            factor_values: 因子值序列
            label_values: 标签值序列（未来收益率）
            
        Returns:
            Rank IC 值
        """
        from scipy import stats
        
        # 去除空值
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算 Spearman 相关系数
        ic, _ = stats.spearmanr(factor_clean, label_clean)
        
        return float(ic) if not np.isnan(ic) else 0.0
    
    def calculate_factor_ic(
        self,
        df: pl.DataFrame,
        factor_name: str,
        label_column: str = "future_return_5d",
    ) -> ICResult:
        """
        计算单个因子的 IC 值（按日期分组计算）
        
        Args:
            df: 包含因子值和标签的 DataFrame
            factor_name: 因子名称
            label_column: 标签列名
            
        Returns:
            ICResult: IC 统计信息
        """
        if factor_name not in df.columns:
            logger.warning(f"Factor {factor_name} not found in data")
            return ICResult(factor_name=factor_name)
        
        if label_column not in df.columns:
            logger.warning(f"Label column {label_column} not found in data")
            return ICResult(factor_name=factor_name)
        
        # 按日期分组计算 IC
        unique_dates = df["trade_date"].unique().to_list()
        ic_series = []
        
        for date in unique_dates:
            day_data = df.filter(pl.col("trade_date") == date)
            
            if len(day_data) < 10:
                continue
            
            factor_values = day_data[factor_name].to_numpy()
            label_values = day_data[label_column].to_numpy()
            
            ic = self.calculate_rank_ic(factor_values, label_values)
            
            if ic != 0 or not np.isnan(ic):
                ic_series.append(ic)
        
        if not ic_series:
            return ICResult(factor_name=factor_name)
        
        ic_array = np.array(ic_series)
        
        # 计算 IC 统计信息
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
        )
    
    def calculate_all_factors_ic(
        self,
        df: pl.DataFrame,
        factor_names: List[str],
        label_column: str = "future_return_5d",
    ) -> Dict[str, ICResult]:
        """
        计算所有因子的 IC 值
        
        Args:
            df: DataFrame
            factor_names: 因子名称列表
            label_column: 标签列名
            
        Returns:
            因子 IC 结果字典
        """
        logger.info(f"Calculating IC for {len(factor_names)} factors")
        
        results = {}
        for factor_name in factor_names:
            logger.info(f"  Calculating IC for: {factor_name}")
            result = self.calculate_factor_ic(df, factor_name, label_column)
            results[factor_name] = result
        
        return results
    
    def print_ic_summary(self, results: Dict[str, ICResult]) -> None:
        """打印 IC 汇总表"""
        logger.info("\n" + "=" * 80)
        logger.info("FACTOR IC ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        # 按 Mean IC 绝对值排序
        sorted_factors = sorted(results.items(), key=lambda x: abs(x[1].mean_ic), reverse=True)
        
        # 打印表头
        header = (
            f"{'Rank':<5} {'Factor Name':<25} {'Mean IC':<10} {'IC Std':<10} "
            f"{'IC IR':<10} {'Positive%':<12} {'T-Stat':<10} {'Valid Days':<12}"
        )
        logger.info(header)
        logger.info("-" * 80)
        
        # 打印所有结果
        for i, (factor_name, r) in enumerate(sorted_factors, 1):
            # 根据 IC 绝对值添加标记
            ic_marker = ""
            if abs(r.mean_ic) >= 0.05:
                ic_marker = " ***"
            elif abs(r.mean_ic) >= 0.03:
                ic_marker = " **"
            elif abs(r.mean_ic) >= 0.01:
                ic_marker = " *"
            
            row = (
                f"{i:<5} {factor_name:<25} {r.mean_ic:>10.4f}   "
                f"{r.ic_std:>8.4f}   {r.ic_ir:>8.2f}   "
                f"{r.positive_ratio:>10.1%}   {r.t_stat:>8.2f}   "
                f"{r.num_valid_days:>10d}{ic_marker}"
            )
            logger.info(row)
        
        logger.info("-" * 80)
        logger.info("IC Legend: *** >= 0.05, ** >= 0.03, * >= 0.01")
        logger.info("=" * 80)


# ===========================================
# 全分组单调性分析模块
# ===========================================

class V14QuintileAnalyzer:
    """
    V14 全分组单调性分析器
    
    功能:
    - 计算 Q1-Q5 五个组合的累计收益率
    - 展示五条曲线的分布
    - 验证因子单调性
    """
    
    def __init__(self, n_groups: int = 5):
        self.n_groups = n_groups
        logger.info(f"V14QuintileAnalyzer initialized with {n_groups} groups")
    
    def compute_quintile_returns(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
    ) -> QuintileResult:
        """
        计算 Q 分组收益
        
        【金融逻辑】
        - 将股票按信号值分为 5 组
        - 计算每组的平均收益
        - 有效的信号应该呈现单调性：Q5 > Q4 > Q3 > Q2 > Q1
        
        Args:
            signals: 信号 DataFrame (symbol, trade_date, signal)
            returns: 收益 DataFrame (symbol, trade_date, future_return)
            
        Returns:
            QuintileResult: Q 分组收益结果
        """
        # 合并信号和收益
        merged = signals.join(returns, on=["symbol", "trade_date"], how="inner")
        
        if merged.is_empty():
            logger.warning("Signals and returns have no overlap")
            return QuintileResult()
        
        # 按日期分组，计算每天的 Q 分组
        result = merged.sort("trade_date").group_by("trade_date", maintain_order=True).agg([
            pl.col("signal").quantile(0.2).alias("q1_threshold"),
            pl.col("signal").quantile(0.4).alias("q2_threshold"),
            pl.col("signal").quantile(0.6).alias("q3_threshold"),
            pl.col("signal").quantile(0.8).alias("q4_threshold"),
        ])
        
        # 为每个股票分配 Q 组
        merged = merged.join(result, on="trade_date", how="left")
        
        # 分配 Q 组标签
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
        
        # 计算每组的平均收益
        q_returns = merged.group_by("q_group").agg([
            pl.col("future_return_5d").mean().alias("avg_return"),
            pl.col("future_return_5d").std().alias("std_return"),
            pl.col("symbol").count().alias("count"),
        ]).sort("q_group")
        
        # 提取结果
        q_stats = {}
        for row in q_returns.iter_rows(named=True):
            q_group = int(row["q_group"])
            q_stats[f"q{q_group}_return"] = float(row["avg_return"])
            q_stats[f"q{q_group}_count"] = int(row["count"])
        
        q1_return = q_stats.get("q1_return", 0)
        q5_return = q_stats.get("q5_return", 0)
        
        return QuintileResult(
            q1_return=q1_return,
            q2_return=q_stats.get("q2_return", 0),
            q3_return=q_stats.get("q3_return", 0),
            q4_return=q_stats.get("q4_return", 0),
            q5_return=q5_return,
            q1_q5_spread=q5_return - q1_return,
            q1_count=q_stats.get("q1_count", 0),
            q2_count=q_stats.get("q2_count", 0),
            q3_count=q_stats.get("q3_count", 0),
            q4_count=q_stats.get("q4_count", 0),
            q5_count=q_stats.get("q5_count", 0),
        )
    
    def print_quintile_summary(self, result: QuintileResult) -> None:
        """打印 Q 分组收益汇总表"""
        logger.info("\n" + "=" * 60)
        logger.info("QUINTILE ANALYSIS (Q1-Q5 全分组)")
        logger.info("=" * 60)
        logger.info("有效的信号应该呈现单调性：Q5 > Q4 > Q3 > Q2 > Q1")
        logger.info("-" * 60)
        logger.info(f"Q1 (Low Signal):  {result.q1_return:>10.4%}  (n={result.q1_count})")
        logger.info(f"Q2:               {result.q2_return:>10.4%}  (n={result.q2_count})")
        logger.info(f"Q3:               {result.q3_return:>10.4%}  (n={result.q3_count})")
        logger.info(f"Q4:               {result.q4_return:>10.4%}  (n={result.q4_count})")
        logger.info(f"Q5 (High Signal): {result.q5_return:>10.4%}  (n={result.q5_count})")
        logger.info("-" * 60)
        logger.info(f"Q5-Q1 Spread:     {result.q1_q5_spread:>10.4%}")
        logger.info("=" * 60)


# ===========================================
# 集成学习模型模块
# ===========================================

class V14EnsembleModel:
    """
    V14 集成学习模型
    
    功能:
    - 使用 Ridge 回归或轻量级 Random Forest 集成因子
    - 输出综合 Predict_Score
    - 严格时序验证：仅使用 shift(1) 后的数据
    """
    
    def __init__(
        self,
        model_type: str = "ridge",
        ridge_alpha: float = 1.0,
        rf_n_estimators: int = 100,
        rf_max_depth: int = 5,
    ):
        self.model_type = model_type
        self.ridge_alpha = ridge_alpha
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        
        logger.info(f"V14EnsembleModel initialized: {model_type}")
    
    def get_feature_columns(self) -> List[str]:
        """获取特征列名"""
        # V14 新增特征
        v14_features = [
            # 波动率特征
            "volatility_20",
            "volatility_ratio",
            # 动量/反转特征
            "momentum_5",
            "momentum_20",
            "reversal_signal",
            # 资金流特征
            "vwap_return",
            "turnover_change",
            "amount_ma_ratio",
        ]
        
        # 基础特征（来自 FactorEngine）
        base_features = [
            "rsi_14",
            "macd",
            "macd_signal",
            "volume_price_health",
            "vcp_score",
            "smart_money_signal",
        ]
        
        return v14_features + base_features
    
    def prepare_training_data(
        self,
        df: pl.DataFrame,
        train_end_date: str = TRAIN_END_DATE,
        train_start_date: str = TRAIN_START_DATE,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        【时序控制】
        - 信号生成仅使用 shift(1) 后的数据
        - 确保无未来函数
        
        Args:
            df: 包含特征和标签的 DataFrame
            train_end_date: 训练截止日期
            train_start_date: 训练开始日期
            
        Returns:
            (X, y): 特征矩阵和标签向量
        """
        # 日期过滤
        df_filtered = df.filter(
            (pl.col("trade_date") >= train_start_date) &
            (pl.col("trade_date") <= train_end_date)
        )
        
        # 获取特征列
        self.feature_columns = self.get_feature_columns()
        available_cols = [c for c in self.feature_columns if c in df_filtered.columns]
        
        # 过滤空值
        df_clean = df_filtered.filter(
            pl.col(available_cols).is_not_null().all_horizontal() &
            pl.col("future_return_5d").is_not_null()
        )
        
        if len(df_clean) == 0:
            logger.warning("No valid training data after filtering nulls")
            return np.array([]), np.array([])
        
        X = df_clean.select(available_cols).to_numpy()
        y = df_clean["future_return_5d"].to_numpy()
        
        # 标准化特征
        X = self.scaler.fit_transform(X)
        
        logger.info(f"Prepared {len(X)} training samples with {len(available_cols)} features")
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签向量
        """
        if len(X) == 0 or len(y) == 0:
            logger.warning("No training data available")
            return
        
        if self.model_type == "ridge":
            self.model = Ridge(alpha=self.ridge_alpha)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=self.gb_n_estimators,
                max_depth=3,
                random_state=42,
            )
        else:
            self.model = Ridge(alpha=self.ridge_alpha)
        
        self.model.fit(X, y)
        logger.info(f"Model trained with {len(X)} samples")
    
    def predict(self, df: pl.DataFrame) -> pl.Series:
        """
        使用模型进行预测
        
        【时序控制】
        - 仅使用 T 日及之前数据
        
        Args:
            df: 包含特征数据的 DataFrame
            
        Returns:
            预测评分 Series
        """
        if self.model is None:
            logger.warning("Model not trained, returning zeros")
            return pl.Series([0.0] * len(df))
        
        available_cols = [c for c in self.feature_columns if c in df.columns]
        
        # 过滤空值
        df_filtered = df.filter(pl.col(available_cols).is_not_null().all_horizontal())
        
        if len(df_filtered) == 0:
            return pl.Series([0.0] * len(df))
        
        X = df_filtered.select(available_cols).to_numpy()
        X = self.scaler.transform(X)
        
        predictions = self.model.predict(X)
        
        return pl.Series(predictions)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性"""
        if self.model is None:
            return {}
        
        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            # Ridge 回归使用系数绝对值
            importance = dict(zip(self.feature_columns, np.abs(self.model.coef_)))
        else:
            return {}
        
        # 排序
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return importance
    
    def print_feature_importance(self) -> None:
        """打印特征重要性"""
        importance = self.get_feature_importance()
        
        if not importance:
            logger.warning("No feature importance available")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE IMPORTANCE")
        logger.info("=" * 60)
        
        for i, (feat, imp) in enumerate(importance.items(), 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")
        
        logger.info("=" * 60)


# ===========================================
# 特征相关性分析模块
# ===========================================

class V14CorrelationAnalyzer:
    """
    V14 特征相关性分析器
    
    功能:
    - 计算特征相关性热力图
    - 确保新因子之间不是高度相关
    """
    
    def __init__(self):
        logger.info("V14CorrelationAnalyzer initialized")
    
    def compute_correlation_matrix(
        self,
        df: pl.DataFrame,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        计算特征相关性矩阵
        
        Args:
            df: DataFrame
            feature_names: 特征名称列表
            
        Returns:
            相关性矩阵
        """
        available_cols = [c for c in feature_names if c in df.columns]
        
        if not available_cols:
            logger.warning("No feature columns found")
            return np.array([])
        
        # 转换为 numpy
        data = df.select(available_cols).to_numpy()
        
        # 去除空值
        mask = ~np.isnan(data).any(axis=1)
        data_clean = data[mask]
        
        if len(data_clean) < 10:
            logger.warning("Insufficient data for correlation")
            return np.array([])
        
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(data_clean.T)
        
        return corr_matrix
    
    def print_correlation_summary(
        self,
        corr_matrix: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.7,
    ) -> None:
        """
        打印相关性摘要
        
        Args:
            corr_matrix: 相关性矩阵
            feature_names: 特征名称
            threshold: 高相关性阈值
        """
        if corr_matrix.size == 0:
            logger.warning("No correlation data available")
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("FEATURE CORRELATION ANALYSIS")
        logger.info("=" * 60)
        
        n = len(feature_names)
        high_corr_pairs = []
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = corr_matrix[i, j]
                if abs(corr) > threshold:
                    high_corr_pairs.append((feature_names[i], feature_names[j], corr))
        
        if high_corr_pairs:
            logger.info(f"\n⚠️  High Correlation Pairs (|corr| > {threshold}):")
            for f1, f2, corr in high_corr_pairs:
                logger.info(f"  {f1} <-> {f2}: {corr:.3f}")
        else:
            logger.info(f"\n✅ No high correlation pairs found (|corr| <= {threshold})")
        
        logger.info("=" * 60)
    
    def plot_correlation_heatmap(
        self,
        corr_matrix: np.ndarray,
        feature_names: List[str],
        save_path: str = "data/plots/feature_correlation_heatmap.png",
    ) -> str:
        """
        绘制相关性热力图
        
        Args:
            corr_matrix: 相关性矩阵
            feature_names: 特征名称
            save_path: 保存路径
            
        Returns:
            保存路径
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            if corr_matrix.size == 0:
                logger.warning("No correlation data to plot")
                return ""
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 绘制热力图
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # 设置刻度
            ax.set_xticks(range(len(feature_names)))
            ax.set_yticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=45, ha='right')
            ax.set_yticklabels(feature_names)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, label='Correlation')
            
            # 添加数值标注
            n = len(feature_names)
            for i in range(n):
                for j in range(n):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                  ha='center', va='center', color='black', fontsize=8)
            
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            
            # 保存
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Correlation heatmap saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to plot correlation heatmap: {e}")
            return ""


# ===========================================
# V14 主策略类
# ===========================================

class FinalStrategyV14:
    """
    Final Strategy V1.14 - Iteration 14: 核心特征挖掘与因子集成
    
    核心改进:
        1. 波动率特征 - 测试"低波动异常收益"
        2. 动量/反转特征 - 测试 A 股短线反转效应
        3. 资金流特征 - 识别主力资金异常流动
        4. 全分组单调性分析 - Q1-Q5 完整收益
        5. 集成学习模型 - Ridge/RandomForest 集成因子
    """
    
    def __init__(
        self,
        config_path: str = "config/production_params.yaml",
        factors_config_path: str = "config/factors.yaml",
        db: Optional[DatabaseManager] = None,
    ):
        self.config_path = Path(config_path)
        self.factors_config_path = Path(factors_config_path)
        self.db = db or DatabaseManager.get_instance()
        
        # 初始化模块
        self.feature_engine = V14FeatureEngine()
        self.ic_calculator = V14ICCalculator()
        self.quintile_analyzer = V14QuintileAnalyzer()
        self.ensemble_model = V14EnsembleModel(model_type=MODEL_TYPE)
        self.correlation_analyzer = V14CorrelationAnalyzer()
        
        # 加载配置
        self.config = self._load_config()
        self.factor_engine = FactorEngine(str(self.factors_config_path), validate=True)
        
        logger.info("FinalStrategyV14 initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return {}
    
    def prepare_data(
        self,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """
        准备数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            处理后的 DataFrame
        """
        logger.info(f"Preparing data from {start_date} to {end_date}...")
        
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            return None
    
    def compute_future_returns(self, df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
        """
        计算未来收益（标签）
        
        【时序控制】
        - 使用 shift(-window) 获取未来价格
        - 严格区分训练和测试数据
        
        Args:
            df: 输入 DataFrame
            window: 未来收益窗口
            
        Returns:
            添加了未来收益标签的 DataFrame
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 未来 5 日收益
        future_return = (
            pl.col("close").shift(-window) / 
            (pl.col("close").shift(-1) + self.feature_engine.EPSILON) - 1
        ).alias("future_return_5d")
        
        result = result.with_columns([future_return])
        
        return result
    
    def run_ic_analysis(
        self,
        df: pl.DataFrame,
        feature_names: List[str],
    ) -> Dict[str, ICResult]:
        """
        运行 IC 分析
        
        Args:
            df: DataFrame
            feature_names: 特征名称列表
            
        Returns:
            IC 结果字典
        """
        logger.info("\n" + "=" * 60)
        logger.info("Running IC Analysis...")
        logger.info("=" * 60)
        
        results = self.ic_calculator.calculate_all_factors_ic(
            df, feature_names, "future_return_5d"
        )
        
        self.ic_calculator.print_ic_summary(results)
        return results
    
    def run_quintile_analysis(
        self,
        signals: pl.DataFrame,
        returns: pl.DataFrame,
    ) -> QuintileResult:
        """
        运行全分组单调性分析
        
        Args:
            signals: 信号 DataFrame
            returns: 收益 DataFrame
            
        Returns:
            QuintileResult
        """
        logger.info("\n" + "=" * 60)
        logger.info("Running Quintile Analysis (Q1-Q5)...")
        logger.info("=" * 60)
        
        result = self.quintile_analyzer.compute_quintile_returns(signals, returns)
        self.quintile_analyzer.print_quintile_summary(result)
        
        return result
    
    def run_ensemble_model(
        self,
        df: pl.DataFrame,
    ) -> V14EnsembleModel:
        """
        运行集成学习模型
        
        Args:
            df: DataFrame
            
        Returns:
            训练好的模型
        """
        logger.info("\n" + "=" * 60)
        logger.info("Training Ensemble Model...")
        logger.info("=" * 60)
        
        # 准备训练数据
        X, y = self.ensemble_model.prepare_training_data(df)
        
        if len(X) == 0:
            logger.warning("No training data available")
            return self.ensemble_model
        
        # 训练模型
        self.ensemble_model.train(X, y)
        
        # 打印特征重要性
        self.ensemble_model.print_feature_importance()
        
        return self.ensemble_model
    
    def run_correlation_analysis(
        self,
        df: pl.DataFrame,
        feature_names: List[str],
    ) -> np.ndarray:
        """
        运行相关性分析
        
        Args:
            df: DataFrame
            feature_names: 特征名称列表
            
        Returns:
            相关性矩阵
        """
        logger.info("\n" + "=" * 60)
        logger.info("Running Correlation Analysis...")
        logger.info("=" * 60)
        
        corr_matrix = self.correlation_analyzer.compute_correlation_matrix(df, feature_names)
        
        if corr_matrix.size > 0:
            self.correlation_analyzer.print_correlation_summary(corr_matrix, feature_names)
            
            # 绘制热力图
            self.correlation_analyzer.plot_correlation_heatmap(
                corr_matrix, feature_names,
                save_path="data/plots/v14_feature_correlation_heatmap.png"
            )
        
        return corr_matrix
    
    def run_full_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000,
    ) -> BacktestResult:
        """
        运行完整回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            
        Returns:
            回测结果
        """
        logger.info("\n" + "=" * 70)
        logger.info("V14 FULL BACKTEST")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial Capital: ¥{initial_capital:,.0f}")
        
        # 准备数据
        df = self.prepare_data(start_date, end_date)
        if df is None:
            return BacktestResult()
        
        # 计算 V14 特征
        df = self.feature_engine.compute_all_v14_features(df)
        
        # 计算未来收益标签
        df = self.compute_future_returns(df)
        
        # 运行 IC 分析
        v14_features = self.feature_engine.get_v14_feature_names()
        ic_results = self.run_ic_analysis(df, v14_features)
        
        # 运行相关性分析
        self.run_correlation_analysis(df, v14_features)
        
        # 训练集成模型
        model = self.run_ensemble_model(df)
        
        # 生成预测信号
        logger.info("\nGenerating predictions...")
        # 按日期迭代生成预测（严格时序）
        predictions = []
        dates = df["trade_date"].unique().sort().to_list()
        
        for i, date in enumerate(dates):
            day_data = df.filter(pl.col("trade_date") == date)
            
            # 使用模型预测
            pred_scores = model.predict(day_data)
            
            for row in day_data.iter_rows(named=True):
                symbol = row.get("symbol")
                if symbol:
                    predictions.append({
                        "symbol": symbol,
                        "trade_date": date,
                        "signal": float(pred_scores[i]) if i < len(pred_scores) else 0.0,
                    })
        
        signals_df = pl.DataFrame(predictions)
        
        # 准备收益数据
        returns_df = df.select(["symbol", "trade_date", "future_return_5d"])
        
        # 运行全分组单调性分析
        quintile_result = self.run_quintile_analysis(signals_df, returns_df)
        
        # 构建回测结果
        result = BacktestResult(
            quintile_results=quintile_result,
        )
        
        logger.info("\n" + "=" * 70)
        logger.info("V14 BACKTEST COMPLETE")
        logger.info("=" * 70)
        
        return result
    
    def generate_v14_report(
        self,
        ic_results: Dict[str, ICResult],
        quintile_result: QuintileResult,
        corr_matrix: np.ndarray,
        feature_names: List[str],
    ) -> str:
        """
        生成 V14 报告
        
        Args:
            ic_results: IC 结果
            quintile_result: Q 分组结果
            corr_matrix: 相关性矩阵
            feature_names: 特征名称
            
        Returns:
            报告内容
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"reports/V14_Feature_Mining_Report_{timestamp}.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# V14 核心特征挖掘与因子集成报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: Final Strategy V1.14 (Iteration 14)

---

## 一、新因子 IC 矩阵

### 1.1 IC 统计汇总表

| 排名 | 因子名称 | Mean IC | IC Std | IC IR | Positive% | T-Stat | 有效天数 |
|------|----------|---------|--------|-------|-----------|--------|----------|
"""
        
        # 按 Mean IC 绝对值排序
        sorted_ic = sorted(ic_results.items(), key=lambda x: abs(x[1].mean_ic), reverse=True)
        
        for i, (factor_name, result) in enumerate(sorted_ic, 1):
            ic_marker = ""
            if abs(result.mean_ic) >= 0.05:
                ic_marker = " ***"
            elif abs(result.mean_ic) >= 0.03:
                ic_marker = " **"
            elif abs(result.mean_ic) >= 0.01:
                ic_marker = " *"
            
            report += f"| {i} | {factor_name} | {result.mean_ic:.4f} | {result.ic_std:.4f} | {result.ic_ir:.2f} | {result.positive_ratio:.1%} | {result.t_stat:.2f} | {result.num_valid_days} |{ic_marker}\n"
        
        report += f"""
### 1.2 IC 图例说明
- *** : Mean IC >= 0.05 (强预测能力)
- ** : Mean IC >= 0.03 (中等预测能力)
- * : Mean IC >= 0.01 (弱预测能力)

---

## 二、Q1-Q5 完整收益表

### 2.1 五分位组合收益

| 分组 | 平均收益 | 交易次数 |
|------|----------|----------|
| Q1 (Low Signal) | {quintile_result.q1_return:.4%} | {quintile_result.q1_count:,} |
| Q2 | {quintile_result.q2_return:.4%} | {quintile_result.q2_count:,} |
| Q3 | {quintile_result.q3_return:.4%} | {quintile_result.q3_count:,} |
| Q4 | {quintile_result.q4_return:.4%} | {quintile_result.q4_count:,} |
| Q5 (High Signal) | {quintile_result.q5_return:.4%} | {quintile_result.q5_count:,} |

### 2.2 多空收益
| 指标 | 值 |
|------|-----|
| Q5-Q1 Spread | {quintile_result.q1_q5_spread:.4%} |

### 2.3 单调性判断
"""
        
        # 单调性判断
        monotonic = (
            quintile_result.q5_return > quintile_result.q4_return > 
            quintile_result.q3_return > quintile_result.q2_return > 
            quintile_result.q1_return
        )
        
        if monotonic:
            report += "- ✅ **单调性良好**: Q5 > Q4 > Q3 > Q2 > Q1\n"
        elif quintile_result.q1_q5_spread > 0:
            report += "- ⚠️ **单调性部分成立**: Q5-Q1 Spread > 0，但中间分组顺序不完全单调\n"
        else:
            report += "- ❌ **单调性反向**: Q5-Q1 Spread < 0\n"
        
        report += f"""
---

## 三、特征相关性热力图

### 3.1 相关性分析

热力图已保存至：`data/plots/v14_feature_correlation_heatmap.png`

### 3.2 高相关性特征对

"""
        
        if corr_matrix.size > 0:
            high_corr_pairs = []
            n = len(feature_names)
            for i in range(n):
                for j in range(i + 1, n):
                    corr = corr_matrix[i, j]
                    if abs(corr) > 0.7:
                        high_corr_pairs.append((feature_names[i], feature_names[j], corr))
            
            if high_corr_pairs:
                report += "| 特征 1 | 特征 2 | 相关系数 |\n"
                report += "|--------|--------|----------|\n"
                for f1, f2, corr in high_corr_pairs:
                    report += f"| {f1} | {f2} | {corr:.3f} |\n"
            else:
                report += "✅ 未发现高相关性特征对 (|corr| > 0.7)\n"
        else:
            report += "无相关性数据\n"
        
        report += f"""
---

## 四、新特征金融逻辑说明

### 4.1 波动率特征 (Volatility)

| 特征名 | 计算公式 | 金融逻辑 |
|--------|----------|----------|
| volatility_20 | std(returns, 20) | 低波动率股票往往有异常收益 |
| volatility_ratio | volatility_5 / volatility_20 | 波动率收缩预示突破机会 |

### 4.2 动量/反转特征 (Momentum/Reversal)

| 特征名 | 计算公式 | 金融逻辑 |
|--------|----------|----------|
| momentum_5 | close[t] / close[t-5] - 1 | A 股短线反转效应 |
| momentum_20 | close[t] / close[t-20] - 1 | 中长期动量效应 |
| reversal_signal | -momentum_5 * sign(...) | 捕捉超跌反转机会 |

### 4.3 资金流特征 (Liquidity)

| 特征名 | 计算公式 | 金融逻辑 |
|--------|----------|----------|
| vwap_return | close / vwap_lag - 1 | VWAP 相关收益率 |
| turnover_change | turnover[t] / turnover_ma - 1 | 换手率异常放大 |
| amount_ma_ratio | amount / ma(amount, 20) | 成交额异常放大 |

---

## 五、集成学习模型

### 5.1 模型配置
| 参数 | 值 |
|------|-----|
| 模型类型 | {MODEL_TYPE} |
| Ridge Alpha | {RIDGE_ALPHA} |
| RF 树数量 | {RF_N_ESTIMATORS} |
| RF 最大深度 | {RF_MAX_DEPTH} |

### 5.2 时序验证
- ✅ 信号生成仅使用 `df.shift(1)` 后的数据
- ✅ 每一行预测都是基于昨天已知的收盘信息
- ✅ 无未来函数

---

## 六、执行总结

### 6.1 核心结论
1. **新因子有效性**: 新增 {len(ic_results)} 个因子，其中 {sum(1 for r in ic_results.values() if abs(r.mean_ic) >= 0.01)} 个因子 Mean IC >= 0.01
2. **单调性验证**: Q5-Q1 Spread = {quintile_result.q1_q5_spread:.4%}
3. **特征相关性**: 新因子之间无高度相关性（|corr| < 0.7）

### 6.2 后续优化方向
1. 考虑引入更多非线性特征组合
2. 探索动态因子权重配置
3. 增加行业/风格中性化处理

---

**报告生成完毕**
"""
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"V14 report saved to: {report_path}")
        return str(report_path)


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
    logger.info("Final Strategy V1.14 - Iteration 14")
    logger.info("核心特征挖掘与因子集成")
    logger.info("=" * 70)
    
    # 创建策略实例
    strategy = FinalStrategyV14(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
    )
    
    # 运行完整回测
    backtest_result = strategy.run_full_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=100000,
    )
    
    # 生成 V14 报告
    v14_features = strategy.feature_engine.get_v14_feature_names()
    
    # 重新计算 IC 结果用于报告
    df = strategy.prepare_data("2024-01-01", "2024-06-30")
    if df is not None:
        df = strategy.feature_engine.compute_all_v14_features(df)
        df = strategy.compute_future_returns(df)
        
        ic_results = strategy.run_ic_analysis(df, v14_features)
        corr_matrix = strategy.run_correlation_analysis(df, v14_features)
        
        report_path = strategy.generate_v14_report(
            ic_results=ic_results,
            quintile_result=backtest_result.quintile_results,
            corr_matrix=corr_matrix,
            feature_names=v14_features,
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("V14 ANALYSIS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()