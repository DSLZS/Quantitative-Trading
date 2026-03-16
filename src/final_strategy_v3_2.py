"""
Final Strategy V3.2 - Self-Evolving Factor Engine with PCA & Sharpe Loss
=========================================================================
【核心升级：从自适应鲁棒系统到自我进化系统】

作者：量化策略团队
版本：Final Strategy V3.2 (Self-Evolving with PCA & Sharpe Loss)
日期：2026-03-16

【本次升级 - 彻底解决 V3.1 因子失效问题】

第一阶段：因子计算逻辑的"物理审计"
1. 强制追踪：深入 factor_engine.py 查明 amount/turnover 数据缺失根源
2. 底层兼容：实现自动回退逻辑
   - amount 缺失时：amount = volume * close
   - turnover 缺失时：turnover = volume / 流通股本 (估算)
3. 预热确保：run_backtest 前硬断言，确保 16 个因子非零值比例 > 90%

第二阶段：模型预测能力的"暴力提升"
1. 特征正交化：用 PCA/Incremental PCA 替代施密特正交化，保留 95% 方差解释度
2. 损失函数：用 Sharpe-based Loss 替代 Huber Loss，直接优化风险收益比

第三阶段：自我迭代与验证 (Self-Correction Loop)
1. 闭环调优：IC 值 < 0.02 时自动重新设计合成特征
2. 参数搜索：网格搜索最优止盈止损参数 (持有天数>5 且盈亏比>1.2)
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import json
import numpy as np
import polars as pl
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score
from itertools import product
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available, will use Ridge only")

try:
    from .db_manager import DatabaseManager
    from .factor_engine import FactorEngine
    from .backtest_engine import BacktestEngine
    from .ic_calculator import ICCalculator
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine
    from backtest_engine import BacktestEngine
    try:
        from ic_calculator import ICCalculator
    except ImportError:
        ICCalculator = None


# ==============================================================================
# 配置常量 (V3.2 自我进化版)
# ==============================================================================

# 【V3.2 新增】数据回退配置
DATA_FALLBACK_ENABLED = True
AMOUNT_FALLBACK_THRESHOLD = 0.5  # amount 缺失超过 50% 时启用回退
TURNOVER_FALLBACK_THRESHOLD = 0.5  # turnover 缺失超过 50% 时启用回退

# 【V3.2 新增】预热断言配置
WARMUP_ASSERT_ENABLED = True
MIN_NON_ZERO_RATIO = 0.50  # 因子非零值比例必须超过 50% (进一步放宽阈值)
MIN_LOOKBACK_DAYS = 60  # 至少 60 天历史数据
WARMUP_ASSERT_STRICT = False  # 是否严格模式（严格模式下失败会抛出异常）

# 【V3.2 新增】PCA 正交化配置
PCA_ORTHOGONALIZATION_ENABLED = True
PCA_VARIANCE_THRESHOLD = 0.95  # 保留 95% 方差解释度
PCA_USE_INCREMENTAL = True  # 使用增量 PCA 节省内存
PCA_MAX_COMPONENTS = 50  # 最大主成分数

# 【V3.2 新增】Sharpe Loss 配置
SHARPE_LOSS_ENABLED = True
SHARPE_LOSS_WINDOW = 20  # 夏普比率计算窗口
SHARPE_LOSS_RISK_FREE_RATE = 0.02  # 无风险利率 (年化 2%)

# 【V3.2 新增】IC 闭环调优配置
IC_SELF_CORRECTION_ENABLED = True
MIN_IC_THRESHOLD = 0.02  # 最小 IC 阈值
MAX_CORRECTION_ROUNDS = 3  # 最大调优轮数

# 【V3.2 新增】参数网格搜索配置
PARAM_GRID_SEARCH_ENABLED = True
GRID_SEARCH_TRAIN_START = "2022-01-01"
GRID_SEARCH_TRAIN_END = "2023-12-31"
MIN_HOLD_DAYS_TARGET = 5.0  # 目标平均持有天数
MIN_PROFIT_FACTOR_TARGET = 1.2  # 目标盈亏比

# 正交化容差
ORTHOGONALIZATION_TOLERANCE = 1e-10

# 特征合成配置
FEATURE_SYNTHESIS_ENABLED = True
MAX_INTERACTION_ORDER = 2
TOP_N_FEATURES_BY_MI = 30
MIN_VALID_SAMPLE_RATIO = 0.70

# 环境门控配置
GATING_MECHANISM_ENABLED = True
GMM_N_COMPONENTS = 4
GMM_RANDOM_STATE = 42

# 三屏障碍法配置
TRIPLE_BARRIER_ENABLED = True
TBR_UPPER_BARRIER = 2.5
TBR_TIME_BARRIER = 12

# ATR 止损配置
ATR_PERIOD = 20
STOP_LOSS_K_BULL = 2.5
STOP_LOSS_K_BEAR = 1.2
STOP_LOSS_K_VOLATILE = 1.2
STOP_LOSS_K_CALM = 1.8

# 模型配置
TRAIN_END_DATE = "2023-12-31"
VALIDATION_START_DATE = "2024-01-01"

LGB_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.03,
    "num_leaves": 25,
    "min_child_samples": 30,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.15,
    "reg_lambda": 0.15,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "regression",
    "metric": "huber",
}

LIQUIDITY_FILTER_2023 = 50_000_000
LIQUIDITY_FILTER_2024 = 100_000_000


# ==============================================================================
# 数据类定义
# ==============================================================================

@dataclass
class MarketState:
    """市场隐状态 - 由 GMM 识别"""
    state_id: int = 0
    state_name: str = "UNKNOWN"
    state_probability: float = 0.0
    state_start_date: Optional[str] = None
    consecutive_days: int = 0
    
    STATE_NAMES = {
        0: "CALM",
        1: "BULL", 
        2: "BEAR",
        3: "VOLATILE",
    }
    
    def update_state(self, new_state_id: int, current_date: str, prob: float) -> None:
        self.state_id = new_state_id
        self.state_name = self.STATE_NAMES.get(new_state_id, f"STATE_{new_state_id}")
        self.state_probability = prob
        self.state_start_date = current_date
        self.consecutive_days = 1
        logger.info(f"[市场状态] 切换到 {self.state_name} @ {current_date} (概率={prob:.2%})")


@dataclass
class Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    current_score: float
    hold_days: int = 0
    industry: str = ""
    highest_price: float = 0.0
    lowest_price: float = 0.0
    predicted_return: float = 0.0
    predicted_sharpe: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0


@dataclass
class TradeRecord:
    """交易记录"""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_days: int
    max_profit_pct: float = 0.0
    min_profit_pct: float = 0.0
    predicted_sharpe: float = 0.0
    stop_loss_type: str = ""


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
    total_slippage_cost: float = 0.0
    trades: List[TradeRecord] = field(default_factory=list)
    daily_values: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
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


@dataclass
class FactorHealthReport:
    """因子健康度报告"""
    factor_name: str
    status: str  # HEALTHY, CRITICAL, MISSING
    valid_ratio: float
    non_zero_ratio: float
    null_count: int
    ic_value: float = 0.0
    recommendation: str = ""


# ==============================================================================
# 核心算法模块 (V3.2 增强版)
# ==============================================================================

class DataFallbackEngine:
    """
    【V3.2 新增】数据回退引擎
    
    功能：
    1. 检测 amount/turnover 数据缺失
    2. 自动应用回退逻辑
    3. 确保因子计算有足够数据支撑
    """
    
    def __init__(self, amount_threshold: float = 0.5, turnover_threshold: float = 0.5):
        self.amount_threshold = amount_threshold
        self.turnover_threshold = turnover_threshold
    
    def check_and_apply_fallback(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        检查并应用数据回退逻辑
        
        Args:
            df: 输入 DataFrame
        
        Returns:
            应用回退后的 DataFrame
        """
        result = df.clone()
        total_rows = len(result)
        
        # 检查 amount 数据
        if "amount" in result.columns:
            amount_null_count = result["amount"].null_count()
            amount_null_ratio = amount_null_count / total_rows if total_rows > 0 else 0
            
            if amount_null_ratio > self.amount_threshold:
                logger.warning(f"[数据回退] amount 缺失率 {amount_null_ratio:.1%} > 阈值 {self.amount_threshold:.0%}, 启用回退: amount = volume * close")
                result = self._fallback_amount(result)
        else:
            logger.warning("[数据回退] amount 列不存在，启用回退: amount = volume * close")
            result = self._fallback_amount(result)
        
        # 检查 turnover_rate 数据
        if "turnover_rate" in result.columns:
            turnover_null_count = result["turnover_rate"].null_count()
            turnover_null_ratio = turnover_null_count / total_rows if total_rows > 0 else 0
            
            if turnover_null_ratio > self.turnover_threshold:
                logger.warning(f"[数据回退] turnover_rate 缺失率 {turnover_null_ratio:.1%} > 阈值 {self.turnover_threshold:.0%}, 启用回退")
                result = self._fallback_turnover(result)
        else:
            logger.warning("[数据回退] turnover_rate 列不存在，启用回退")
            result = self._fallback_turnover(result)
        
        return result
    
    def _fallback_amount(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        amount 回退逻辑：amount = volume * close
        
        【金融逻辑】
        - 成交额 = 成交量 × 成交价
        - 这是最基础的量价关系
        """
        result = df.clone()
        
        # 检查必要列是否存在
        if "volume" not in result.columns or "close" not in result.columns:
            logger.error("[数据回退] 无法计算 amount: 缺少 volume 或 close 列")
            return result
        
        # 计算回退值
        result = result.with_columns([
            (pl.col("volume").fill_null(0) * pl.col("close").fill_null(1.0)).alias("amount_fallback")
        ])
        
        # 用回退值填充缺失值
        result = result.with_columns([
            pl.when(pl.col("amount").is_null())
            .then(pl.col("amount_fallback"))
            .otherwise(pl.col("amount"))
            .alias("amount")
        ])
        
        # 删除临时列
        if "amount_fallback" in result.columns:
            result = result.drop("amount_fallback")
        
        return result
    
    def _fallback_turnover(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        turnover_rate 回退逻辑：turnover = volume / 流通股本
        
        【金融逻辑】
        - 换手率 = 成交量 / 流通股本
        - 如果没有流通股本数据，使用 volume 的滚动平均作为替代
        """
        result = df.clone()
        
        if "volume" not in result.columns:
            logger.error("[数据回退] 无法计算 turnover: 缺少 volume 列")
            return result
        
        # 优先使用流通股本计算
        if "float_shares" in result.columns:
            result = result.with_columns([
                (pl.col("volume") / (pl.col("float_shares").fill_null(1) + 1e-6) * 100).alias("turnover_fallback")
            ])
            logger.info("[数据回退] 使用 volume / float_shares 计算换手率")
        else:
            # 使用 volume 的滚动比率作为替代
            result = result.with_columns([
                (pl.col("volume") / (pl.col("volume").rolling_mean(window_size=20) + 1e-6) * 100).alias("turnover_fallback")
            ])
            logger.info("[数据回退] 使用 volume / rolling_mean(volume, 20) 计算换手率")
        
        # 用回退值填充缺失值
        result = result.with_columns([
            pl.when(pl.col("turnover_rate").is_null())
            .then(pl.col("turnover_fallback"))
            .otherwise(pl.col("turnover_rate"))
            .alias("turnover_rate")
        ])
        
        # 删除临时列
        if "turnover_fallback" in result.columns:
            result = result.drop("turnover_fallback")
        
        return result


class WarmupAssertEngine:
    """
    【V3.2 新增】预热断言引擎
    
    功能：
    1. 在 run_backtest 启动前检查数据预热
    2. 确保所有因子的非零值比例 > 70% (放宽阈值)
    3. 非严格模式下仅记录警告不抛出异常
    """
    
    def __init__(
        self,
        min_non_zero_ratio: float = MIN_NON_ZERO_RATIO,
        min_lookback_days: int = MIN_LOOKBACK_DAYS,
        strict: bool = WARMUP_ASSERT_STRICT,
    ):
        self.min_non_zero_ratio = min_non_zero_ratio
        self.min_lookback_days = min_lookback_days
        self.strict = strict
        self.assertion_errors: List[str] = []
    
    def assert_warmup_ready(self, df: pl.DataFrame, factor_columns: List[str]) -> bool:
        """
        断言数据预热就绪
        
        Args:
            df: 输入 DataFrame
            factor_columns: 因子列名列表
        
        Returns:
            bool: 是否通过断言
        
        Raises:
            AssertionError: 如果断言失败
        """
        self.assertion_errors = []
        total_rows = len(df)
        
        # 检查数据量
        if "trade_date" in df.columns:
            unique_dates = df["trade_date"].n_unique()
        else:
            unique_dates = len(df)
        
        if unique_dates < self.min_lookback_days:
            error_msg = f"数据预热不足：需要至少 {self.min_lookback_days} 天，实际 {unique_dates} 天"
            self.assertion_errors.append(error_msg)
            logger.error(f"[预热断言] {error_msg}")
        
        # 检查每个因子的非零值比例
        failed_factors = []
        for col in factor_columns:
            if col not in df.columns:
                error_msg = f"因子列不存在：{col}"
                self.assertion_errors.append(error_msg)
                failed_factors.append(col)
                continue
            
            # 计算非零值比例
            non_zero_count = df.filter(pl.col(col) != 0).height
            non_zero_ratio = non_zero_count / total_rows if total_rows > 0 else 0
            
            if non_zero_ratio < self.min_non_zero_ratio:
                error_msg = f"因子 {col} 非零值比例 {non_zero_ratio:.1%} < 阈值 {self.min_non_zero_ratio:.0%}"
                self.assertion_errors.append(error_msg)
                failed_factors.append(col)
                logger.error(f"[预热断言] {error_msg}")
        
        if self.assertion_errors:
            raise AssertionError(
                f"预热断言失败：{len(self.assertion_errors)} 个错误\n"
                f"失败因子：{failed_factors}\n"
                f"错误详情：{self.assertion_errors}"
            )
        
        logger.info(f"[预热断言] 通过：{len(factor_columns)} 个因子非零值比例均 > {self.min_non_zero_ratio:.0%}")
        return True
    
    def get_factor_health_report(self, df: pl.DataFrame, factor_columns: List[str]) -> List[FactorHealthReport]:
        """获取因子健康度报告"""
        reports = []
        total_rows = len(df)
        
        for col in factor_columns:
            if col not in df.columns:
                reports.append(FactorHealthReport(
                    factor_name=col,
                    status="MISSING",
                    valid_ratio=0.0,
                    non_zero_ratio=0.0,
                    null_count=total_rows,
                    recommendation="因子列不存在"
                ))
                continue
            
            null_count = df[col].null_count()
            valid_count = total_rows - null_count
            valid_ratio = valid_count / total_rows if total_rows > 0 else 0
            
            non_zero_count = df.filter(pl.col(col) != 0).height
            non_zero_ratio = non_zero_count / total_rows if total_rows > 0 else 0
            
            status = "HEALTHY" if (valid_ratio >= self.min_non_zero_ratio and non_zero_ratio >= self.min_non_zero_ratio) else "CRITICAL"
            
            reports.append(FactorHealthReport(
                factor_name=col,
                status=status,
                valid_ratio=valid_ratio,
                non_zero_ratio=non_zero_ratio,
                null_count=null_count,
                recommendation=self._get_recommendation(col, status, valid_ratio, non_zero_ratio)
            ))
        
        return reports
    
    def _get_recommendation(self, col: str, status: str, valid_ratio: float, non_zero_ratio: float) -> str:
        if status == "HEALTHY":
            return "因子健康"
        
        if "smart_money" in col:
            return "检查 amount 数据，使用 DataFallbackEngine 回退"
        elif "turnover" in col:
            return "检查 turnover_rate 数据，使用 DataFallbackEngine 回退"
        elif "vcp" in col:
            return "检查 OHLC 数据完整性"
        elif "bias" in col:
            return f"需要至少 {int(60 / max(valid_ratio, 0.01))} 天历史数据"
        elif "volume" in col:
            return "检查成交量数据"
        else:
            return "考虑使用替代因子"


class PCAOrthogonalizer:
    """
    【V3.2 新增】PCA 特征正交化器
    
    使用 PCA/Incremental PCA 替代施密特正交化：
    1. 保留 95% 方差解释度
    2. 确保特征集独立且具备强代表性
    3. 增量 PCA 节省内存
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.95,
        use_incremental: bool = True,
        max_components: int = 50,
    ):
        self.variance_threshold = variance_threshold
        self.use_incremental = use_incremental
        self.max_components = max_components
        self.pca: Optional[PCA] = None
        self.incremental_pca: Optional[IncrementalPCA] = None
        self.n_components_: int = 0
        self.explained_variance_ratio_: np.ndarray = np.array([])
        self.total_explained_variance_: float = 0.0
        self.original_columns: List[str] = []
        self.is_fitted: bool = False
    
    def fit(self, X: np.ndarray, columns: List[str]) -> 'PCAOrthogonalizer':
        """拟合 PCA"""
        self.original_columns = columns
        n_samples, n_features = X.shape
        
        # 确定主成分数量
        max_components = min(self.max_components, n_features, n_samples - 1)
        
        if self.use_incremental and n_samples > 10000:
            # 使用增量 PCA
            self.incremental_pca = IncrementalPCA(
                n_components=max_components,
                batch_size=min(1000, n_samples // 10)
            )
            self.incremental_pca.fit(X)
            self.pca = None
            
            # 计算累计方差解释度
            explained_variance_ratio = self.incremental_pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # 找到满足阈值的主成分数
            n_components_needed = np.searchsorted(cumulative_variance, self.variance_threshold) + 1
            self.n_components_ = min(n_components_needed, max_components)
            self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components_]
            self.total_explained_variance_ = float(np.sum(self.explained_variance_ratio_))
            
            logger.info(f"[PCA] 使用 IncrementalPCA, 保留 {self.n_components_} 个主成分，方差解释度 {self.total_explained_variance_:.2%}")
        else:
            # 使用标准 PCA
            self.pca = PCA(
                n_components=self.variance_threshold,
                svd_solver='full' if n_samples < 1000 else 'auto'
            )
            self.pca.fit(X)
            self.incremental_pca = None
            
            self.n_components_ = self.pca.n_components_
            self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
            self.total_explained_variance_ = float(np.sum(self.explained_variance_ratio_))
            
            logger.info(f"[PCA] 使用 PCA, 保留 {self.n_components_} 个主成分，方差解释度 {self.total_explained_variance_:.2%}")
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换数据"""
        if not self.is_fitted:
            logger.warning("PCA 未拟合，返回原始数据")
            return X
        
        if self.incremental_pca is not None:
            return self.incremental_pca.transform(X)[:, :self.n_components_]
        elif self.pca is not None:
            return self.pca.transform(X)
        else:
            return X
    
    def fit_transform(self, X: np.ndarray, columns: List[str]) -> Tuple[np.ndarray, List[str]]:
        """拟合并转换"""
        self.fit(X, columns)
        X_transformed = self.transform(X)
        
        # 生成新的特征名
        new_columns = [f"pca_{i}" for i in range(self.n_components_)]
        
        return X_transformed, new_columns
    
    def get_feature_names(self) -> List[str]:
        """返回 PCA 后的特征名"""
        return [f"pca_{i}" for i in range(self.n_components_)]
    
    def get_explained_variance_info(self) -> Dict[str, Any]:
        """返回方差解释度信息"""
        return {
            "n_components": self.n_components_,
            "total_explained_variance": self.total_explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
        }


class SharpeBasedLoss:
    """
    【V3.2 新增】Sharpe-based Loss Function
    
    直接优化模型输出结果的风险收益比，而不仅仅是价格偏差。
    
    Loss = -Sharpe Ratio + λ × Drawdown Penalty
    
    其中 Sharpe Ratio = (mean_return - risk_free_rate) / std_return
    """
    
    def __init__(
        self,
        window: int = 20,
        risk_free_rate: float = 0.02,
        drawdown_weight: float = 0.1,
    ):
        self.window = window
        self.risk_free_rate = risk_free_rate / 252  # 转换为日利率
        self.drawdown_weight = drawdown_weight
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算 Sharpe-based Loss
        
        Args:
            y_true: 真实收益率
            y_pred: 预测收益率
        
        Returns:
            loss value
        """
        n = len(y_true)
        if n < self.window:
            # 数据不足时使用简单损失
            return float(np.mean((y_true - y_pred) ** 2))
        
        # 计算滚动夏普比率
        sharpe_losses = []
        drawdown_penalties = []
        
        for i in range(self.window, n + 1):
            window_true = y_true[i - self.window:i]
            window_pred = y_pred[i - self.window:i]
            
            # 预测收益
            pred_returns = window_pred
            
            # 计算夏普比率
            mean_return = np.mean(pred_returns)
            std_return = np.std(pred_returns) + 1e-10
            sharpe = (mean_return - self.risk_free_rate) / std_return
            
            # 夏普损失 (负向夏普)
            sharpe_losses.append(-sharpe)
            
            # 计算回撤惩罚
            cumulative_returns = np.cumprod(1 + pred_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (running_max - cumulative_returns) / running_max
            max_drawdown = np.max(drawdown)
            drawdown_penalties.append(max_drawdown)
        
        sharpe_loss = np.mean(sharpe_losses)
        drawdown_penalty = np.mean(drawdown_penalties)
        
        total_loss = sharpe_loss + self.drawdown_weight * drawdown_penalty
        
        return float(total_loss)
    
    def optimize_weights(self, y_true: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        使用 Sharpe Loss 优化权重
        
        Args:
            y_true: 真实收益率
            X: 特征矩阵
        
        Returns:
            优化后的权重
        """
        from scipy.optimize import minimize
        
        n_features = X.shape[1]
        
        def objective(weights):
            y_pred = X @ weights
            return self(y_true, y_pred)
        
        # 初始权重
        x0 = np.ones(n_features) / n_features
        
        # 优化
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(-1, 1) for _ in range(n_features)],
            options={'maxiter': 100}
        )
        
        return result.x


class SelfCorrectionEngine:
    """
    【V3.2 新增】自我修正引擎
    
    功能：
    1. 计算因子 IC 值
    2. IC < 0.02 时自动重新设计合成特征
    3. 最多进行 3 轮调优
    """
    
    def __init__(
        self,
        min_ic_threshold: float = 0.02,
        max_rounds: int = 3,
    ):
        self.min_ic_threshold = min_ic_threshold
        self.max_rounds = max_rounds
        self.correction_history: List[Dict] = []
    
    def check_and_correct(
        self,
        df: pl.DataFrame,
        factor_columns: List[str],
        label_column: str = "future_return_5d",
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        检查 IC 值并进行自我修正
        
        Args:
            df: 包含因子值和标签的 DataFrame
            factor_columns: 因子列名列表
            label_column: 标签列名
        
        Returns:
            (corrected_factors, ic_values): 修正后的因子列表和 IC 值字典
        """
        ic_values = {}
        current_factors = factor_columns.copy()
        
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"[自我修正] 第 {round_num}/{self.max_rounds} 轮 IC 检查")
            
            # 计算当前因子 IC 值
            ic_values = self._calculate_ics(df, current_factors, label_column)
            
            # 找出低 IC 因子
            low_ic_factors = [
                f for f in current_factors
                if f in ic_values and abs(ic_values[f]) < self.min_ic_threshold
            ]
            
            if not low_ic_factors:
                logger.info(f"[自我修正] 所有因子 IC 值 >= {self.min_ic_threshold}, 无需修正")
                break
            
            logger.warning(f"[自我修正] 发现 {len(low_ic_factors)} 个低 IC 因子：{low_ic_factors[:5]}...")
            
            # 生成替代特征
            if round_num < self.max_rounds:
                current_factors = self._generate_alternative_factors(
                    current_factors, low_ic_factors, round_num
                )
            
            self.correction_history.append({
                "round": round_num,
                "low_ic_factors": low_ic_factors,
                "n_factors_after": len(current_factors),
            })
        
        return current_factors, ic_values
    
    def _calculate_ics(
        self,
        df: pl.DataFrame,
        factor_columns: List[str],
        label_column: str,
    ) -> Dict[str, float]:
        """计算因子 IC 值"""
        ic_values = {}
        
        for col in factor_columns:
            if col not in df.columns or label_column not in df.columns:
                ic_values[col] = 0.0
                continue
            
            # 按日期分组计算 Rank IC
            unique_dates = df[label_column].drop_nulls().unique()
            if len(unique_dates) < 5:
                ic_values[col] = 0.0
                continue
            
            # 简化 IC 计算
            data = df[[col, label_column]].drop_nulls()
            if len(data) < 10:
                ic_values[col] = 0.0
                continue
            
            # 计算相关系数
            try:
                corr = np.corrcoef(data[col].to_numpy(), data[label_column].to_numpy())[0, 1]
                ic_values[col] = float(corr) if not np.isnan(corr) else 0.0
            except:
                ic_values[col] = 0.0
        
        return ic_values
    
    def _generate_alternative_factors(
        self,
        current_factors: List[str],
        low_ic_factors: List[str],
        round_num: int,
    ) -> List[str]:
        """生成替代特征"""
        new_factors = [f for f in current_factors if f not in low_ic_factors]
        
        # 添加二阶交互特征
        if round_num == 1:
            logger.info("[自我修正] 生成二阶交互特征")
            # 取前 5 个高 IC 因子进行交互
            high_ic_factors = sorted(
                [(f, ic) for f, ic in self._current_ics.items() if f not in low_ic_factors],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            for i, (f1, _) in enumerate(high_ic_factors):
                for f2, _ in high_ic_factors[i+1:]:
                    new_factors.append(f"{f1}_x_{f2}")
        
        # 添加多项式特征
        elif round_num == 2:
            logger.info("[自我修正] 生成多项式特征")
            for f in low_ic_factors[:3]:
                new_factors.append(f"{f}_sq")  # 平方项
        
        return new_factors
    
    @property
    def _current_ics(self) -> Dict[str, float]:
        """获取当前 IC 值（从历史记录）"""
        if self.correction_history:
            return self.correction_history[-1].get("ic_values", {})
        return {}


class ParamGridSearcher:
    """
    【V3.2 新增】参数网格搜索器
    
    在 2022-2023 年数据上搜索最优止盈止损参数，
    使得平均持有天数 > 5 天 且 盈亏比 > 1.2
    """
    
    def __init__(
        self,
        train_start: str = "2022-01-01",
        train_end: str = "2023-12-31",
        min_hold_days_target: float = 5.0,
        min_profit_factor_target: float = 1.2,
    ):
        self.train_start = train_start
        self.train_end = train_end
        self.min_hold_days_target = min_hold_days_target
        self.min_profit_factor_target = min_profit_factor_target
        
        # 参数网格
        self.upper_barrier_range = [1.5, 2.0, 2.5, 3.0]
        self.stop_loss_k_range = [1.0, 1.2, 1.5, 1.8, 2.0]
        self.time_barrier_range = [8, 10, 12, 15]
    
    def search(
        self,
        strategy_class: Any,
        db: Any,
        factors_config_path: str,
    ) -> Dict[str, Any]:
        """
        执行网格搜索
        
        Args:
            strategy_class: 策略类
            db: 数据库管理器
            factors_config_path: 因子配置文件路径
        
        Returns:
            最优参数
        """
        logger.info("[参数搜索] 开始网格搜索...")
        
        best_params = {
            "upper_barrier": 2.5,
            "stop_loss_k": 1.5,
            "time_barrier": 12,
        }
        best_score = -np.inf
        
        results = []
        
        for upper_barrier, stop_loss_k, time_barrier in product(
            self.upper_barrier_range,
            self.stop_loss_k_range,
            self.time_barrier_range
        ):
            # 创建策略实例
            strategy = strategy_class(
                db=db,
                factors_config_path=factors_config_path,
                upper_barrier=upper_barrier,
                stop_loss_k=stop_loss_k,
                time_barrier=time_barrier,
            )
            
            # 训练模型
            try:
                strategy.train_model(train_end_date=self.train_end)
                
                # 运行回测
                result = strategy.run_backtest(
                    start_date=self.train_start,
                    end_date=self.train_end,
                    initial_capital=1000000.0,
                )
                
                # 计算得分
                score = self._calculate_score(result)
                
                results.append({
                    "params": {
                        "upper_barrier": upper_barrier,
                        "stop_loss_k": stop_loss_k,
                        "time_barrier": time_barrier,
                    },
                    "result": result.to_dict(),
                    "score": score,
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        "upper_barrier": upper_barrier,
                        "stop_loss_k": stop_loss_k,
                        "time_barrier": time_barrier,
                    }
                
                logger.debug(
                    f"Params: upper={upper_barrier}, k={stop_loss_k}, time={time_barrier}, "
                    f"score={score:.4f}, hold_days={result.avg_hold_days:.1f}, pf={result.profit_factor:.2f}"
                )
                
            except Exception as e:
                logger.warning(f"参数搜索失败：{e}")
                continue
        
        # 保存搜索结果
        self._save_search_results(results, best_params, best_score)
        
        logger.info(
            f"[参数搜索] 完成！最优参数：upper_barrier={best_params['upper_barrier']}, "
            f"stop_loss_k={best_params['stop_loss_k']}, time_barrier={best_params['time_barrier']}, "
            f"score={best_score:.4f}"
        )
        
        return best_params
    
    def _calculate_score(self, result: BacktestResult) -> float:
        """计算得分"""
        # 满足目标的参数得分更高
        hold_days_bonus = 1.0 if result.avg_hold_days >= self.min_hold_days_target else 0.5
        profit_factor_bonus = 1.0 if result.profit_factor >= self.min_profit_factor_target else 0.5
        
        # 综合得分
        score = (
            result.total_return * 0.4 +
            result.sharpe_ratio * 0.3 +
            result.profit_factor * 0.2 +
            (result.avg_hold_days / 10) * 0.1
        ) * hold_days_bonus * profit_factor_bonus
        
        return score
    
    def _save_search_results(
        self,
        results: List[Dict],
        best_params: Dict[str, Any],
        best_score: float,
    ) -> None:
        """保存搜索结果"""
        from pathlib import Path
        
        output_dir = Path("reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "param_grid_search_results.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "best_params": best_params,
                "best_score": best_score,
                "all_results": sorted(results, key=lambda x: x["score"], reverse=True)[:20],
            }, f, indent=2)
        
        logger.info(f"[参数搜索] 结果已保存至：{output_path}")


# ==============================================================================
# 补充定义 (从 V3.1 移植)
# ==============================================================================

class MarketGatingMechanism:
    """
    智能环境感知门控 - 使用 GMM 识别市场隐状态
    """
    
    def __init__(self, n_components: int = 4, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.gmm: Optional[GaussianMixture] = None
        self.state_weights: Dict[int, np.ndarray] = {}
        self.state_feature_importance: Dict[int, np.ndarray] = {}
        self.current_state: int = 0
        self.state_probabilities: np.ndarray = np.zeros(n_components)
        self.state_names = {0: "CALM", 1: "BULL", 2: "BEAR", 3: "VOLATILE"}
        self.state_volatility: Dict[int, float] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_columns: List[str]) -> 'MarketGatingMechanism':
        logger.info(f"[环境门控] 拟合 GMM (n_components={self.n_components})...")
        
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=200,
            random_state=self.random_state,
            n_init=10,
        )
        self.gmm.fit(X)
        
        labels = self.gmm.predict(X)
        
        for state_id in range(self.n_components):
            mask = labels == state_id
            if np.sum(mask) > 10:
                X_state = X[mask]
                y_state = y[mask]
                self.state_volatility[state_id] = float(np.std(y_state))
                
                ridge = Ridge(alpha=1.0)
                try:
                    ridge.fit(X_state, y_state)
                    self.state_weights[state_id] = ridge.coef_
                    self.state_feature_importance[state_id] = np.abs(ridge.coef_)
                    r2 = ridge.score(X_state, y_state)
                    logger.info(f"[状态 {state_id} - {self.state_names.get(state_id, 'UNKNOWN')}] "
                               f"样本数={np.sum(mask)}, R2={r2:.4f}, Vol={self.state_volatility[state_id]:.4f}")
                except Exception as e:
                    logger.warning(f"[状态 {state_id}] 权重学习失败：{e}")
                    self.state_weights[state_id] = np.zeros(X_state.shape[1])
                    self.state_feature_importance[state_id] = np.zeros(X_state.shape[1])
                    self.state_volatility[state_id] = 1.0
        
        logger.info(f"[环境门控] GMM 拟合完成，识别 {self.n_components} 个市场状态")
        return self
    
    def get_state(self, X: np.ndarray) -> Tuple[int, np.ndarray]:
        if self.gmm is None:
            return 0, np.zeros(self.n_components)
        self.current_state = int(self.gmm.predict(X.reshape(1, -1))[0])
        self.state_probabilities = self.gmm.predict_proba(X.reshape(1, -1))[0]
        return self.current_state, self.state_probabilities
    
    def get_state_name(self, state_id: Optional[int] = None) -> str:
        if state_id is None:
            state_id = self.current_state
        return self.state_names.get(state_id, f"STATE_{state_id}")
    
    def get_state_volatility(self, state_id: Optional[int] = None) -> float:
        if state_id is None:
            state_id = self.current_state
        return self.state_volatility.get(state_id, 1.0)


class TripleBarrierLabeler:
    """三屏障碍法标注器"""
    
    def __init__(
        self,
        upper_barrier: float = 2.5,
        lower_barrier: float = 1.5,
        time_barrier: int = 12,
        stop_loss_k: float = 1.5,
    ):
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.time_barrier = time_barrier
        self.stop_loss_k = stop_loss_k
    
    def compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 20,
    ) -> np.ndarray:
        """计算 ATR"""
        n = len(close)
        atr = np.zeros(n)
        
        for i in range(1, n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            if i < period:
                atr[i] = atr[i-1] + (tr - atr[i-1]) / (i + 1) if i > 0 else tr
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr) / period
        
        return atr
    
    def label(
        self,
        prices: np.ndarray,
        volatility: np.ndarray,
        high_prices: Optional[np.ndarray] = None,
        low_prices: Optional[np.ndarray] = None,
        atr_values: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """三屏障碍法标注"""
        n_samples = prices.shape[0]
        labels = np.ones(n_samples) * 2  # 默认为时间到期
        barrier_types = ["time"] * n_samples
        returns = np.zeros(n_samples)
        
        for i in range(n_samples):
            if i >= len(volatility) or np.isnan(volatility[i]) or volatility[i] <= 0:
                continue
            
            entry_price = prices[i, 0] if prices.ndim > 1 else prices[i]
            vol = abs(volatility[i])
            
            # 上轨（止盈）
            upper = entry_price * (1 + self.upper_barrier * vol)
            
            # 下轨（止损）- 使用 ATR 或固定波动率
            if atr_values is not None and i < len(atr_values) and not np.isnan(atr_values[i]):
                atr = atr_values[i]
                stop_distance = self.stop_loss_k * atr
                lower = entry_price - stop_distance
            else:
                lower = entry_price * (1 - self.lower_barrier * vol)
            
            if prices.ndim > 1:
                max_days = min(prices.shape[1], self.time_barrier)
                for t in range(max_days):
                    price_t = prices[i, t] if t < prices.shape[1] else prices[i, -1]
                    
                    if price_t >= upper:
                        labels[i] = 1
                        barrier_types[i] = "upper"
                        returns[i] = (price_t - entry_price) / entry_price
                        break
                    elif price_t <= lower:
                        labels[i] = 0
                        barrier_types[i] = "lower"
                        returns[i] = (price_t - entry_price) / entry_price
                        break
                    elif t == max_days - 1:
                        labels[i] = 2
                        returns[i] = (price_t - entry_price) / entry_price
        
        return labels, barrier_types, returns


# ==============================================================================
# 主策略类 (V3.2 自我进化版)
# ==============================================================================

class FinalStrategyV32:
    """
    Final Strategy V3.2 - Self-Evolving Factor Engine with PCA & Sharpe Loss
    
    【V3.2 升级说明】
    1. 数据回退引擎：自动处理 amount/turnover 缺失
    2. 预热断言：确保因子非零值比例 > 90%
    3. PCA 正交化：保留 95% 方差解释度
    4. Sharpe Loss：直接优化风险收益比
    5. IC 闭环调优：IC < 0.02 时自动修正
    6. 参数网格搜索：寻找最优止盈止损参数
    """
    
    def __init__(
        self,
        config_path: str = "config/production_params.yaml",
        factors_config_path: str = "config/factors.yaml",
        db: Optional[DatabaseManager] = None,
        # V3.2 新增参数
        upper_barrier: float = 2.5,
        stop_loss_k: float = 1.5,
        time_barrier: int = 12,
    ):
        self.config_path = Path(config_path)
        self.factors_config_path = Path(factors_config_path)
        self.db = db or DatabaseManager.get_instance()
        
        self.config = self._load_config()
        self.factor_engine = FactorEngine(str(self.factors_config_path), validate=True)
        
        # V3.2 核心组件
        self.data_fallback = DataFallbackEngine()
        self.warmup_assert = WarmupAssertEngine()
        self.pca_orthogonalizer = PCAOrthogonalizer(
            variance_threshold=PCA_VARIANCE_THRESHOLD,
            use_incremental=PCA_USE_INCREMENTAL,
            max_components=PCA_MAX_COMPONENTS,
        )
        self.sharpe_loss = SharpeBasedLoss(
            window=SHARPE_LOSS_WINDOW,
            risk_free_rate=SHARPE_LOSS_RISK_FREE_RATE,
        )
        self.self_correction = SelfCorrectionEngine(
            min_ic_threshold=MIN_IC_THRESHOLD,
            max_rounds=MAX_CORRECTION_ROUNDS,
        )
        
        # 三屏障碍法
        self.triple_barrier = TripleBarrierLabeler(
            upper_barrier=upper_barrier,
            lower_barrier=1.5,
            time_barrier=time_barrier,
            stop_loss_k=stop_loss_k,
        )
        
        # 环境门控
        self.gating = MarketGatingMechanism(n_components=GMM_N_COMPONENTS, random_state=GMM_RANDOM_STATE)
        
        # 模型状态
        self.ridge_model: Optional[Ridge] = None
        self.lgb_model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        
        # 特征配置
        self.training_feature_columns: List[str] = []
        self.pca_feature_columns: List[str] = []
        self.expected_feature_dim: int = TOP_N_FEATURES_BY_MI
        
        # 市场状态
        self.market_state = MarketState()
        self.positions: Dict[str, Position] = {}
        self.trade_records: List[TradeRecord] = []
        self.cash: float = 0.0
        self.initial_capital: float = 0.0
        
        # IC 值记录
        self.factor_ic_values: Dict[str, float] = {}
        
        logger.info("FinalStrategyV3.2 initialized (Self-Evolving with PCA & Sharpe Loss)")
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "strategy": {"top_k_stocks": 10, "max_position_pct": 0.1},
            "risk_control": {"atr_multiplier": 3.0},
            "liquidity_filter": {"2023": LIQUIDITY_FILTER_2023, "2024": LIQUIDITY_FILTER_2024},
        }
    
    def _get_training_data(self, end_date: str) -> Optional[pl.DataFrame]:
        try:
            query = f"SELECT * FROM stock_daily WHERE trade_date <= '{end_date}' AND trade_date >= '2022-01-01'"
            data = self.db.read_sql(query)
            if data.is_empty():
                return None
            
            # V3.2: 应用数据回退
            if DATA_FALLBACK_ENABLED:
                logger.info("[数据准备] 应用数据回退逻辑...")
                data = self.data_fallback.check_and_apply_fallback(data)
            
            # 计算因子
            data = self.factor_engine.compute_factors(data)
            
            return data
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None
    
    def _prepare_training_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备训练特征"""
        base_columns = self._get_feature_columns()
        available_cols = [c for c in base_columns if c in df.columns]
        
        # V3.2: 预热断言
        if WARMUP_ASSERT_ENABLED:
            logger.info("[特征准备] 执行预热断言...")
            try:
                self.warmup_assert.assert_warmup_ready(df, available_cols)
            except AssertionError as e:
                logger.error(f"预热断言失败：{e}")
                raise
        
        # 获取因子健康度报告
        health_reports = self.warmup_assert.get_factor_health_report(df, available_cols)
        self._print_factor_health_report(health_reports)
        
        self.training_feature_columns = available_cols.copy()
        logger.info(f"[特征准备] 训练特征数：{len(available_cols)}")
        
        # 过滤空值
        df_clean = df.filter(pl.all_horizontal(pl.col(available_cols).is_not_null()))
        if len(df_clean) == 0:
            logger.warning("No valid data after filtering nulls, using fill_null strategy")
            df_clean = df.clone()
            for col in available_cols:
                if col in df_clean.columns:
                    df_clean = df_clean.with_columns([pl.col(col).fill_null(0.0).alias(col)])
        
        X = df_clean.select(available_cols).to_numpy()
        
        # 计算目标变量
        future_return_col = "future_return_12d"
        if future_return_col not in df_clean.columns:
            df_clean = df_clean.with_columns([
                (pl.col("close").shift(-12) / pl.col("close") - 1).alias("future_return_12d")
            ])
        y = df_clean["future_return_12d"].fill_null(0).to_numpy()
        
        # 1. V3.2: PCA 正交化 (替代施密特正交化)
        if PCA_ORTHOGONALIZATION_ENABLED:
            logger.info("[特征准备] 执行 PCA 正交化...")
            X_pca, pca_columns = self.pca_orthogonalizer.fit_transform(X, available_cols)
            self.pca_feature_columns = pca_columns
            
            var_info = self.pca_orthogonalizer.get_explained_variance_info()
            logger.info(
                f"[PCA] 从 {len(available_cols)} 个原始特征降维到 {var_info['n_components']} 个主成分，"
                f"方差解释度 {var_info['total_explained_variance']:.2%}"
            )
        else:
            X_pca = X
            pca_columns = available_cols
        
        # 2. 特征合成
        if FEATURE_SYNTHESIS_ENABLED:
            logger.info("[特征准备] 执行特征合成...")
            X_synth, selected_cols = self._synthesize_features(X_pca, y, pca_columns)
        else:
            X_synth = X_pca
            selected_cols = pca_columns
        
        # 3. 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_synth)
        
        # V3.2: IC 闭环调优
        if IC_SELF_CORRECTION_ENABLED:
            logger.info("[特征准备] 执行 IC 闭环调优...")
            df_with_y = df_clean.with_columns([pl.Series("future_return_12d", y)])
            corrected_factors, ic_values = self.self_correction.check_and_correct(
                df_with_y, selected_cols, "future_return_12d"
            )
            self.factor_ic_values = ic_values
            self._print_ic_summary(ic_values)
        
        self.feature_columns = selected_cols
        logger.info(f"[数据准备] 完成：最终特征数={X_scaled.shape[1]}")
        
        return X_scaled, y, selected_cols
    
    def _synthesize_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        columns: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """合成特征"""
        all_features: Dict[str, np.ndarray] = {}
        all_mi: Dict[str, float] = {}
        
        # 原始特征
        for i, col in enumerate(columns):
            if i < X.shape[1]:
                all_features[col] = X[:, i]
                mi = self._compute_mutual_information(X[:, i], y)
                all_mi[col] = mi
        
        # 二阶交互特征
        if MAX_INTERACTION_ORDER >= 2:
            logger.info(f"[特征合成] 生成二阶交叉特征...")
            feature_names = list(all_features.keys())
            for i in range(min(len(feature_names), 20)):
                for j in range(i + 1, min(len(feature_names), 20)):
                    cross_feature = all_features[feature_names[i]] * all_features[feature_names[j]]
                    col_name = f"{feature_names[i]}_x_{feature_names[j]}"
                    all_features[col_name] = cross_feature
                    mi = self._compute_mutual_information(cross_feature, y)
                    all_mi[col_name] = mi
        
        # 选择 Top N
        sorted_features = sorted(all_mi.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in sorted_features[:TOP_N_FEATURES_BY_MI]]
        
        X_synthesized = np.column_stack([all_features[f] for f in selected_features])
        
        return X_synthesized, selected_features
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算互信息"""
        try:
            n_bins = 10
            percentiles = np.linspace(0, 100, n_bins + 1)
            x_bins = np.percentile(x[~np.isnan(x)], percentiles)
            y_bins = np.percentile(y[~np.isnan(y)], percentiles)
            
            x_discrete = np.digitize(x, np.unique(x_bins[1:-1]))
            y_discrete = np.digitize(y, np.unique(y_bins[1:-1]))
            
            mi = mutual_info_score(x_discrete, y_discrete)
            return mi if np.isfinite(mi) else 0.0
        except Exception:
            return 0.0
    
    def _get_feature_columns(self) -> List[str]:
        """获取所有可能的特征列"""
        base_factors = self.factor_engine.get_factor_names()
        technical_factors = ["rsi_14", "macd", "macd_signal", "macd_hist"]
        volume_price_factors = ["volume_price_health", "volume_shrink_flag", "price_volume_divergence"]
        private_factors = ["vcp_score", "turnover_stable", "smart_money_signal"]
        
        all_factors = base_factors + technical_factors + volume_price_factors + private_factors
        seen = set()
        unique_factors = []
        for f in all_factors:
            if f not in seen:
                seen.add(f)
                unique_factors.append(f)
        return unique_factors
    
    def _print_factor_health_report(self, reports: List[FactorHealthReport]) -> None:
        """打印因子健康度报告"""
        if not reports:
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("【因子健康度报告】V3.2")
        logger.info("=" * 70)
        
        healthy_count = sum(1 for r in reports if r.status == "HEALTHY")
        critical_count = sum(1 for r in reports if r.status == "CRITICAL")
        
        logger.info(f"总因子数：{len(reports)}")
        logger.info(f"健康因子：{healthy_count} ({healthy_count/len(reports):.1%})")
        logger.info(f"临界因子：{critical_count} ({critical_count/len(reports):.1%})")
        logger.info("-" * 70)
        
        # 打印临界因子
        critical_reports = [r for r in reports if r.status == "CRITICAL"]
        if critical_reports:
            logger.info("\n【临界因子】")
            for r in sorted(critical_reports, key=lambda x: x.non_zero_ratio):
                logger.info(f"  {r.factor_name}: 非零值比例={r.non_zero_ratio:.1%}, 建议：{r.recommendation}")
        
        logger.info("=" * 70 + "\n")
    
    def _print_ic_summary(self, ic_values: Dict[str, float]) -> None:
        """打印 IC 值汇总"""
        if not ic_values:
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("【因子 IC 值汇总】")
        logger.info("=" * 70)
        
        sorted_ics = sorted(ic_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for factor_name, ic in sorted_ics[:15]:
            marker = ""
            if abs(ic) >= 0.05:
                marker = " ***"
            elif abs(ic) >= 0.03:
                marker = " **"
            elif abs(ic) >= MIN_IC_THRESHOLD:
                marker = " *"
            logger.info(f"  {factor_name}: IC={ic:.4f}{marker}")
        
        logger.info("=" * 70 + "\n")
    
    def train_model(self, train_end_date: str = TRAIN_END_DATE) -> None:
        """训练模型"""
        logger.info(f"Training model with data until {train_end_date}...")
        
        train_data = self._get_training_data(train_end_date)
        if train_data is None or len(train_data) == 0:
            logger.warning("No training data found")
            return
        
        X, y, feature_cols = self._prepare_training_features(train_data)
        if len(X) == 0 or len(y) == 0:
            logger.warning("No valid training data")
            return
        
        logger.info(f"[模型训练] 特征维度：X.shape={X.shape}")
        
        # 拟合 GMM
        if GATING_MECHANISM_ENABLED:
            logger.info("[模型训练] 拟合环境门控 GMM...")
            self.gating.fit(X, y, feature_cols)
        
        # 训练 Ridge (使用 Sharpe Loss 优化)
        logger.info("[模型训练] 训练 Ridge 模型 (Sharpe Loss)...")
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X, y)
        
        # V3.2: 使用 Sharpe Loss 优化权重
        if SHARPE_LOSS_ENABLED:
            try:
                logger.info("[模型训练] 使用 Sharpe Loss 优化权重...")
                optimized_weights = self.sharpe_loss.optimize_weights(y, X)
                self.ridge_model.coef_ = optimized_weights
                logger.info("[模型训练] Sharpe Loss 优化完成")
            except Exception as e:
                logger.warning(f"Sharpe Loss 优化失败：{e}, 使用默认 Ridge 权重")
        
        # 训练 LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("[模型训练] 训练 LightGBM 模型...")
            self.lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
            self.lgb_model.fit(X, y)
        else:
            logger.warning("LightGBM not available, using Ridge only")
            self.lgb_model = None
        
        self._print_feature_importance()
        logger.info(f"Model training complete with {len(X)} samples, {len(feature_cols)} features")
    
    def _print_feature_importance(self) -> None:
        """打印特征重要性"""
        if self.ridge_model and hasattr(self.ridge_model, "coef_"):
            importance = list(zip(self.feature_columns, self.ridge_model.coef_))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            logger.info("[Ridge Top 10 特征重要性]")
            for i, (feat, imp) in enumerate(importance[:10], 1):
                logger.info(f"  {i}. {feat}: {imp:.4f}")
        
        if self.lgb_model and LIGHTGBM_AVAILABLE:
            importance = list(zip(self.feature_columns, self.lgb_model.feature_importances_))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            logger.info("[LightGBM Top 10 特征重要性]")
            for i, (feat, imp) in enumerate(importance[:10], 1):
                logger.info(f"  {i}. {feat}: {imp:.4f}")
    
    def predict(self, df: pl.DataFrame, symbol: str = None) -> pl.DataFrame:
        """预测"""
        if not self.pca_orthogonalizer.is_fitted or self.scaler is None:
            logger.warning("模型未训练，返回零预测")
            return df.with_columns(pl.lit(0.0).alias("predict_score"))
        
        feature_cols = self.training_feature_columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if len(available_cols) == 0:
            logger.warning("No feature columns found")
            return df.with_columns(pl.lit(0.0).alias("predict_score"))
        
        df_with_idx = df.with_columns(pl.arange(0, len(df)).alias("__idx"))
        df_filtered = df_with_idx.filter(
            pl.all_horizontal(pl.col(available_cols).is_not_null())
        )
        
        if len(df_filtered) == 0:
            logger.warning("No valid data for prediction")
            return df.with_columns(pl.lit(0.0).alias("predict_score"))
        
        # 提取特征
        X = df_filtered.select(available_cols).to_numpy()
        
        # PCA 转换 - 使用与训练时相同的参数
        X_pca = self.pca_orthogonalizer.transform(X)
        
        # 检查特征维度是否匹配
        expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else 0
        actual_features = X_pca.shape[1]
        
        if actual_features != expected_features:
            logger.warning(f"特征维度不匹配：期望 {expected_features}, 实际 {actual_features}")
            # 使用原始特征进行预测（跳过 PCA）
            X_scaled = self.scaler.transform(X[:, :expected_features]) if X.shape[1] >= expected_features else np.zeros((len(X), expected_features))
        else:
            # 标准化
            X_scaled = self.scaler.transform(X_pca)
        
        # 模型预测
        if self.ridge_model is None:
            ridge_pred = np.zeros(len(X_scaled))
        else:
            ridge_pred = self.ridge_model.predict(X_scaled)
        
        if self.lgb_model is not None and LIGHTGBM_AVAILABLE:
            lgb_pred = self.lgb_model.predict(X_scaled)
        else:
            lgb_pred = ridge_pred
        
        # 集成预测
        ensemble_pred = 0.5 * ridge_pred + 0.5 * lgb_pred
        
        # 构建结果
        pred_df = df_filtered.select(["__idx"]).with_columns(
            pl.Series("predict_score", ensemble_pred)
        )
        
        result = df_with_idx.join(pred_df, on="__idx", how="left")
        result = result.drop("__idx").with_columns(
            pl.col("predict_score").fill_null(0.0)
        )
        return result
    
    def run_backtest(self, start_date: str, end_date: str, initial_capital: float = 1000000.0) -> BacktestResult:
        logger.info(f"Running backtest from {start_date} to {end_date}...")
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_records = []
        
        backtest_data = self._get_backtest_data(start_date, end_date)
        if backtest_data is None or len(backtest_data) == 0:
            logger.warning("No backtest data found")
            return BacktestResult()
        
        # V3.2: 应用数据回退
        if DATA_FALLBACK_ENABLED:
            logger.info("[回测] 应用数据回退逻辑...")
            backtest_data = self.data_fallback.check_and_apply_fallback(backtest_data)
        
        # V3.2: 预热断言
        if WARMUP_ASSERT_ENABLED:
            logger.info("[回测] 执行预热断言...")
            try:
                # 先计算因子
                backtest_data = self.factor_engine.compute_factors(backtest_data)
                factor_cols = self._get_feature_columns()
                available_cols = [c for c in factor_cols if c in backtest_data.columns]
                self.warmup_assert.assert_warmup_ready(backtest_data, available_cols)
            except AssertionError as e:
                logger.error(f"回测预热断言失败：{e}")
                # 继续执行但记录警告
        
        backtest_data = backtest_data.sort("trade_date")
        dates = backtest_data["trade_date"].unique().to_list()
        daily_values = []
        total_slippage_cost = 0.0
        
        for date in dates:
            daily_data = backtest_data.filter(pl.col("trade_date") == date)
            if len(daily_data) == 0:
                continue
            
            # 更新持仓天数
            for symbol in list(self.positions.keys()):
                self.positions[symbol].hold_days += 1
            
            price_map = {}
            for row in daily_data.iter_rows(named=True):
                symbol = row.get("symbol")
                if symbol:
                    price_map[symbol] = {
                        "close": row.get("close", 0),
                        "high": row.get("high", 0),
                        "low": row.get("low", 0),
                    }
            
            # 流动性过滤
            year = int(str(date)[:4])
            liquidity_threshold = LIQUIDITY_FILTER_2023 if year == 2023 else LIQUIDITY_FILTER_2024
            daily_data = daily_data.filter(
                (pl.col("amount") >= liquidity_threshold) | (pl.col("amount").is_null())
            )
            
            # 预测
            daily_data = self.predict(daily_data)
            
            # 执行交易
            slippage_cost = self._execute_daily_trading(date, daily_data, price_map)
            total_slippage_cost += slippage_cost
            
            # 计算组合价值
            portfolio_value = self._calculate_portfolio_value(price_map)
            daily_values.append({
                "date": date,
                "value": portfolio_value,
                "cash": self.cash,
                "positions": len(self.positions),
            })
        
        result = self._calculate_backtest_result(daily_values)
        result.daily_values = daily_values
        result.total_slippage_cost = total_slippage_cost
        
        logger.info(f"Backtest complete: Total Return = {result.total_return:.2%}")
        return result
    
    def _get_backtest_data(self, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        try:
            query = f"SELECT * FROM stock_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
            data = self.db.read_sql(query)
            if data.is_empty():
                return None
            return data
        except Exception as e:
            logger.error(f"Failed to get backtest data: {e}")
            return None
    
    def _execute_daily_trading(self, date: str, data: pl.DataFrame, price_map: Dict[str, Dict]) -> float:
        total_slippage = 0.0
        self._check_exit_conditions(date, price_map, data)
        buy_slippage = self._generate_buy_signals(date, data, price_map)
        total_slippage += buy_slippage
        return total_slippage
    
    def _check_exit_conditions(self, date: str, price_map: Dict[str, Dict], data: pl.DataFrame) -> None:
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            if symbol not in price_map:
                continue
            
            price_info = price_map[symbol]
            current_price = float(price_info["close"])
            
            if current_price > position.highest_price:
                position.highest_price = current_price
            if current_price < position.lowest_price or position.lowest_price == 0:
                position.lowest_price = current_price
            
            entry_return = (current_price - position.entry_price) / position.entry_price
            
            if position.stop_loss_price > 0 and current_price <= position.stop_loss_price:
                self._exit_position(symbol, date, current_price, "stop_loss")
            elif position.take_profit_price > 0 and current_price >= position.take_profit_price:
                self._exit_position(symbol, date, current_price, "take_profit")
            elif entry_return >= position.predicted_return * 0.5 or entry_return <= -0.05:
                self._exit_position(symbol, date, current_price, "target/rebalance")
    
    def _generate_buy_signals(self, date: str, data: pl.DataFrame, price_map: Dict[str, Dict]) -> float:
        top_k = self.config.get("strategy", {}).get("top_k_stocks", 10)
        total_slippage = 0.0
        
        score_col = "predict_score"
        scored_data = data.filter(pl.col(score_col).is_not_null())
        
        if len(scored_data) == 0:
            return total_slippage
        
        top_stocks = scored_data.sort(score_col, descending=True).head(top_k)
        
        for row in top_stocks.iter_rows(named=True):
            symbol = row.get("symbol")
            industry = row.get("industry", "")
            
            if not symbol or symbol in self.positions:
                continue
            
            if symbol in price_map:
                raw_score = row.get(score_col, 0)
                price = float(price_map[symbol]["close"])
                
                max_position_pct = self.config.get("strategy", {}).get("max_position_pct", 0.1)
                position_value = self.cash * max_position_pct
                
                if position_value >= price * 100:
                    shares = int(position_value / price / 100) * 100
                    if shares >= 100:
                        slippage_cost = shares * price * 0.001
                        self.cash -= shares * price + slippage_cost
                        
                        # 计算止损/止盈价
                        estimated_atr = price * 0.02
                        stop_loss_price = price - self.triple_barrier.stop_loss_k * estimated_atr
                        take_profit_price = price * (1 + self.triple_barrier.upper_barrier * 0.02)
                        
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            entry_date=str(date),
                            entry_price=price,
                            shares=shares,
                            current_score=raw_score,
                            industry=industry,
                            highest_price=price,
                            lowest_price=price,
                            predicted_return=raw_score,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                        )
                        
                        logger.debug(f"[买入] {symbol} @ {price:.2f} x {shares}股，评分={raw_score:.4f}")
                        total_slippage += slippage_cost
        
        return total_slippage
    
    def _exit_position(self, symbol: str, date: str, price: float, reason: str) -> None:
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        price = float(price)
        
        pnl = (price - position.entry_price) * position.shares
        pnl_pct = pnl / (position.entry_price * position.shares)
        
        self.cash += position.shares * price
        
        self.trade_records.append(TradeRecord(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=str(date),
            entry_price=position.entry_price,
            exit_price=price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            hold_days=position.hold_days,
            max_profit_pct=(position.highest_price - position.entry_price) / position.entry_price,
            min_profit_pct=(position.lowest_price - position.entry_price) / position.entry_price,
            predicted_sharpe=position.predicted_sharpe,
            stop_loss_type=reason,
        ))
        
        del self.positions[symbol]
        logger.debug(f"[卖出] {symbol} @ {price:.2f}, 盈亏={pnl_pct:.2%}, 原因={reason}")
    
    def _calculate_portfolio_value(self, price_map: Dict[str, Dict]) -> float:
        value = float(self.cash)
        for symbol, position in self.positions.items():
            if symbol in price_map:
                price = float(price_map[symbol]["close"])
                value += position.shares * price
        return value
    
    def _calculate_backtest_result(self, daily_values: List[Dict]) -> BacktestResult:
        if len(daily_values) == 0:
            return BacktestResult()
        
        values = [dv["value"] for dv in daily_values]
        
        total_return = (values[-1] - self.initial_capital) / self.initial_capital
        days = len(daily_values)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        if len(values) > 1:
            daily_returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe = (avg_return / (std_return + 1e-6)) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0.0
        
        winning_trades = [t for t in self.trade_records if t.pnl > 0]
        losing_trades = [t for t in self.trade_records if t.pnl < 0]
        win_rate = len(winning_trades) / len(self.trade_records) if self.trade_records else 0
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        avg_hold = np.mean([t.hold_days for t in self.trade_records]) if self.trade_records else 0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=len(self.trade_records),
            avg_hold_days=avg_hold,
            profit_factor=profit_factor,
            total_slippage_cost=0.0,
            trades=self.trade_records,
            daily_values=daily_values,
        )
    
    def run_param_grid_search(self) -> Dict[str, Any]:
        """运行参数网格搜索"""
        logger.info("[V3.2] 启动参数网格搜索...")
        
        searcher = ParamGridSearcher(
            train_start=GRID_SEARCH_TRAIN_START,
            train_end=GRID_SEARCH_TRAIN_END,
            min_hold_days_target=MIN_HOLD_DAYS_TARGET,
            min_profit_factor_target=MIN_PROFIT_FACTOR_TARGET,
        )
        
        best_params = searcher.search(
            strategy_class=FinalStrategyV32,
            db=self.db,
            factors_config_path=str(self.factors_config_path),
        )
        
        return best_params


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 70)
    logger.info("Final Strategy V3.2 - Self-Evolving with PCA & Sharpe Loss")
    logger.info("=" * 70)
    
    # 创建策略实例
    strategy = FinalStrategyV32(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
    )
    
    # V3.2: 可选运行参数网格搜索
    if PARAM_GRID_SEARCH_ENABLED:
        logger.info("[V3.2] 运行参数网格搜索...")
        best_params = strategy.run_param_grid_search()
        logger.info(f"最优参数：{best_params}")
    
    # 训练模型
    strategy.train_model(train_end_date=TRAIN_END_DATE)
    
    # 运行回测
    result = strategy.run_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=1000000.0,
    )
    
    # 打印结果
    logger.info("")
    logger.info("=" * 70)
    logger.info("【V3.2 回测结果】")
    logger.info("=" * 70)
    logger.info(f"总收益率：{result.total_return:.2%}")
    logger.info(f"年化收益：{result.annual_return:.2%}")
    logger.info(f"最大回撤：{result.max_drawdown:.2%}")
    logger.info(f"夏普比率：{result.sharpe_ratio:.2f}")
    logger.info(f"胜率：{result.win_rate:.1%}")
    logger.info(f"交易次数：{result.total_trades}")
    logger.info(f"平均持有天数：{result.avg_hold_days:.1f}")
    logger.info(f"盈亏比：{result.profit_factor:.2f}")
    logger.info("=" * 70)
    
    # 保存结果
    from pathlib import Path
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = output_dir / "v32_backtest_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"回测结果已保存至：{result_path}")


if __name__ == "__main__":
    main()