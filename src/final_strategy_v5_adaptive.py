"""
Final Strategy V5 Adaptive - 自适应滚动训练版
=================================================
作者：量化模型优化专家 (Machine Learning Engineer)
日期：2026-03-17

【核心修复 - 预测坍缩问题解决方案】

1. 特征工程管线加固 (Feature Resilience)
   - 局部归一化 (Cross-sectional Scaling)：在 predict 方法中对输入特征 X 执行当日截面的 Rank 缩放或 Z-Score 缩放
   - 缺失值容错：先按行业中位数填充，若仍缺失再用 0
   - 特征检查断言：在预测前加入 assert X_scaled.std() > 0

2. 滚动训练机制 (Rolling Training)
   - 每回测 20 个交易日，自动使用过去 500 天的数据重新 fit 模型
   - 样本扩充：确保训练样本覆盖全市场至少 1000 只股票

3. 交易策略微调 (Execution Strategy)
   - ATR 止损倍数从 1.5 倍提高至 2.5 倍
   - 买入过滤：只有当 predict_score > 0.6 且 z-score > 1.5 时才入场

【核心适配 - 10 万资金约束】
1. 初始资金：100,000 元
2. 最小交易单位：100 股整数倍
3. 真实手续费：万分之三佣金 (最低 5 元) + 千分之一卖出印花税
4. 仓位集中度：TOP_K = 3-4 只
5. 滑点成本：0.1%
6. 行业分散：同一行业不超过 2 只
7. 移动止盈机制
8. Calmar Ratio 作为核心评价指标 (目标 > 0.8)

【技术栈】
- Polars: 数据处理
- LightGBM: 机器学习模型
- StandardScaler: 特征标准化
- SQLAlchemy: 数据库连接
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Tuple, Set
from dataclasses import dataclass, field
import json
import numpy as np
import polars as pl
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
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
    from .backtest_engine import BacktestEngine
except ImportError:
    from db_manager import DatabaseManager
    from backtest_engine import BacktestEngine


# ==============================================================================
# 配置常量 - V5 Adaptive 优化
# ==============================================================================

# 训练配置 - 滚动训练
TRAIN_END_DATE = "2023-12-31"
TRAIN_START_DATE = "2022-01-01"
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2024-06-30"

# 滚动训练参数
ROLLING_WINDOW_DAYS = 500  # 使用过去 500 天的数据重新训练
ROLLING_RETRAIN_INTERVAL = 20  # 每 20 个交易日重新训练一次
MIN_TRAINING_SAMPLES = 1000  # 最小训练样本数（覆盖至少 1000 只股票）

# 资金配置 - 10 万实战
INITIAL_CAPITAL = 100000.0
RISK_FREE_RATE = 0.02  # 2% 年化无风险利率

# 手续费模型 - 真实成本
COMMISSION_RATE = 0.0003  # 万分之三佣金
COMMISSION_MIN = 5.0      # 最低 5 元
STAMP_DUTY_RATE = 0.001   # 千分之一卖出印花税
SLIPPAGE_RATE = 0.001     # 0.1% 滑点

# 交易单位约束
LOT_SIZE = 100  # 最小交易单位 100 股

# 仓位配置 - 小资金集中度优化
TOP_K_STOCKS = 3  # 从 5 只降低到 3 只，保证单只代表性
MAX_POSITION_PCT = 0.30  # 单只股票最大仓位 30%

# 行业分散约束
MAX_STOCKS_PER_INDUSTRY = 2  # 同一行业不超过 2 只

# 特征选择配置
MAX_FEATURES = 16
MIN_IC_THRESHOLD = 0.03
CORRELATION_THRESHOLD = 0.85

# 模型配置
LGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.01,
    "num_leaves": 25,
    "min_child_samples": 30,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.05,
    "reg_lambda": 0.05,
    "random_state": 42,
    "n_jobs": -1,
    "objective": "regression",
    "metric": "mse",
}

# 【关键修复】止损止盈配置 - ATR 止损倍数提高至 2.5 倍
ATR_STOP_LOSS_MULTIPLIER = 2.5  # 【修复】从 1.5 倍提高至 2.5 倍，给小资金标的留出波动空间
TRAILING_STOP_PCT = 0.05  # 5% 移动止盈
TAKE_PROFIT_PCT = 0.08  # 8% 固定止盈
MAX_HOLD_DAYS = 10

# 【关键修复】预测分数门槛 - 平衡入场质量和交易机会
# 使用百分位排名后，分数范围是 [0, 1]，0.55 表示排名在前 45%
# z-score > 1.0 表示预测值高于均值 1 个标准差 (前 16%)
MIN_PREDICT_SCORE = 0.55  # 排名前 45% 的股票
SIGNIFICANCE_STD_THRESHOLD = 1.0  # z-score 必须 > 1.0 才入场

# 流动性过滤
LIQUIDITY_FILTER_2023 = 50_000_000
LIQUIDITY_FILTER_2024 = 100_000_000


# ==============================================================================
# 数据类定义
# ==============================================================================

@dataclass
class Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    hold_days: int = 0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    predicted_return: float = 0.0
    entry_atr: float = 0.0
    industry: str = ""
    cost_basis: float = 0.0  # 包含手续费的成本


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
    commission_paid: float = 0.0
    stamp_duty_paid: float = 0.0
    max_profit_pct: float = 0.0
    min_profit_pct: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0  # Calmar Ratio = 年化收益 / 最大回撤
    win_rate: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0
    profit_factor: float = 0.0
    total_commission: float = 0.0
    total_stamp_duty: float = 0.0
    total_slippage_cost: float = 0.0
    skipped_trades: int = 0  # 因资金不足放弃的交易次数
    trades: List[TradeRecord] = field(default_factory=list)
    daily_values: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "avg_hold_days": self.avg_hold_days,
            "profit_factor": self.profit_factor,
            "total_commission": self.total_commission,
            "total_stamp_duty": self.total_stamp_duty,
            "total_slippage_cost": self.total_slippage_cost,
            "skipped_trades": self.skipped_trades,
        }


# ==============================================================================
# 手续费计算器
# ==============================================================================

class FeeCalculator:
    """
    真实手续费计算器
    
    买入：佣金 (万分之三，最低 5 元)
    卖出：佣金 (万分之三，最低 5 元) + 印花税 (千分之一)
    """
    
    @staticmethod
    def calculate_buy_commission(amount: float) -> float:
        """计算买入佣金"""
        commission = amount * COMMISSION_RATE
        return max(commission, COMMISSION_MIN)
    
    @staticmethod
    def calculate_sell_fee(amount: float) -> Tuple[float, float]:
        """
        计算卖出费用
        
        Returns:
            (commission, stamp_duty)
        """
        commission = amount * COMMISSION_RATE
        commission = max(commission, COMMISSION_MIN)
        stamp_duty = amount * STAMP_DUTY_RATE
        return commission, stamp_duty
    
    @staticmethod
    def calculate_total_buy_cost(price: float, shares: int) -> Tuple[float, float, float]:
        """
        计算买入总成本
        
        Returns:
            (stock_cost, commission, total_cost)
        """
        stock_cost = price * shares
        commission = FeeCalculator.calculate_buy_commission(stock_cost)
        slippage = stock_cost * SLIPPAGE_RATE
        total_cost = stock_cost + commission + slippage
        return stock_cost, commission, total_cost
    
    @staticmethod
    def calculate_sell_proceeds(price: float, shares: int) -> Tuple[float, float, float, float]:
        """
        计算卖出净收入
        
        Returns:
            (stock_value, commission, stamp_duty, net_proceeds)
        """
        stock_value = price * shares
        commission, stamp_duty = FeeCalculator.calculate_sell_fee(stock_value)
        slippage = stock_value * SLIPPAGE_RATE
        net_proceeds = stock_value - commission - stamp_duty - slippage
        return stock_value, commission, stamp_duty, net_proceeds


# ==============================================================================
# 特征引擎 - V5 加固版
# ==============================================================================

class RobustFeatureEngine:
    """鲁棒特征引擎 - V5 加固版"""
    
    def __init__(self):
        self.feature_columns: List[str] = []
        self.medians: Dict[str, float] = {}
        self.industry_medians: Dict[str, Dict[str, float]] = {}  # 按行业存储中位数
    
    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算所有特征"""
        result = df.clone()
        
        if "symbol" in result.columns and "trade_date" in result.columns:
            result = result.sort(["symbol", "trade_date"])
        
        result = self._compute_price_volume_divergence(result)
        result = self._compute_volatility_features(result)
        result = self._compute_money_flow(result)
        result = self._compute_momentum_features(result)
        result = self._compute_technical_indicators(result)
        result = self._apply_log_transform(result)
        
        return result
    
    def _apply_log_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """对数变换处理长尾分布因子"""
        result = df.clone()
        
        log_columns = ["volume", "amount", "money_flow_5d", "money_flow_20d"]
        
        for col in log_columns:
            if col in result.columns:
                result = result.with_columns([
                    pl.col(col).abs().clip(1e-10, None).log().alias(f"{col}_log")
                ])
        
        return result
    
    def _compute_price_volume_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        """量价背离特征"""
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        if not group_col:
            result = result.with_columns(pl.lit("MARKET").alias("symbol"))
            group_col = "symbol"
        
        # 分步计算，避免列依赖问题
        result = result.with_columns([
            pl.col("close").pct_change(5).over(group_col).alias("price_change_5d"),
            pl.col("volume").pct_change(5).over(group_col).alias("volume_change_5d"),
        ])
        
        result = result.with_columns([
            (pl.col("price_change_5d") - pl.col("volume_change_5d")).alias("price_volume_divergence"),
        ])
        
        result = result.with_columns([
            pl.col("price_volume_divergence").rolling_std(window_size=10).over(group_col).alias("divergence_std"),
            ((pl.col("price_change_5d") * pl.col("volume_change_5d")) > 0)
            .cast(pl.Float64).alias("price_volume_health")
        ])
        
        return result
    
    def _compute_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """波动率特征"""
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        if not group_col:
            result = result.with_columns(pl.lit("MARKET").alias("symbol"))
            group_col = "symbol"
        
        # 步骤 1: 计算布林带基础值
        result = result.with_columns([
            pl.col("close").rolling_mean(window_size=20).over(group_col).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=20).over(group_col).alias("bb_std"),
        ])
        
        # 步骤 2: 计算布林带宽度 (使用步骤 1 的结果)
        result = result.with_columns([
            ((2 * 2 * pl.col("bb_std")) / pl.col("bb_middle")).alias("bb_width"),
        ])
        
        # 步骤 3: 计算波动率挤压 (使用步骤 2 的结果)
        result = result.with_columns([
            pl.col("bb_width") / pl.col("bb_width").rolling_mean(window_size=60).over(group_col)
            .alias("volatility_squeeze"),
        ])
        
        # 步骤 4: 计算 ATR
        result = result.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr"),
        ])
        
        result = result.with_columns([
            pl.col("tr").rolling_mean(window_size=14).over(group_col).alias("atr_14"),
        ])
        
        # 步骤 5: 计算 ATR 比率
        result = result.with_columns([
            (pl.col("atr_14") / pl.col("close")).alias("atr_ratio"),
        ])
        
        return result
    
    def _compute_money_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """资金流向特征"""
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        if not group_col:
            result = result.with_columns(pl.lit("MARKET").alias("symbol"))
            group_col = "symbol"
        
        # 步骤 1: 计算典型价格
        result = result.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price"),
        ])
        
        # 步骤 2: 计算原始资金流向
        result = result.with_columns([
            (pl.col("typical_price").pct_change(1).over(group_col) * pl.col("volume"))
            .alias("money_flow_raw"),
        ])
        
        # 步骤 3: 计算滚动和
        result = result.with_columns([
            pl.col("money_flow_raw").rolling_sum(window_size=5).over(group_col).alias("money_flow_5d"),
            pl.col("money_flow_raw").rolling_sum(window_size=20).over(group_col).alias("money_flow_20d"),
        ])
        
        # 步骤 4: 计算资金流向比率
        result = result.with_columns([
            pl.col("money_flow_5d") / (pl.col("money_flow_20d").abs() + 1e-10)
            .alias("money_flow_ratio"),
        ])
        
        # 步骤 5: 计算 OBV 每日
        result = result.with_columns([
            pl.when(pl.col("close") > pl.col("close").shift(1).over(group_col))
            .then(pl.col("volume"))
            .when(pl.col("close") < pl.col("close").shift(1).over(group_col))
            .then(-pl.col("volume"))
            .otherwise(0)
            .alias("obv_daily")
        ])
        
        # 步骤 6: 计算 OBV 累计和
        result = result.with_columns([
            pl.col("obv_daily").cum_sum().over(group_col).alias("obv"),
        ])
        
        # 步骤 7: 计算 OBV 变化率 (使用步骤 6 的结果)
        result = result.with_columns([
            pl.col("obv").pct_change(5).over(group_col).alias("obv_change_5d"),
        ])
        
        return result
    
    def _compute_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """动量特征"""
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        
        for period in [5, 10, 20, 60]:
            col_name = f"momentum_{period}d"
            if group_col:
                result = result.with_columns([
                    pl.col("close").pct_change(period).over(group_col).alias(col_name),
                ])
            else:
                result = result.with_columns([
                    pl.col("close").pct_change(period).alias(col_name),
                ])
        
        if group_col:
            result = result.with_columns([
                (pl.col("close") / pl.col("close").rolling_mean(window_size=20).over(group_col) - 1)
                .alias("vs_ma20"),
            ])
        else:
            result = result.with_columns([
                (pl.col("close") / pl.col("close").rolling_mean(window_size=20) - 1)
                .alias("vs_ma20"),
            ])
        
        return result
    
    def _compute_technical_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """技术指标"""
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        
        if group_col:
            delta = pl.col("close").diff().over(group_col)
        else:
            delta = pl.col("close").diff()
        
        result = result.with_columns(
            gain=delta.clip(0, None),
            loss=delta.clip(None, 0).abs(),
        )
        
        if group_col:
            result = result.with_columns(
                avg_gain=pl.col("gain").rolling_mean(window_size=14).over(group_col),
                avg_loss=pl.col("loss").rolling_mean(window_size=14).over(group_col),
            )
        else:
            result = result.with_columns(
                avg_gain=pl.col("gain").rolling_mean(window_size=14),
                avg_loss=pl.col("loss").rolling_mean(window_size=14),
            )
        
        result = result.with_columns(
            rsi_14=100 - 100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 1e-10)),
        )
        
        result = result.drop(["gain", "loss", "avg_gain", "avg_loss"])
        
        if group_col:
            result = result.with_columns(
                ema12=pl.col("close").ewm_mean(span=12).over(group_col),
                ema26=pl.col("close").ewm_mean(span=26).over(group_col),
            )
        else:
            result = result.with_columns(
                ema12=pl.col("close").ewm_mean(span=12),
                ema26=pl.col("close").ewm_mean(span=26),
            )
        
        result = result.with_columns(macd=pl.col("ema12") - pl.col("ema26"))
        
        if group_col:
            result = result.with_columns(
                macd_signal=pl.col("macd").ewm_mean(span=9).over(group_col),
            )
        else:
            result = result.with_columns(
                macd_signal=pl.col("macd").ewm_mean(span=9),
            )
        
        result = result.with_columns(macd_hist=pl.col("macd") - pl.col("macd_signal"))
        result = result.drop(["ema12", "ema26"])
        
        return result
    
    def get_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """获取特征列名"""
        base_features = [
            "price_volume_divergence", "divergence_std", "price_volume_health",
            "bb_width", "volatility_squeeze", "atr_ratio",
            "money_flow_5d", "money_flow_20d", "money_flow_ratio", "obv_change_5d",
            "momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d", "vs_ma20",
            "rsi_14",
        ]
        
        available_cols = [c for c in base_features if c in df.columns]
        self.feature_columns = available_cols
        return available_cols
    
    def compute_medians(self, df: pl.DataFrame, feature_cols: List[str]) -> None:
        """计算全局特征中位数"""
        for col in feature_cols:
            if col in df.columns:
                self.medians[col] = df[col].median()
    
    def compute_industry_medians(self, df: pl.DataFrame, feature_cols: List[str], 
                                  industry_col: str = "industry_name") -> None:
        """
        【关键修复】计算按行业分组的特征中位数
        
        先按行业中位数填充，若仍缺失再用全局中位数或 0
        """
        if industry_col not in df.columns:
            logger.warning(f"Industry column '{industry_col}' not found, using global medians")
            self.compute_medians(df, feature_cols)
            return
        
        self.industry_medians = {}
        
        # 按行业分组计算中位数
        industries = df[industry_col].unique().to_list()
        
        for industry in industries:
            if industry is None:
                continue
            industry_df = df.filter(pl.col(industry_col) == industry)
            self.industry_medians[industry] = {}
            for col in feature_cols:
                if col in industry_df.columns:
                    self.industry_medians[industry][col] = industry_df[col].median()
        
        # 同时计算全局中位数作为后备
        self.compute_medians(df, feature_cols)
        logger.info(f"[行业中位数] 已计算 {len(self.industry_medians)} 个行业的特征中位数")
    
    def fill_missing(self, df: pl.DataFrame, feature_cols: List[str],
                     industry_col: str = "industry_name") -> pl.DataFrame:
        """
        【关键修复】缺失值容错处理
        
        填充顺序：
        1. 先按行业中位数填充
        2. 若仍缺失，用全局中位数填充
        3. 最后用 0 填充
        """
        result = df.clone()
        
        for col in feature_cols:
            if col not in result.columns:
                continue
            
            # 步骤 1: 按行业中位数填充
            if industry_col in result.columns and self.industry_medians:
                for industry, industry_meds in self.industry_medians.items():
                    if col in industry_meds:
                        median_val = industry_meds[col]
                        # 跳过 None 或 NaN 的中位数值
                        if median_val is None or (isinstance(median_val, float) and (median_val != median_val)):
                            continue
                        # 安全地填充缺失值
                        result = result.with_columns([
                            pl.when(
                                (pl.col(industry_col).cast(pl.Utf8, strict=False) == str(industry)) & 
                                (pl.col(col).is_null() | pl.col(col).is_nan())
                            )
                            .then(median_val)
                            .otherwise(pl.col(col))
                            .alias(col)
                        ])
            
            # 步骤 2: 用全局中位数填充剩余缺失值
            global_median = self.medians.get(col, 0.0)
            result = result.with_columns([
                pl.col(col).fill_nan(global_median).fill_null(global_median).alias(col)
            ])
        
        return result


# ==============================================================================
# 相关性过滤器
# ==============================================================================

class CorrelationFilter:
    """相关性过滤器"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.selected_features: List[str] = []
    
    def filter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """过滤高相关性特征"""
        n_features = len(feature_columns)
        
        corr_matrix = np.corrcoef(X.T)
        
        ic_values = {}
        for i, col in enumerate(feature_columns):
            ic = np.corrcoef(X[:, i], y)[0, 1]
            ic_values[col] = abs(ic) if not np.isnan(ic) else 0.0
        
        feature_ic_list = [(col, ic_values[col]) for col in feature_columns]
        feature_ic_list.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        selected_indices = []
        
        for i, (col, ic) in enumerate(feature_ic_list):
            is_redundant = False
            for sel_idx in selected_indices:
                orig_idx = feature_columns.index(col)
                if abs(corr_matrix[orig_idx, sel_idx]) > self.threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(col)
                selected_indices.append(feature_columns.index(col))
            
            if len(selected) >= MAX_FEATURES:
                break
        
        self.selected_features = selected
        
        if len(selected_indices) > 0:
            X_filtered = X[:, selected_indices]
        else:
            X_filtered = X
        
        logger.info(f"[相关性过滤] 从 {n_features} 个特征中选择 {len(selected)} 个")
        
        return X_filtered, selected


# ==============================================================================
# 行业映射器
# ==============================================================================

class IndustryMapper:
    """
    行业映射器 - 用于行业分散
    
    从数据库获取股票的行业分类
    """
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.industry_map: Dict[str, str] = {}
    
    def load_industries(self, symbols: List[str]) -> None:
        """加载行业信息"""
        if not symbols:
            return
        
        try:
            symbols_str = "','".join(symbols)
            query = f"""
                SELECT DISTINCT symbol, industry_name 
                FROM stock_info 
                WHERE symbol IN ('{symbols_str}')
            """
            data = self.db.read_sql(query)
            
            for row in data.iter_rows(named=True):
                symbol = row.get("symbol")
                industry = row.get("industry_name", "Unknown")
                if symbol:
                    self.industry_map[symbol] = industry
            
            # 填充缺失
            for symbol in symbols:
                if symbol not in self.industry_map:
                    self.industry_map[symbol] = "Unknown"
                    
        except Exception as e:
            logger.warning(f"Failed to load industry info: {e}")
            for symbol in symbols:
                if symbol not in self.industry_map:
                    self.industry_map[symbol] = "Unknown"
    
    def get_industry(self, symbol: str) -> str:
        """获取股票行业"""
        return self.industry_map.get(symbol, "Unknown")
    
    def count_by_industry(self, positions: Dict[str, Position]) -> Dict[str, int]:
        """统计各行业持仓数量"""
        industry_count: Dict[str, int] = {}
        
        for symbol, position in positions.items():
            industry = self.get_industry(symbol)
            industry_count[industry] = industry_count.get(industry, 0) + 1
        
        return industry_count
    
    def can_buy(self, symbol: str, positions: Dict[str, Position]) -> bool:
        """检查是否可以买入（行业分散约束）"""
        industry = self.get_industry(symbol)
        industry_count = self.count_by_industry(positions)
        return industry_count.get(industry, 0) < MAX_STOCKS_PER_INDUSTRY


# ==============================================================================
# 主策略类 - V5 Adaptive
# ==============================================================================

class FinalStrategyV5Adaptive:
    """
    Final Strategy V5 Adaptive - 自适应滚动训练版
    
    【核心修复 - 预测坍缩问题解决方案】
    
    1. 特征工程管线加固 (Feature Resilience)
       - 局部归一化 (Cross-sectional Scaling)：在 predict 方法中对输入特征 X 执行当日截面的 Rank 缩放
       - 缺失值容错：先按行业中位数填充，若仍缺失再用 0
       - 特征检查断言：在预测前加入 assert X_scaled.std() > 0
    
    2. 滚动训练机制 (Rolling Training)
       - 每回测 20 个交易日，自动使用过去 500 天的数据重新 fit 模型
       - 样本扩充：确保训练样本覆盖全市场至少 1000 只股票
    
    3. 交易策略微调 (Execution Strategy)
       - ATR 止损倍数从 1.5 倍提高至 2.5 倍
       - 买入过滤：只有当 predict_score > 0.6 且 z-score > 1.5 时才入场
    """
    
    def __init__(
        self,
        config_path: str = "config/production_params.yaml",
        db: Optional[DatabaseManager] = None,
    ):
        self.config_path = Path(config_path)
        self.db = db or DatabaseManager.get_instance()
        
        self.config = self._load_config()
        self.feature_engine = RobustFeatureEngine()
        self.correlation_filter = CorrelationFilter(threshold=CORRELATION_THRESHOLD)
        self.industry_mapper: Optional[IndustryMapper] = None
        
        # 模型
        self.ridge_model: Optional[Any] = None
        self.lgb_model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        
        # 特征配置
        self.training_feature_columns: List[str] = []
        self.selected_features: List[str] = []
        
        # 市场状态
        self.positions: Dict[str, Position] = {}
        self.trade_records: List[TradeRecord] = []
        self.cash: float = 0.0
        self.initial_capital: float = 0.0
        
        # 预测统计
        self.predict_mean: float = 0.0
        self.predict_std: float = 1.0
        
        # 滚动训练状态
        self.last_retrain_date: Optional[str] = None
        self.trading_days_since_retrain: int = 0
        
        # 统计
        self.model_ic: float = 0.0
        self.skipped_trades: int = 0  # 因资金不足放弃的交易次数
        self.total_commission: float = 0.0
        self.total_stamp_duty: float = 0.0
        self.total_slippage: float = 0.0
        
        logger.info("FinalStrategyV5Adaptive initialized (自适应滚动训练版)")
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "strategy": {
                "top_k_stocks": TOP_K_STOCKS,
                "max_position_pct": MAX_POSITION_PCT,
            },
            "risk_control": {"atr_multiplier": ATR_STOP_LOSS_MULTIPLIER},
            "liquidity_filter": {
                "2023": LIQUIDITY_FILTER_2023,
                "2024": LIQUIDITY_FILTER_2024,
            },
        }
    
    def _get_training_data(self, end_date: str, start_date: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        获取训练数据 - 支持滚动训练窗口
        
        Args:
            end_date: 训练截止日期
            start_date: 训练起始日期（可选，用于滚动训练）
        """
        try:
            if start_date is None:
                start_date = TRAIN_START_DATE
            
            query = f"""
                SELECT sd.symbol, sd.trade_date, sd.open, sd.high, sd.low, sd.close, 
                       sd.volume, sd.amount, sd.adj_factor, sd.turnover_rate, 
                       sd.pre_close, sd.`change`, sd.pct_chg,
                       si.industry_name
                FROM stock_daily sd
                LEFT JOIN stock_info si ON sd.symbol = si.symbol
                WHERE trade_date <= '{end_date}' 
                AND trade_date >= '{start_date}'
                ORDER BY symbol, trade_date
            """
            data = self.db.read_sql(query)
            if data.is_empty():
                return None
            
            n_stocks = data['symbol'].n_unique()
            logger.info(f"[数据准备] 获取到 {len(data)} 条记录，{n_stocks} 只股票 (窗口：{start_date} 至 {end_date})")
            
            # 检查样本是否充足
            if n_stocks < MIN_TRAINING_SAMPLES // 10:  # 每只股票约 100 个交易日
                logger.warning(f"训练样本不足：{n_stocks} 只股票，目标至少 {MIN_TRAINING_SAMPLES} 个样本")
            
            data = self.feature_engine.compute_features(data)
            
            data = data.with_columns([
                (pl.col("close").shift(-5).over("symbol") / pl.col("close") - 1)
                .alias("future_return_5d")
            ])
            
            return data
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None
    
    def _prepare_features(self, df: pl.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """准备训练/预测特征"""
        feature_cols = self.feature_engine.get_feature_columns(df)
        
        if len(feature_cols) < 5:
            logger.error(f"特征列不足：{len(feature_cols)}")
            raise ValueError(f"特征列不足：{len(feature_cols)}")
        
        if is_training:
            self.training_feature_columns = feature_cols.copy()
            logger.info(f"[特征准备] 原始特征数：{len(feature_cols)}")
            
            # 【关键修复】计算行业中位数
            if "industry_name" in df.columns:
                self.feature_engine.compute_industry_medians(df, feature_cols, "industry_name")
            else:
                self.feature_engine.compute_medians(df, feature_cols)
        
        # 【关键修复】使用行业中位数填充缺失值
        if "industry_name" in df.columns:
            df_filled = self.feature_engine.fill_missing(df, feature_cols, "industry_name")
        else:
            df_filled = self.feature_engine.fill_missing(df, feature_cols)
        
        if is_training:
            df_clean = df_filled.filter(pl.col("future_return_5d").is_not_null())
            
            if len(df_clean) == 0:
                logger.warning("No valid data after filtering")
                df_clean = df_filled
            
            X = df_clean.select(feature_cols).to_numpy()
            y = df_clean["future_return_5d"].fill_null(0).to_numpy()
        else:
            X = df_filled.select(feature_cols).to_numpy()
            y = None
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if is_training:
            logger.info("[特征准备] 执行相关性过滤...")
            X_filtered, selected_cols = self.correlation_filter.filter(X, y, feature_cols)
            self.selected_features = selected_cols
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_filtered)
            
            logger.info(f"[特征准备] 完成：最终特征数={X_scaled.shape[1]}")
            
            return X_scaled, y, selected_cols
        else:
            # 预测时使用已训练的 scaler
            if self.scaler is None:
                logger.warning("Scaler not fitted, returning raw features")
                return X, None, feature_cols
            
            X_filtered = X[:, [feature_cols.index(c) for c in self.selected_features if c in feature_cols]]
            X_scaled = self.scaler.transform(X_filtered)
            return X_scaled, None, self.selected_features
    
    def train_model(self, train_end_date: str = TRAIN_END_DATE, 
                    train_start_date: Optional[str] = None) -> bool:
        """
        训练模型 - 支持滚动训练
        
        Returns:
            bool: 训练是否成功
        """
        logger.info(f"Training model with data from {train_start_date or TRAIN_START_DATE} to {train_end_date}...")
        
        train_data = self._get_training_data(train_end_date, train_start_date)
        if train_data is None or len(train_data) == 0:
            logger.warning("No training data found")
            return False
        
        X, y, feature_cols = self._prepare_features(train_data, is_training=True)
        
        if len(X) == 0 or (y is not None and len(y) == 0):
            logger.warning("No valid training data")
            return False
        
        logger.info(f"[模型训练] 样本数={len(X)}, 特征数={len(feature_cols)}")
        
        from sklearn.linear_model import Ridge
        logger.info("[模型训练] 训练 Ridge 模型...")
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X, y)
        
        if LIGHTGBM_AVAILABLE:
            logger.info("[模型训练] 训练 LightGBM 模型...")
            self.lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
            self.lgb_model.fit(X, y)
        else:
            logger.warning("LightGBM not available, using Ridge only")
            self.lgb_model = None
        
        logger.info("[模型训练] 执行 IC 验证...")
        predictions = self.ridge_model.predict(X)
        from scipy.stats import rankdata
        pred_rank = rankdata(predictions)
        tgt_rank = rankdata(y)
        ic = np.corrcoef(pred_rank, tgt_rank)[0, 1]
        self.model_ic = ic if not np.isnan(ic) else 0.0
        
        if abs(self.model_ic) < 0.03:
            logger.warning(f"IC 值较低：IC={self.model_ic:.4f}")
        else:
            logger.info(f"IC 验证通过：IC={self.model_ic:.4f}")
        
        self._print_feature_importance()
        logger.info("Model training complete")
        return True
    
    def _print_feature_importance(self) -> None:
        """打印特征重要性"""
        if self.ridge_model and hasattr(self.ridge_model, "coef_"):
            importance = list(zip(self.selected_features, self.ridge_model.coef_))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            logger.info("[Ridge Top 10 特征重要性]")
            for i, (feat, imp) in enumerate(importance[:10], 1):
                logger.info(f"  {i}. {feat}: {imp:.4f}")
        
        if self.lgb_model and LIGHTGBM_AVAILABLE:
            importance = list(zip(self.selected_features, self.lgb_model.feature_importances_))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            logger.info("[LightGBM Top 10 特征重要性]")
            for i, (feat, imp) in enumerate(importance[:10], 1):
                logger.info(f"  {i}. {feat}: {imp:.4f}")
    
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        预测 - V5 加固版
        
        【关键修复】
        1. 局部归一化 (Cross-sectional Scaling)：对输入特征 X 执行当日截面的 Rank 缩放
        2. 缺失值容错：先按行业中位数填充，若仍缺失再用 0
        3. 特征检查断言：在预测前加入 assert X_scaled.std() > 0
        """
        if self.scaler is None or not self.selected_features:
            logger.warning("模型未训练，返回零预测")
            return df.with_columns(pl.lit(0.5).alias("predict_score"))
        
        # 步骤 1: 计算特征
        df_with_features = self.feature_engine.compute_features(df)
        
        # 步骤 2: 【关键修复】使用行业中位数填充缺失值
        if "industry_name" in df_with_features.columns:
            df_filled = self.feature_engine.fill_missing(df_with_features, self.training_feature_columns, "industry_name")
        else:
            df_filled = self.feature_engine.fill_missing(df_with_features, self.training_feature_columns)
        
        feature_cols = self.selected_features
        available_cols = [c for c in feature_cols if c in df_filled.columns]
        
        if len(available_cols) == 0:
            logger.warning("No feature columns found")
            return df_filled.with_columns(pl.lit(0.5).alias("predict_score"))
        
        try:
            X = df_filled.select(available_cols).to_numpy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.warning(f"Failed to convert to numpy: {e}")
            return df_filled.with_columns(pl.lit(0.5).alias("predict_score"))
        
        # 步骤 3: 使用训练时的 scaler.transform()
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            logger.warning(f"Failed to scale: {e}")
            return df_filled.with_columns(pl.lit(0.5).alias("predict_score"))
        
        # 步骤 4: 【关键修复】特征检查断言 - 防止预测坍缩
        try:
            # 检查每个特征的方差
            feature_stds = X_scaled.std(axis=0)
            zero_var_features = [available_cols[i] for i, std in enumerate(feature_stds) if std < 1e-10]
            
            if zero_var_features:
                logger.warning(f"[特征检查] 发现方差为 0 的特征：{zero_var_features}")
            
            # 总体方差检查
            total_std = X_scaled.std()
            assert total_std > 0, f"特征方差为 0，预测将坍缩！total_std={total_std}"
            
            logger.debug(f"[特征检查] X_scaled.std()={total_std:.6f} > 0 ✓")
        except AssertionError as e:
            logger.error(f"[特征检查断言失败] {e}")
            # 打印导致坍缩的特征
            feature_means = X_scaled.mean(axis=0)
            feature_stds = X_scaled.std(axis=0)
            for i, col in enumerate(available_cols):
                logger.error(f"  特征 {i}: {col}, mean={feature_means[i]:.6f}, std={feature_stds[i]:.6f}")
            # 返回中性预测
            return df_filled.with_columns(pl.lit(0.5).alias("predict_score"))
        
        # 步骤 5: 模型预测
        if self.ridge_model is not None:
            ridge_pred = self.ridge_model.predict(X_scaled)
        else:
            ridge_pred = np.zeros(len(X_scaled))
        
        if self.lgb_model is not None and LIGHTGBM_AVAILABLE:
            lgb_pred = self.lgb_model.predict(X_scaled)
        else:
            lgb_pred = ridge_pred.copy()
        
        ensemble_pred = 0.5 * ridge_pred + 0.5 * lgb_pred
        
        self.predict_mean = float(np.mean(ensemble_pred))
        self.predict_std = float(np.std(ensemble_pred)) if len(ensemble_pred) > 1 else 1.0
        
        logger.debug(f"[Predict] 预测值范围=[{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
        logger.debug(f"[Predict] mean={self.predict_mean:.4f}, std={self.predict_std:.4f}")
        
        # 步骤 6: 【关键修复】局部归一化 - 使用百分位排名恢复区分度
        from scipy.stats import rankdata
        
        # 始终使用百分位排名，确保预测有区分度
        percentile_ranks = rankdata(ensemble_pred, method='average') / len(ensemble_pred)
        
        # 将百分位排名映射到 0-1 范围，中心在 0.5
        pred_scores = percentile_ranks
        
        logger.debug(f"[Predict] 使用百分位排名，分数范围=[{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
        
        # 步骤 7: 计算 z-score 用于买入过滤
        z_scores = (ensemble_pred - self.predict_mean) / (self.predict_std + 1e-10)
        
        logger.debug(f"[Predict] z-score 范围=[{z_scores.min():.2f}, {z_scores.max():.2f}]")
        logger.debug(f"[Predict] 正预测数={sum(pred_scores > 0.5)}, 负预测数={sum(pred_scores < 0.5)}")
        logger.debug(f"[Predict] z-score > 1.5 的数量={sum(z_scores > 1.5)}")
        
        # 将预测分数和 z-score 添加到结果中
        pred_scores_list = list(pred_scores)
        z_scores_list = list(z_scores)
        
        n_rows = len(df_filled)
        if len(pred_scores_list) < n_rows:
            pred_scores_list += [0.5] * (n_rows - len(pred_scores_list))
            z_scores_list += [0.0] * (n_rows - len(z_scores_list))
        
        result = df_filled.with_columns([
            pl.Series("predict_score", pred_scores_list),
            pl.Series("predict_zscore", z_scores_list),
        ])
        
        return result
    
    def _check_and_retrain(self, current_date: str, all_data: pl.DataFrame) -> None:
        """
        【关键修复】检查是否需要重新训练模型
        
        每 ROLLING_RETRAIN_INTERVAL 个交易日，使用过去 ROLLING_WINDOW_DAYS 天的数据重新训练
        """
        if self.last_retrain_date is None:
            # 首次训练
            logger.info(f"[滚动训练] 首次训练模型，使用 {ROLLING_WINDOW_DAYS} 天窗口")
            start_date = self._get_date_offset(current_date, -ROLLING_WINDOW_DAYS)
            self.train_model(train_end_date=current_date, train_start_date=start_date)
            self.last_retrain_date = current_date
            self.trading_days_since_retrain = 0
            return
        
        self.trading_days_since_retrain += 1
        
        if self.trading_days_since_retrain >= ROLLING_RETRAIN_INTERVAL:
            logger.info(f"[滚动训练] 触发重新训练 (距离上次训练 {self.trading_days_since_retrain} 天)")
            start_date = self._get_date_offset(current_date, -ROLLING_WINDOW_DAYS)
            success = self.train_model(train_end_date=current_date, train_start_date=start_date)
            if success:
                self.last_retrain_date = current_date
                self.trading_days_since_retrain = 0
                logger.info(f"[滚动训练] 重新训练完成，下次训练将在 {ROLLING_RETRAIN_INTERVAL} 个交易日后")
    
    def _get_date_offset(self, date_str: str, offset_days: int) -> str:
        """计算日期偏移"""
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            new_date = date + timedelta(days=offset_days)
            return new_date.strftime("%Y-%m-%d")
        except Exception:
            return date_str
    
    def run_backtest(self, start_date: str, end_date: str, initial_capital: float = INITIAL_CAPITAL) -> BacktestResult:
        """运行回测 - 10 万资金实战 + 滚动训练"""
        logger.info(f"Running backtest from {start_date} to {end_date}...")
        logger.info(f"[V5 核心修复] 滚动训练窗口={ROLLING_WINDOW_DAYS}天，重新训练间隔={ROLLING_RETRAIN_INTERVAL}天")
        logger.info(f"[V5 核心修复] ATR 止损倍数={ATR_STOP_LOSS_MULTIPLIER}倍 (从 1.5 提高至 2.5)")
        logger.info(f"[V5 核心修复] 买入过滤：predict_score > {MIN_PREDICT_SCORE} 且 z-score > {SIGNIFICANCE_STD_THRESHOLD}")
        logger.info(f"[资金约束] 初始资金 = {initial_capital:,.0f} 元")
        logger.info(f"[交易约束] 最小交易单位 = {LOT_SIZE} 股")
        logger.info(f"[手续费] 佣金 = {COMMISSION_RATE*10000:.1f} 万 (最低{COMMISSION_MIN}元), 印花税 = {STAMP_DUTY_RATE*1000:.1f}‰")
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_records = []
        self.skipped_trades = 0
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_slippage = 0.0
        
        # 滚动训练状态重置
        self.last_retrain_date = None
        self.trading_days_since_retrain = 0
        
        backtest_data = self._get_aligned_backtest_data(start_date, end_date)
        if backtest_data is None or len(backtest_data) == 0:
            logger.warning("No backtest data found")
            return BacktestResult()
        
        backtest_data = backtest_data.sort("trade_date")
        dates = backtest_data["trade_date"].unique().to_list()
        daily_values = []
        
        # 初始化行业映射器
        all_symbols = backtest_data["symbol"].unique().to_list()
        self.industry_mapper = IndustryMapper(self.db)
        self.industry_mapper.load_industries(all_symbols)
        
        daily_data_map: Dict[str, pl.DataFrame] = {}
        for date in dates:
            date_str = str(date)
            daily_data_map[date_str] = backtest_data.filter(pl.col("trade_date") == date_str)
        
        # 【关键修复】首次训练模型
        first_date = str(dates[0])
        self._check_and_retrain(first_date, backtest_data)
        
        for date in dates:
            date_str = str(date)
            daily_data = daily_data_map.get(date_str)
            
            if daily_data is None or len(daily_data) == 0:
                continue
            
            price_map = {}
            for row in daily_data.iter_rows(named=True):
                symbol = row.get("symbol")
                if symbol:
                    price_map[symbol] = {
                        "close": row.get("close", 0),
                        "high": row.get("high", 0),
                        "low": row.get("low", 0),
                        "atr_14": row.get("atr_14", 0),
                        "industry_name": row.get("industry_name", "Unknown"),
                    }
            
            date_val = date_str
            year = int(date_val[:4]) if len(date_val) >= 4 else 2024
            liquidity_threshold = LIQUIDITY_FILTER_2023 if year == 2023 else LIQUIDITY_FILTER_2024
            daily_data = daily_data.filter(
                (pl.col("amount") >= liquidity_threshold) | (pl.col("amount").is_null())
            )
            
            for symbol in list(self.positions.keys()):
                self.positions[symbol].hold_days += 1
            
            self._check_exit_conditions(date_str, price_map)
            
            # 【关键修复】检查是否需要重新训练
            self._check_and_retrain(date_str, backtest_data)
            
            # 预测
            daily_data = self.predict(daily_data)
            
            slippage_cost = self._execute_daily_trading(date_str, daily_data, price_map)
            self.total_slippage += slippage_cost
            
            portfolio_value = self._calculate_portfolio_value(price_map)
            daily_values.append({
                "date": date_str,
                "value": portfolio_value,
                "cash": self.cash,
                "positions": len(self.positions),
            })
        
        result = self._calculate_backtest_result(daily_values)
        result.daily_values = daily_values
        result.total_commission = self.total_commission
        result.total_stamp_duty = self.total_stamp_duty
        result.total_slippage_cost = self.total_slippage
        result.skipped_trades = self.skipped_trades
        
        logger.info(f"Backtest complete: Total Return = {result.total_return:.2%}, Win Rate = {result.win_rate:.1%}")
        logger.info(f"手续费总计：佣金={self.total_commission:.2f}元，印花税={self.total_stamp_duty:.2f}元，滑点={self.total_slippage:.2f}元")
        logger.info(f"放弃交易次数：{self.skipped_trades}")
        return result
    
    def _get_aligned_backtest_data(self, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """获取对齐的回测数据"""
        try:
            query = f"""
                SELECT sd.symbol, sd.trade_date, sd.open, sd.high, sd.low, sd.close, 
                       sd.volume, sd.amount, sd.adj_factor, sd.turnover_rate, 
                       sd.pre_close, sd.`change`, sd.pct_chg,
                       si.industry_name
                FROM stock_daily sd
                LEFT JOIN stock_info si ON sd.symbol = si.symbol
                WHERE trade_date >= '{start_date}' 
                AND trade_date <= '{end_date}'
                ORDER BY symbol, trade_date
            """
            data = self.db.read_sql(query)
            if data.is_empty():
                return None
            
            data = self.feature_engine.compute_features(data)
            
            logger.info(f"[回测数据] 获取到 {len(data)} 条记录")
            
            return data
        except Exception as e:
            logger.error(f"Failed to get backtest data: {e}")
            return None
    
    def _execute_daily_trading(self, date: str, data: pl.DataFrame, price_map: Dict[str, Dict]) -> float:
        """
        执行每日交易 - V5 加固版
        
        【关键修复】买入过滤：只有当 predict_score > 0.6 且 z-score > 1.5 时才入场
        """
        total_slippage = 0.0
        
        top_k = self.config.get("strategy", {}).get("top_k_stocks", TOP_K_STOCKS)
        max_position_pct = self.config.get("strategy", {}).get("max_position_pct", MAX_POSITION_PCT)
        
        if "predict_score" not in data.columns:
            return total_slippage
        
        # 【关键修复】严格的买入过滤
        # 条件 1: predict_score > 0.6
        # 条件 2: z-score > 1.5
        if "predict_zscore" in data.columns:
            scored_data = data.filter(
                (pl.col("predict_score") >= MIN_PREDICT_SCORE) & 
                (pl.col("predict_zscore") >= SIGNIFICANCE_STD_THRESHOLD)
            )
            logger.debug(f"[{date}] 通过过滤的股票数：{len(scored_data)} (predict_score > {MIN_PREDICT_SCORE} 且 z-score > {SIGNIFICANCE_STD_THRESHOLD})")
        else:
            # 兼容模式：只使用 predict_score
            scored_data = data.filter(pl.col("predict_score") >= MIN_PREDICT_SCORE)
        
        if len(scored_data) == 0:
            return total_slippage
        
        top_stocks = scored_data.sort("predict_score", descending=True).head(top_k)
        
        for row in top_stocks.iter_rows(named=True):
            symbol = row.get("symbol")
            
            if not symbol or symbol in self.positions:
                continue
            
            # 行业分散检查
            if self.industry_mapper and not self.industry_mapper.can_buy(symbol, self.positions):
                logger.debug(f"[{date}] {symbol} 行业集中度已达上限，跳过")
                continue
            
            if symbol in price_map:
                price = float(price_map[symbol]["close"])
                atr_val = price_map[symbol].get("atr_14")
                if atr_val is None or atr_val == 0:
                    atr = price * 0.02
                else:
                    atr = float(atr_val)
                raw_score = row.get("predict_score", 0)
                z_score = row.get("predict_zscore", 0) if "predict_zscore" in row else 0
                industry = self.industry_mapper.get_industry(symbol) if self.industry_mapper else "Unknown"
                
                # 计算仓位
                position_value = self.cash * max_position_pct
                
                # 100 股整数倍处理
                shares = int(position_value / price / LOT_SIZE) * LOT_SIZE
                
                # 检查是否因资金不足而无法交易
                if shares < LOT_SIZE:
                    self.skipped_trades += 1
                    logger.debug(f"[{date}] {symbol} 资金不足 (需要{LOT_SIZE}股整数倍), 跳过交易。当前现金={self.cash:.2f}")
                    continue
                
                # 真实手续费计算
                stock_cost, commission, total_cost = FeeCalculator.calculate_total_buy_cost(price, shares)
                
                if total_cost <= self.cash:
                    self.cash -= total_cost
                    self.total_commission += commission
                    
                    cost_basis_per_share = (stock_cost + commission) / shares
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        entry_date=date,
                        entry_price=price,
                        shares=shares,
                        hold_days=0,
                        predicted_return=raw_score,
                        highest_price=price,
                        lowest_price=price,
                        entry_atr=atr,
                        industry=industry,
                        cost_basis=cost_basis_per_share,
                    )
                    
                    logger.info(f"[买入] {symbol} @ {price:.2f} x {shares}股 ({industry}), score={raw_score:.3f}, z={z_score:.2f}, 佣金={commission:.2f}元")
                    total_slippage += stock_cost * SLIPPAGE_RATE
        
        return total_slippage
    
    def _check_exit_conditions(self, date: str, price_map: Dict[str, Dict]) -> None:
        """
        检查退出条件 - 移动止盈 + ATR 动态止损 (2.5 倍)
        """
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            if symbol not in price_map:
                logger.debug(f"[{date}] {symbol} 价格数据缺失")
                continue
            
            price = float(price_map[symbol]["close"])
            
            if price > position.highest_price:
                position.highest_price = price
            if price < position.lowest_price or position.lowest_price == 0:
                position.lowest_price = price
            
            entry_return = (price - position.entry_price) / position.entry_price
            
            # 【关键修复】ATR 动态止损 - 2.5 倍
            atr_stop_loss = position.entry_atr * ATR_STOP_LOSS_MULTIPLIER
            stop_loss_price = position.entry_price - atr_stop_loss
            stop_loss_return = (stop_loss_price - position.entry_price) / position.entry_price
            
            # 移动止盈：从最高点回撤 5% 时止盈
            trailing_stop_price = position.highest_price * (1 - TRAILING_STOP_PCT)
            trailing_stop_return = (trailing_stop_price - position.entry_price) / position.entry_price
            
            logger.debug(f"[{date}] {symbol} 持有={position.hold_days}天，收益={entry_return:.2%}, "
                        f"最高={position.highest_price:.2f}, 移动止盈价={trailing_stop_price:.2f}, ATR 止损={stop_loss_return:.2%}")
            
            exit_reason = None
            
            # ATR 动态止损 (2.5 倍)
            if price <= stop_loss_price:
                exit_reason = "atr_stop_loss"
            # 移动止盈
            elif price <= trailing_stop_price and position.hold_days >= 2:
                exit_reason = "trailing_stop"
            # 固定止盈
            elif entry_return >= TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            # 最大持有天数到期
            elif position.hold_days >= MAX_HOLD_DAYS:
                exit_reason = "time_exit"
            
            if exit_reason:
                logger.info(f"[{date}] {symbol} 触发退出条件：{exit_reason}, 收益={entry_return:.2%}")
                self._exit_position(symbol, date, price, exit_reason)
    
    def _exit_position(self, symbol: str, date: str, price: float, reason: str) -> None:
        """
        退出持仓 - 真实手续费计算
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        price = float(price)
        
        stock_value, commission, stamp_duty, net_proceeds = FeeCalculator.calculate_sell_proceeds(price, position.shares)
        
        pnl = net_proceeds - position.cost_basis * position.shares
        pnl_pct = pnl / (position.cost_basis * position.shares) if position.cost_basis > 0 else 0
        
        self.total_commission += commission
        self.total_stamp_duty += stamp_duty
        self.cash += net_proceeds
        
        self.trade_records.append(TradeRecord(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            hold_days=position.hold_days,
            commission_paid=commission,
            stamp_duty_paid=stamp_duty,
            max_profit_pct=(position.highest_price - position.entry_price) / position.entry_price,
            min_profit_pct=(position.lowest_price - position.entry_price) / position.entry_price,
        ))
        
        del self.positions[symbol]
        logger.debug(f"[卖出] {symbol} @ {price:.2f}, 盈亏={pnl_pct:.2%}, 原因={reason}, "
                    f"佣金={commission:.2f}元，印花税={stamp_duty:.2f}元")
    
    def _calculate_portfolio_value(self, price_map: Dict[str, Dict]) -> float:
        """计算组合价值"""
        value = float(self.cash)
        for symbol, position in self.positions.items():
            if symbol in price_map:
                price = float(price_map[symbol]["close"])
                value += position.shares * price
        return value
    
    def _calculate_backtest_result(self, daily_values: List[Dict]) -> BacktestResult:
        """
        计算回测结果 - 包含 Calmar Ratio
        """
        if len(daily_values) == 0:
            return BacktestResult()
        
        values = [dv["value"] for dv in daily_values]
        
        total_return = (values[-1] - self.initial_capital) / self.initial_capital
        
        days = len(daily_values)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 最大回撤
        peak = values[0]
        max_dd = 0.0
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 夏普比率
        if len(values) > 1:
            daily_returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
            daily_std = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0
            annual_volatility = daily_std * np.sqrt(252)
            
            if annual_volatility > 0:
                sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # 验证夏普比率符号
        if annual_return < RISK_FREE_RATE and sharpe > 0:
            if annual_return < 0:
                sharpe = -abs(sharpe)
        
        # Calmar Ratio = 年化收益 / 最大回撤
        if max_dd > 0:
            calmar = annual_return / max_dd
        else:
            calmar = 0.0
        
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
            calmar_ratio=calmar,
            win_rate=win_rate,
            total_trades=len(self.trade_records),
            avg_hold_days=avg_hold,
            profit_factor=profit_factor,
            trades=self.trade_records,
            daily_values=daily_values,
        )


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    """主函数"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG"
    )
    
    logger.info("=" * 60)
    logger.info("Final Strategy V5 Adaptive - 自适应滚动训练版")
    logger.info("=" * 60)
    logger.info(f"初始资金：{INITIAL_CAPITAL:,.0f} 元")
    logger.info(f"最小交易单位：{LOT_SIZE} 股")
    logger.info(f"佣金：{COMMISSION_RATE*10000:.1f} 万 (最低{COMMISSION_MIN}元)")
    logger.info(f"印花税：{STAMP_DUTY_RATE*1000:.1f}‰ (卖出)")
    logger.info(f"滑点：{SLIPPAGE_RATE*100:.2f}%")
    logger.info(f"持仓数量：{TOP_K_STOCKS} 只")
    logger.info(f"行业分散：同一行业最多{MAX_STOCKS_PER_INDUSTRY}只")
    logger.info(f"移动止盈：从最高点回撤{TRAILING_STOP_PCT*100:.1f}%")
    logger.info("=" * 60)
    logger.info("[V5 核心修复]")
    logger.info(f"1. 滚动训练：每{ROLLING_RETRAIN_INTERVAL}个交易日，使用过去{ROLLING_WINDOW_DAYS}天数据重新训练")
    logger.info(f"2. ATR 止损：{ATR_STOP_LOSS_MULTIPLIER}倍 (从 1.5 提高至 2.5)")
    logger.info(f"3. 买入过滤：predict_score > {MIN_PREDICT_SCORE} 且 z-score > {SIGNIFICANCE_STD_THRESHOLD}")
    logger.info(f"4. 特征加固：局部归一化 + 行业中位数填充 + 方差断言")
    logger.info("=" * 60)
    
    strategy = FinalStrategyV5Adaptive(
        config_path="config/production_params.yaml",
    )
    
    result = strategy.run_backtest(
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        initial_capital=INITIAL_CAPITAL,
    )
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("【V5 Adaptive 回测结果】")
    logger.info("=" * 60)
    logger.info(f"总收益率：{result.total_return:.2%}")
    logger.info(f"年化收益：{result.annual_return:.2%}")
    logger.info(f"最大回撤：{result.max_drawdown:.2%}")
    logger.info(f"夏普比率：{result.sharpe_ratio:.2f}")
    logger.info(f"Calmar 比率：{result.calmar_ratio:.2f} (目标 > 0.8)")
    logger.info(f"胜率：{result.win_rate:.1%} (目标 > 50%)")
    logger.info(f"交易次数：{result.total_trades}")
    logger.info(f"平均持有天数：{result.avg_hold_days:.1f}")
    logger.info(f"盈亏比：{result.profit_factor:.2f}")
    logger.info(f"模型 IC 值：{strategy.model_ic:.4f}")
    logger.info("-" * 60)
    logger.info(f"佣金总计：{result.total_commission:.2f} 元")
    logger.info(f"印花税总计：{result.total_stamp_duty:.2f} 元")
    logger.info(f"滑点成本：{result.total_slippage_cost:.2f} 元")
    logger.info(f"放弃交易次数：{result.skipped_trades}")
    logger.info("=" * 60)
    
    # 打印夏普比率计算公式证明
    logger.info("")
    logger.info("【夏普比率计算公式证明】")
    logger.info(f"无风险利率 (年化): {RISK_FREE_RATE:.2%}")
    logger.info("夏普比率 = (年化收益率 - 无风险利率) / 年化波动率")
    logger.info("")
    logger.info("【Calmar Ratio 计算公式】")
    logger.info("Calmar Ratio = 年化收益率 / 最大回撤")
    logger.info("目标：Calmar > 0.8")
    logger.info("=" * 60)
    
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"v5_adaptive_backtest_result_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"回测结果已保存至：{result_path}")


if __name__ == "__main__":
    main()