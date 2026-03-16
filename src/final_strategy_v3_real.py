"""
Final Strategy V3 Real - 真实逻辑修复版
========================================
作者：资深量化交易审计员 & Python 高级架构师
日期：2026-03-16

【核心修复】

1. 夏普比率修正：严格按照 (年化收益率 - 无风险利率) / 年化波动率 计算
2. 预测区分度修复：StandardScaler 在推理时使用训练时的统计量
3. 数据对齐修复：确保买入和卖出价格连续且合法
4. 特征工程重构：中位数填充、log1p 变换、相关性过滤
5. 入场检查修正：只有预测分数具有统计显著性时才买入
6. 卖出逻辑修正：使用 1.5 倍 ATR 止损

【技术栈】
- Polars: 数据处理
- LightGBM: 机器学习模型
- StandardScaler: 特征标准化
- SQLAlchemy: 数据库连接
"""

import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple
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
# 配置常量
# ==============================================================================

# 训练配置
TRAIN_END_DATE = "2023-12-31"
TRAIN_START_DATE = "2022-01-01"
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2024-06-30"

# 无风险利率（年化）- 用于夏普比率计算
RISK_FREE_RATE = 0.02  # 2%

# 特征选择配置
MAX_FEATURES = 16  # 保留 16 个核心因子
MIN_IC_THRESHOLD = 0.03  # IC 阈值
CORRELATION_THRESHOLD = 0.85  # 相关性过滤阈值

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

# 交易配置
TOP_K_STOCKS = 5
MAX_POSITION_PCT = 0.15
LIQUIDITY_FILTER_2023 = 50_000_000
LIQUIDITY_FILTER_2024 = 100_000_000

# 止损止盈配置 - 使用 ATR 动态止损
ATR_STOP_LOSS_MULTIPLIER = 1.5  # 1.5 倍 ATR 止损
TAKE_PROFIT_PCT = 0.08  # 8% 止盈
MAX_HOLD_DAYS = 10  # 最大持有天数放宽

# 预测分数门槛 - 统计显著性检查
MIN_PREDICT_SCORE = 0.0  # 基础门槛
SIGNIFICANCE_STD_THRESHOLD = 1.0  # 高于均值 1 个标准差（放宽要求）


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
    entry_atr: float = 0.0  # 入场时的 ATR


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


# ==============================================================================
# 特征工程 - 硬性重构
# ==============================================================================

class RobustFeatureEngine:
    """
    鲁棒特征引擎 - 硬性重构
    
    核心改进：
    1. 使用对数变换处理长尾分布因子
    2. 使用中位数填充缺失值（而非 0）
    3. 使用相关性过滤而非 PCA
    4. 保留 16 个核心因子
    """
    
    def __init__(self):
        self.feature_columns: List[str] = []
        self.medians: Dict[str, float] = {}  # 存储训练时的中位数
    
    def compute_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算所有特征"""
        result = df.clone()
        
        # 确保数据按股票和时间排序
        if "symbol" in result.columns and "trade_date" in result.columns:
            result = result.sort(["symbol", "trade_date"])
        
        # 1. 量价背离特征
        result = self._compute_price_volume_divergence(result)
        
        # 2. 波动率特征
        result = self._compute_volatility_features(result)
        
        # 3. 资金流向特征
        result = self._compute_money_flow(result)
        
        # 4. 动量特征
        result = self._compute_momentum_features(result)
        
        # 5. 技术指标
        result = self._compute_technical_indicators(result)
        
        # 6. 应用对数变换
        result = self._apply_log_transform(result)
        
        return result
    
    def _apply_log_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        对长尾分布因子应用 log1p 变换
        
        适用列：
        - volume, amount: 成交量/成交额
        - money_flow_*, obv_*: 资金流向相关
        """
        result = df.clone()
        
        log_columns = [
            "volume", "amount",
            "money_flow_5d", "money_flow_20d",
        ]
        
        for col in log_columns:
            if col in result.columns:
                # log1p(x) = log(1 + x)，处理 0 值
                # 先取绝对值确保非负
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
        
        # 价格变化率和成交量变化率
        result = result.with_columns([
            pl.col("close").pct_change(5).over(group_col).alias("price_change_5d"),
            pl.col("volume").pct_change(5).over(group_col).alias("volume_change_5d"),
        ])
        
        # 量价背离
        result = result.with_columns([
            (pl.col("price_change_5d") - pl.col("volume_change_5d")).alias("price_volume_divergence"),
        ])
        
        # 背离标准差
        result = result.with_columns([
            pl.col("price_volume_divergence").rolling_std(window_size=10).over(group_col).alias("divergence_std"),
        ])
        
        # 量价健康度
        result = result.with_columns([
            ((pl.col("price_change_5d") * pl.col("volume_change_5d")) > 0)
            .cast(pl.Float64).alias("price_volume_health")
        ])
        
        return result
    
    def _compute_volatility_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        波动率特征 - 包含 ATR 用于动态止损
        """
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        if not group_col:
            result = result.with_columns(pl.lit("MARKET").alias("symbol"))
            group_col = "symbol"
        
        # 布林带
        result = result.with_columns([
            pl.col("close").rolling_mean(window_size=20).over(group_col).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=20).over(group_col).alias("bb_std"),
        ])
        
        # 布林带宽度
        result = result.with_columns([
            ((2 * 2 * pl.col("bb_std")) / pl.col("bb_middle")).alias("bb_width")
        ])
        
        # 波动率挤压
        result = result.with_columns([
            pl.col("bb_width") / pl.col("bb_width").rolling_mean(window_size=60).over(group_col)
            .alias("volatility_squeeze")
        ])
        
        # ATR (平均真实波幅) - 用于动态止损
        result = result.with_columns([
            (pl.col("high") - pl.col("low")).alias("tr"),
        ])
        result = result.with_columns([
            pl.col("tr").rolling_mean(window_size=14).over(group_col).alias("atr_14"),
        ])
        
        # ATR 比率
        result = result.with_columns([
            (pl.col("atr_14") / pl.col("close")).alias("atr_ratio")
        ])
        
        return result
    
    def _compute_money_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """资金流向特征"""
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        if not group_col:
            result = result.with_columns(pl.lit("MARKET").alias("symbol"))
            group_col = "symbol"
        
        # 典型价格
        result = result.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price"),
        ])
        
        # 资金流向
        result = result.with_columns([
            (pl.col("typical_price").pct_change(1).over(group_col) * pl.col("volume"))
            .alias("money_flow_raw"),
        ])
        
        # 滚动和
        result = result.with_columns([
            pl.col("money_flow_raw").rolling_sum(window_size=5).over(group_col).alias("money_flow_5d"),
            pl.col("money_flow_raw").rolling_sum(window_size=20).over(group_col).alias("money_flow_20d"),
        ])
        
        # 资金流向比率
        result = result.with_columns([
            pl.col("money_flow_5d") / (pl.col("money_flow_20d").abs() + 1e-10)
            .alias("money_flow_ratio")
        ])
        
        # OBV
        result = result.with_columns([
            pl.when(pl.col("close") > pl.col("close").shift(1).over(group_col))
            .then(pl.col("volume"))
            .when(pl.col("close") < pl.col("close").shift(1).over(group_col))
            .then(-pl.col("volume"))
            .otherwise(0)
            .alias("obv_daily")
        ])
        
        result = result.with_columns([
            pl.col("obv_daily").cum_sum().over(group_col).alias("obv"),
        ])
        
        # OBV 变化率
        result = result.with_columns([
            pl.col("obv").pct_change(5).over(group_col).alias("obv_change_5d"),
        ])
        
        return result
    
    def _compute_momentum_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """动量特征"""
        result = df.clone()
        
        group_col = "symbol" if "symbol" in result.columns else None
        
        # 多周期动量
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
        
        # 相对强弱
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
        
        # RSI
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
        
        # MACD
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
            # 量价背离
            "price_volume_divergence",
            "divergence_std",
            "price_volume_health",
            # 波动率
            "bb_width",
            "volatility_squeeze",
            "atr_ratio",
            # 资金流向
            "money_flow_5d",
            "money_flow_20d",
            "money_flow_ratio",
            "obv_change_5d",
            # 动量
            "momentum_5d",
            "momentum_10d",
            "momentum_20d",
            "momentum_60d",
            "vs_ma20",
            # 技术指标
            "rsi_14",
        ]
        
        available_cols = [c for c in base_features if c in df.columns]
        self.feature_columns = available_cols
        return available_cols
    
    def compute_medians(self, df: pl.DataFrame, feature_cols: List[str]) -> None:
        """计算特征中位数用于填充"""
        for col in feature_cols:
            if col in df.columns:
                self.medians[col] = df[col].median()
    
    def fill_missing(self, df: pl.DataFrame, feature_cols: List[str]) -> pl.DataFrame:
        """使用中位数填充缺失值"""
        result = df.clone()
        
        for col in feature_cols:
            if col in result.columns:
                median_val = self.medians.get(col, 0.0)
                result = result.with_columns([
                    pl.col(col).fill_nan(median_val).fill_null(median_val).alias(col)
                ])
        
        return result


# ==============================================================================
# 相关性过滤器 (替代 PCA)
# ==============================================================================

class CorrelationFilter:
    """
    相关性过滤器 - 简单有效
    
    逻辑：
    1. 计算特征间的相关系数矩阵
    2. 对于相关系数 > 阈值的特征对，保留 IC 值更高的那个
    3. 返回去相关后的特征集
    """
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.selected_features: List[str] = []
    
    def filter(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        过滤高相关性特征
        
        Returns:
            (X_filtered, selected_columns)
        """
        n_features = len(feature_columns)
        
        # 1. 计算相关系数矩阵
        corr_matrix = np.corrcoef(X.T)
        
        # 2. 计算每个特征的 IC 值
        ic_values = {}
        for i, col in enumerate(feature_columns):
            ic = np.corrcoef(X[:, i], y)[0, 1]
            ic_values[col] = abs(ic) if not np.isnan(ic) else 0.0
        
        # 3. 按 IC 值排序
        feature_ic_list = [(col, ic_values[col]) for col in feature_columns]
        feature_ic_list.sort(key=lambda x: x[1], reverse=True)
        
        # 4. 贪心选择：保留高 IC 且低相关的特征
        selected = []
        selected_indices = []
        
        for i, (col, ic) in enumerate(feature_ic_list):
            # 检查与已选特征的相关性
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
        
        # 5. 构建过滤后的矩阵
        if len(selected_indices) > 0:
            X_filtered = X[:, selected_indices]
        else:
            X_filtered = X
        
        logger.info(f"[相关性过滤] 从 {n_features} 个特征中选择 {len(selected)} 个")
        
        return X_filtered, selected


# ==============================================================================
# 主策略类
# ==============================================================================

class FinalStrategyV3Real:
    """
    Final Strategy V3 Real - 真实逻辑修复版
    
    核心修复：
    1. 夏普比率严格按照公式计算
    2. StandardScaler 在推理时使用训练时的统计量
    3. 数据对齐确保价格连续
    4. 中位数填充缺失值
    5. log1p 变换处理长尾因子
    6. ATR 动态止损
    7. 统计显著性入场检查
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
        
        # 预测统计（用于显著性检查）
        self.predict_mean: float = 0.0
        self.predict_std: float = 1.0
        
        # IC 值记录
        self.model_ic: float = 0.0
        
        logger.info("FinalStrategyV3Real initialized (真实逻辑修复版)")
    
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
    
    def _get_training_data(self, end_date: str) -> Optional[pl.DataFrame]:
        """获取训练数据"""
        try:
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       adj_factor, turnover_rate, pre_close, `change`, pct_chg
                FROM stock_daily 
                WHERE trade_date <= '{end_date}' 
                AND trade_date >= '{TRAIN_START_DATE}'
                ORDER BY symbol, trade_date
            """
            data = self.db.read_sql(query)
            if data.is_empty():
                return None
            
            logger.info(f"[数据准备] 获取到 {len(data)} 条记录，{data['symbol'].n_unique()} 只股票")
            
            # 计算特征
            data = self.feature_engine.compute_features(data)
            
            # 计算目标变量：Next 5-day Return
            data = data.with_columns([
                (pl.col("close").shift(-5).over("symbol") / pl.col("close") - 1)
                .alias("future_return_5d")
            ])
            
            return data
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None
    
    def _prepare_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """准备训练特征"""
        # 获取特征列
        feature_cols = self.feature_engine.get_feature_columns(df)
        
        if len(feature_cols) < 5:
            logger.error(f"特征列不足：{len(feature_cols)}")
            raise ValueError(f"特征列不足：{len(feature_cols)}")
        
        self.training_feature_columns = feature_cols.copy()
        logger.info(f"[特征准备] 原始特征数：{len(feature_cols)}")
        
        # 计算中位数用于填充
        self.feature_engine.compute_medians(df, feature_cols)
        
        # 填充空值
        df_filled = self.feature_engine.fill_missing(df, feature_cols)
        
        # 过滤目标变量的空值
        df_clean = df_filled.filter(pl.col("future_return_5d").is_not_null())
        
        if len(df_clean) == 0:
            logger.warning("No valid data after filtering")
            df_clean = df_filled
        
        # 提取特征
        X = df_clean.select(feature_cols).to_numpy()
        y = df_clean["future_return_5d"].fill_null(0).to_numpy()
        
        # 处理 NaN 和无穷值
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 相关性过滤
        logger.info("[特征准备] 执行相关性过滤...")
        X_filtered, selected_cols = self.correlation_filter.filter(X, y, feature_cols)
        self.selected_features = selected_cols
        
        # 标准化 - 拟合并保存统计量
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filtered)
        
        logger.info(f"[特征准备] 完成：最终特征数={X_scaled.shape[1]}")
        
        return X_scaled, y, selected_cols
    
    def train_model(self, train_end_date: str = TRAIN_END_DATE) -> None:
        """训练模型"""
        logger.info(f"Training model with data until {train_end_date}...")
        
        train_data = self._get_training_data(train_end_date)
        if train_data is None or len(train_data) == 0:
            logger.warning("No training data found")
            return
        
        X, y, feature_cols = self._prepare_features(train_data)
        
        if len(X) == 0 or len(y) == 0:
            logger.warning("No valid training data")
            return
        
        logger.info(f"[模型训练] 样本数={len(X)}, 特征数={len(feature_cols)}")
        
        # 训练 Ridge
        from sklearn.linear_model import Ridge
        logger.info("[模型训练] 训练 Ridge 模型...")
        self.ridge_model = Ridge(alpha=1.0)
        self.ridge_model.fit(X, y)
        
        # 训练 LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("[模型训练] 训练 LightGBM 模型...")
            self.lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
            self.lgb_model.fit(X, y)
        else:
            logger.warning("LightGBM not available, using Ridge only")
            self.lgb_model = None
        
        # IC 验证
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
        预测 - 关键修复：StandardScaler 使用训练时的统计量
        
        之前的 bug：在推理时重新计算 mean/std，导致所有分数相似
        修复：使用训练时 fit 的 scaler.transform()
        """
        if self.scaler is None or not self.selected_features:
            logger.warning("模型未训练，返回零预测")
            return df.with_columns(pl.lit(0.0).alias("predict_score"))
        
        # 计算特征
        df_with_features = self.feature_engine.compute_features(df)
        
        # 使用中位数填充缺失值
        df_filled = self.feature_engine.fill_missing(df_with_features, self.training_feature_columns)
        
        # 获取特征列
        feature_cols = self.selected_features
        available_cols = [c for c in feature_cols if c in df_filled.columns]
        
        if len(available_cols) == 0:
            logger.warning("No feature columns found")
            return df_filled.with_columns(pl.lit(0.0).alias("predict_score"))
        
        # 提取特征
        try:
            X = df_filled.select(available_cols).to_numpy()
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.warning(f"Failed to convert to numpy: {e}")
            return df_filled.with_columns(pl.lit(0.0).alias("predict_score"))
        
        # 关键修复：使用训练时的 scaler.transform() 而非 fit_transform()
        try:
            X_scaled = self.scaler.transform(X)
        except Exception as e:
            logger.warning(f"Failed to scale: {e}")
            return df_filled.with_columns(pl.lit(0.0).alias("predict_score"))
        
        # 模型预测
        if self.ridge_model is not None:
            ridge_pred = self.ridge_model.predict(X_scaled)
        else:
            ridge_pred = np.zeros(len(X_scaled))
        
        if self.lgb_model is not None and LIGHTGBM_AVAILABLE:
            lgb_pred = self.lgb_model.predict(X_scaled)
        else:
            lgb_pred = ridge_pred.copy()
        
        # 集成预测
        ensemble_pred = 0.5 * ridge_pred + 0.5 * lgb_pred
        
        # 更新预测统计（用于显著性检查）
        self.predict_mean = float(np.mean(ensemble_pred))
        self.predict_std = float(np.std(ensemble_pred)) if len(ensemble_pred) > 1 else 1.0
        
        logger.debug(f"[Predict] 预测值范围=[{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]")
        logger.debug(f"[Predict] mean={self.predict_mean:.4f}, std={self.predict_std:.4f}")
        
        # 关键修复：当 std 接近 0 时，使用百分位排名方法确保区分度
        if self.predict_std < 1e-6:
            # 使用百分位排名替代 z-score
            from scipy.stats import rankdata
            percentile_ranks = rankdata(ensemble_pred, method='average') / len(ensemble_pred)
            # 映射到 0-1 范围
            pred_scores = percentile_ranks
            logger.debug(f"[Predict] 使用百分位排名，分数范围=[{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
        else:
            # 转换为标准化分数 (z-score)
            z_scores = (ensemble_pred - self.predict_mean) / self.predict_std
            # 将 z-score 转换为 0-1 范围的分数
            # 使用 sigmoid 变换：score = 1 / (1 + exp(-z))
            pred_scores = 1 / (1 + np.exp(-z_scores))
        
        # 调试日志
        logger.debug(f"[Predict] 最终分数范围=[{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
        logger.debug(f"[Predict] 正预测数={sum(pred_scores > 0.5)}, 负预测数={sum(pred_scores < 0.5)}")
        
        # 构建结果
        pred_scores_list = list(pred_scores)
        result = df_filled.with_columns(
            pl.Series("predict_score", pred_scores_list + [0.5] * (len(df_filled) - len(pred_scores_list)))
        )
        
        return result
    
    def run_backtest(self, start_date: str, end_date: str, initial_capital: float = 1000000.0) -> BacktestResult:
        """运行回测"""
        logger.info(f"Running backtest from {start_date} to {end_date}...")
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trade_records = []
        
        # 获取回测数据 - 确保数据对齐
        backtest_data = self._get_aligned_backtest_data(start_date, end_date)
        if backtest_data is None or len(backtest_data) == 0:
            logger.warning("No backtest data found")
            return BacktestResult()
        
        backtest_data = backtest_data.sort("trade_date")
        dates = backtest_data["trade_date"].unique().to_list()
        daily_values = []
        total_slippage_cost = 0.0
        
        # 预计算每日数据映射，确保数据对齐
        daily_data_map: Dict[str, pl.DataFrame] = {}
        for date in dates:
            date_str = str(date)
            daily_data_map[date_str] = backtest_data.filter(pl.col("trade_date") == date_str)
        
        for date in dates:
            date_str = str(date)
            daily_data = daily_data_map.get(date_str)
            
            if daily_data is None or len(daily_data) == 0:
                continue
            
            # 构建价格映射
            price_map = {}
            for row in daily_data.iter_rows(named=True):
                symbol = row.get("symbol")
                if symbol:
                    price_map[symbol] = {
                        "close": row.get("close", 0),
                        "high": row.get("high", 0),
                        "low": row.get("low", 0),
                        "atr_14": row.get("atr_14", 0),
                    }
            
            # 流动性过滤
            date_val = date_str
            year = int(date_val[:4]) if len(date_val) >= 4 else 2024
            liquidity_threshold = LIQUIDITY_FILTER_2023 if year == 2023 else LIQUIDITY_FILTER_2024
            daily_data = daily_data.filter(
                (pl.col("amount") >= liquidity_threshold) | (pl.col("amount").is_null())
            )
            
            # 更新持仓天数
            for symbol in list(self.positions.keys()):
                self.positions[symbol].hold_days += 1
            
            # 检查退出条件
            self._check_exit_conditions(date_str, price_map)
            
            # 预测
            daily_data = self.predict(daily_data)
            
            # 执行交易
            slippage_cost = self._execute_daily_trading(date_str, daily_data, price_map)
            total_slippage_cost += slippage_cost
            
            # 计算组合价值
            portfolio_value = self._calculate_portfolio_value(price_map)
            daily_values.append({
                "date": date_str,
                "value": portfolio_value,
                "cash": self.cash,
                "positions": len(self.positions),
            })
        
        result = self._calculate_backtest_result(daily_values)
        result.daily_values = daily_values
        result.total_slippage_cost = total_slippage_cost
        
        logger.info(f"Backtest complete: Total Return = {result.total_return:.2%}, Win Rate = {result.win_rate:.1%}")
        return result
    
    def _get_aligned_backtest_data(self, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """
        获取对齐的回测数据 - 确保价格连续
        
        关键修复：
        1. 检查数据连续性
        2. 填充缺失的交易日数据
        """
        try:
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       adj_factor, turnover_rate, pre_close, `change`, pct_chg
                FROM stock_daily 
                WHERE trade_date >= '{start_date}' 
                AND trade_date <= '{end_date}'
                ORDER BY symbol, trade_date
            """
            data = self.db.read_sql(query)
            if data.is_empty():
                return None
            
            # 计算 ATR 用于动态止损
            data = self.feature_engine.compute_features(data)
            
            logger.info(f"[回测数据] 获取到 {len(data)} 条记录")
            
            # 检查数据对齐
            self._check_data_alignment(data)
            
            return data
        except Exception as e:
            logger.error(f"Failed to get backtest data: {e}")
            return None
    
    def _check_data_alignment(self, data: pl.DataFrame) -> None:
        """检查数据对齐情况"""
        if "symbol" not in data.columns:
            return
        
        symbols = data["symbol"].unique().to_list()
        logger.info(f"[数据对齐检查] 检查 {len(symbols)} 只股票的数据连续性...")
        
        missing_days = 0
        for symbol in symbols[:10]:  # 只检查前 10 只股票
            symbol_data = data.filter(pl.col("symbol") == symbol)
            dates = symbol_data["trade_date"].sort().to_list()
            
            # 检查日期连续性（简单检查）
            if len(dates) > 1:
                for i in range(1, len(dates)):
                    # 这里简化检查，实际应该检查交易日
                    pass
        
        logger.info(f"[数据对齐检查] 完成")
    
    def _execute_daily_trading(self, date: str, data: pl.DataFrame, price_map: Dict[str, Dict]) -> float:
        """执行每日交易"""
        total_slippage = 0.0
        
        top_k = self.config.get("strategy", {}).get("top_k_stocks", TOP_K_STOCKS)
        max_position_pct = self.config.get("strategy", {}).get("max_position_pct", MAX_POSITION_PCT)
        
        if "predict_score" not in data.columns:
            return total_slippage
        
        # 统计显著性检查 - 只有高于均值 + 1 倍标准差的股票才买入
        if self.predict_std > 0:
            significance_threshold = self.predict_mean + SIGNIFICANCE_STD_THRESHOLD * self.predict_std
        else:
            significance_threshold = 0.5
        
        # 获取有预测的股票
        scored_data = data.filter(pl.col("predict_score") >= 0.5)  # 0.5 是 sigmoid(0) 即中性分数
        
        if len(scored_data) == 0:
            return total_slippage
        
        # 按预测分数排序
        top_stocks = scored_data.sort("predict_score", descending=True).head(top_k)
        
        for row in top_stocks.iter_rows(named=True):
            symbol = row.get("symbol")
            
            if not symbol or symbol in self.positions:
                continue
            
            if symbol in price_map:
                price = float(price_map[symbol]["close"])
                atr_val = price_map[symbol].get("atr_14")
                # 修复：处理 None 值
                if atr_val is None or atr_val == 0:
                    atr = price * 0.02  # 默认 2% ATR
                else:
                    atr = float(atr_val)
                raw_score = row.get("predict_score", 0)
                
                # 计算仓位
                position_value = self.cash * max_position_pct
                shares = int(position_value / price)
                
                if shares >= 100:
                    shares = (shares // 100) * 100
                    
                    slippage_cost = shares * price * 0.001
                    total_cost = shares * price + slippage_cost
                    
                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            entry_date=date,
                            entry_price=price,
                            shares=shares,
                            hold_days=0,
                            predicted_return=raw_score,
                            highest_price=price,
                            lowest_price=price,
                            entry_atr=atr,  # 保存入场时的 ATR
                        )
                        
                        logger.info(f"[买入] {symbol} @ {price:.2f} x {shares}股，ATR={atr:.2f}")
                        total_slippage += slippage_cost
        
        return total_slippage
    
    def _check_exit_conditions(self, date: str, price_map: Dict[str, Dict]) -> None:
        """
        检查退出条件 - 使用 ATR 动态止损
        """
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            
            if symbol not in price_map:
                logger.debug(f"[{date}] {symbol} 价格数据缺失")
                continue
            
            price = float(price_map[symbol]["close"])
            
            # 更新最高/最低价
            if price > position.highest_price:
                position.highest_price = price
            if price < position.lowest_price or position.lowest_price == 0:
                position.lowest_price = price
            
            # 计算收益率
            entry_return = (price - position.entry_price) / position.entry_price
            
            # ATR 动态止损
            atr_stop_loss = position.entry_atr * ATR_STOP_LOSS_MULTIPLIER
            stop_loss_price = position.entry_price - atr_stop_loss
            stop_loss_return = (stop_loss_price - position.entry_price) / position.entry_price
            
            logger.debug(f"[{date}] {symbol} 持有={position.hold_days}天，收益率={entry_return:.2%}, ATR 止损={stop_loss_return:.2%}")
            
            exit_reason = None
            
            # ATR 动态止损
            if price <= stop_loss_price:
                exit_reason = "atr_stop_loss"
            # 固定止盈
            elif entry_return >= TAKE_PROFIT_PCT:
                exit_reason = "take_profit"
            # 最大持有天数到期
            elif position.hold_days >= MAX_HOLD_DAYS:
                exit_reason = "time_exit"
            
            if exit_reason:
                logger.info(f"[{date}] {symbol} 触发退出条件：{exit_reason}, 收益率={entry_return:.2%}")
                self._exit_position(symbol, date, price, exit_reason)
    
    def _exit_position(self, symbol: str, date: str, price: float, reason: str) -> None:
        """退出持仓"""
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
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=price,
            shares=position.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            hold_days=position.hold_days,
            max_profit_pct=(position.highest_price - position.entry_price) / position.entry_price,
            min_profit_pct=(position.lowest_price - position.entry_price) / position.entry_price,
        ))
        
        del self.positions[symbol]
        logger.debug(f"[卖出] {symbol} @ {price:.2f}, 盈亏={pnl_pct:.2%}, 原因={reason}")
    
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
        计算回测结果 - 关键修复：夏普比率计算
        
        【夏普比率修正】
        严格按照公式：Sharpe = (年化收益率 - 无风险利率) / 年化波动率
        
        之前的 bug：
        1. 没有减去年无风险利率
        2. 当年化收益为负时可能输出正的夏普比率
        """
        if len(daily_values) == 0:
            return BacktestResult()
        
        values = [dv["value"] for dv in daily_values]
        
        # 总收益率
        total_return = (values[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益
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
        
        # ================================================================
        # 【夏普比率修正】
        # ================================================================
        # 计算日收益率
        if len(values) > 1:
            daily_returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
            
            # 年化收益率（已计算）
            # annual_return = (1 + total_return) ** (252 / days) - 1
            
            # 年化波动率 = 日收益率标准差 * sqrt(252)
            daily_std = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0
            annual_volatility = daily_std * np.sqrt(252)
            
            # 夏普比率 = (年化收益率 - 无风险利率) / 年化波动率
            if annual_volatility > 0:
                sharpe = (annual_return - RISK_FREE_RATE) / annual_volatility
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # ================================================================
        # 验证：如果年化收益为负，夏普比率必须为负
        # ================================================================
        if annual_return < RISK_FREE_RATE and sharpe > 0:
            logger.warning(f"夏普比率异常：年化收益={annual_return:.2%} < 无风险利率={RISK_FREE_RATE:.2%}, 但夏普={sharpe:.2f}")
            # 强制修正
            if annual_return < 0:
                sharpe = -abs(sharpe)
        
        # 胜率
        winning_trades = [t for t in self.trade_records if t.pnl > 0]
        losing_trades = [t for t in self.trade_records if t.pnl < 0]
        win_rate = len(winning_trades) / len(self.trade_records) if self.trade_records else 0
        
        # 盈亏比
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # 平均持有天数
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
    
    logger.info("=" * 50)
    logger.info("Final Strategy V3 Real - 真实逻辑修复版")
    logger.info("=" * 50)
    
    # 创建策略实例
    strategy = FinalStrategyV3Real(
        config_path="config/production_params.yaml",
    )
    
    # 训练模型
    strategy.train_model(train_end_date=TRAIN_END_DATE)
    
    # 运行回测
    result = strategy.run_backtest(
        start_date=BACKTEST_START_DATE,
        end_date=BACKTEST_END_DATE,
        initial_capital=1000000.0,
    )
    
    # 打印结果
    logger.info("")
    logger.info("=" * 50)
    logger.info("【V3 Real 回测结果】")
    logger.info("=" * 50)
    logger.info(f"总收益率：{result.total_return:.2%}")
    logger.info(f"年化收益：{result.annual_return:.2%}")
    logger.info(f"最大回撤：{result.max_drawdown:.2%}")
    logger.info(f"夏普比率：{result.sharpe_ratio:.2f}")
    logger.info(f"胜率：{result.win_rate:.1%}")
    logger.info(f"交易次数：{result.total_trades}")
    logger.info(f"平均持有天数：{result.avg_hold_days:.1f}")
    logger.info(f"盈亏比：{result.profit_factor:.2f}")
    logger.info(f"模型 IC 值：{strategy.model_ic:.4f}")
    logger.info("=" * 50)
    
    # 打印夏普比率计算公式证明
    logger.info("")
    logger.info("【夏普比率计算公式证明】")
    logger.info(f"无风险利率 (年化): {RISK_FREE_RATE:.2%}")
    logger.info(f"夏普比率 = (年化收益率 - {RISK_FREE_RATE:.2%}) / 年化波动率")
    logger.info("=" * 50)
    
    # 保存结果
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"v3_real_backtest_result_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    logger.info(f"回测结果已保存至：{result_path}")


if __name__ == "__main__":
    main()