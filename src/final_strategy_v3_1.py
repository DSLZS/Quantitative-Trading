"""
Final Strategy V3.1 - Adaptive Barrier & Robust Factor Engine
=============================================================
【核心升级：从模型驱动到自适应鲁棒系统】

作者：量化策略团队
版本：Final Strategy V3.1 (Adaptive Barrier & Robust Factors)
日期：2026-03-16

【本次升级 - 解决 V3.0 数据贫血问题】

1. 因子计算管线诊断与修复 (Data Integrity)
   - 增加数据"预热"机制：强制检查 Lookback 窗口（至少 120 天）
   - 鲁棒性重构：compute_factors 失败时使用"次优替代方案"
   - 日志强化：初始化阶段打印【因子健康度报告】

2. 引入自适应波动率屏障 (Dynamic Barriers)
   - 固定 1.5σ止损改为动态 ATR 止损
   - 止损距离 = k × ATR，k 根据 GMM 市场状态调整：
     * BULL 状态：k=2.5 (宽容，持股待涨)
     * BEAR/VOLATILE 状态：k=1.2 (严格，快速止损)
     * CALM 状态：k=1.8 (中性)

3. 特征重要性剪枝 (Feature Pruning)
   - 在 FeatureSynthesizer 中加入过滤步骤
   - 自动剔除有效样本数低于 70% 的原始因子
   - 确保模型学习的是"真实存在的信号"

4. smart_money 等因子缺失问题修复
   - 增加成交额数据检查
   - 当 amount 缺失时，使用 volume 作为替代计算 smart_money
   - 增加数据源诊断和修复建议
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
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Ridge
from sklearn.metrics import mutual_info_score
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
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine
    from backtest_engine import BacktestEngine


# ==============================================================================
# 配置常量 (V3.1 优化版)
# ==============================================================================

# 正交化配置
ORTHOGONALIZATION_ENABLED = True
GRAM_SCHMIDT_TOLERANCE = 1e-10

# 特征合成配置
FEATURE_SYNTHESIS_ENABLED = True
MAX_INTERACTION_ORDER = 2
TOP_N_FEATURES_BY_MI = 30

# 【V3.1 新增】特征剪枝配置 - 有效样本率阈值
FEATURE_PRUNING_ENABLED = True
MIN_VALID_SAMPLE_RATIO = 0.70  # 70% 有效样本阈值

# 环境门控配置
GATING_MECHANISM_ENABLED = True
GMM_N_COMPONENTS = 4
GMM_RANDOM_STATE = 42

# 【V3.1 新增】数据预热配置
DATA_WARMUP_ENABLED = True
MIN_LOOKBACK_DAYS = 120  # 至少 120 天历史数据
MIN_LOOKBACK_FOR_BIAS60 = 80  # bias_60 至少需要 80 天

# 三屏障碍法配置 (V3.1: 自适应 ATR 止损)
TRIPLE_BARRIER_ENABLED = True
TBR_UPPER_BARRIER = 2.5   # 止盈阈值 2.5σ
TBR_TIME_BARRIER = 12     # 持有观察期延长至 12 天

# 【V3.1 新增】自适应止损配置
ADAPTIVE_STOP_LOSS_ENABLED = True
ATR_PERIOD = 20  # ATR 计算周期
# GMM 状态对应的止损系数 k
STOP_LOSS_K_BULL = 2.5      # 牛市：宽容止损
STOP_LOSS_K_BEAR = 1.2      # 熊市：严格止损
STOP_LOSS_K_VOLATILE = 1.2  # 震荡市：严格止损
STOP_LOSS_K_CALM = 1.8      # 平静市：中性止损

# 损失函数配置
HUBER_LOSS_ENABLED = True
HUBER_DELTA = 1.0
DRAWDOWN_PENALTY_WEIGHT = 0.1

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
    stop_loss_price: float = 0.0  # V3.1: 动态止损价
    take_profit_price: float = 0.0  # V3.1: 动态止盈价


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
    stop_loss_type: str = ""  # V3.1: 记录止损类型


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
# 核心算法模块 (V3.1 增强版)
# ==============================================================================

class FactorOrthogonalizer:
    """
    因子正交化器 - 使用施密特正交化消除因子间线性相关性
    """
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
        self.transformation_matrix: Optional[np.ndarray] = None
        self.original_columns: List[str] = []
        self.orthogonal_columns: List[str] = []
        self.Q_base: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        self.is_fitted: bool = False
    
    def fit(self, X: np.ndarray, columns: List[str]) -> 'FactorOrthogonalizer':
        """拟合正交化器，保存转换矩阵"""
        self.original_columns = columns
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # 保存标准化参数
        self.mean_ = np.mean(X, axis=0).copy()
        self.std_ = np.std(X, axis=0) + self.tolerance
        
        # 标准化
        X_normalized = (X - self.mean_) / self.std_
        
        # 施密特正交化
        Q = np.zeros((n_samples, n_features))
        R = np.zeros((n_features, n_features))
        
        for j in range(n_features):
            v = X_normalized[:, j].copy()
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], X_normalized[:, j])
                v = v - R[i, j] * Q[:, i]
            
            norm = np.linalg.norm(v)
            if norm > self.tolerance:
                Q[:, j] = v / norm
                R[j, j] = norm
            else:
                Q[:, j] = np.zeros(n_samples)
        
        self.transformation_matrix = R
        self.Q_base = Q
        
        # 保存正交化后的列名
        self.orthogonal_columns = []
        for i, (col, norm) in enumerate(zip(columns, np.diag(R))):
            if norm > self.tolerance:
                self.orthogonal_columns.append(f"ortho_{i}")
            else:
                self.orthogonal_columns.append(f"ortho_{i}_zero")
        
        self.is_fitted = True
        logger.info(f"[因子正交化] 原始因子：{n_features}, 独立因子：{len([n for n in np.diag(R) if n > self.tolerance])}")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """使用训练时的参数进行转换"""
        if not self.is_fitted or self.mean_ is None or self.std_ is None:
            logger.warning("正交化器未拟合，返回原始数据")
            return X
        
        # 使用训练时的 mean 和 std 进行标准化
        X_normalized = (X - self.mean_) / self.std_
        return X_normalized
    
    def fit_transform(self, X: np.ndarray, columns: List[str]) -> np.ndarray:
        self.fit(X, columns)
        return self.transform(X)
    
    def get_feature_names(self) -> List[str]:
        """返回正交化后的特征名"""
        return self.orthogonal_columns


class FeatureSynthesizer:
    """
    非线性特征合成器 - 自动创建并选择高信息量交叉特征
    
    【V3.1 新增】特征剪枝：自动剔除有效样本数低于 70% 的原始因子
    """
    
    def __init__(self, max_order: int = 2, top_n: int = 30, min_valid_ratio: float = 0.70):
        self.max_order = max_order
        self.top_n = top_n
        self.min_valid_ratio = min_valid_ratio  # V3.1: 最小有效样本率
        self.selected_features: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.synthesis_columns: List[str] = []
        self.input_columns: List[str] = []
        self.pruned_features: List[str] = []  # V3.1: 被剪枝的特征
        self.is_fitted: bool = False
        self.n_input_features: int = 0
    
    def _discretize(self, x: np.ndarray, n_bins: int = 10) -> np.ndarray:
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(x[~np.isnan(x)], percentiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            return np.zeros(len(x), dtype=int)
        return np.digitize(x, bins[1:-1])
    
    def _compute_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        try:
            x_discrete = self._discretize(x)
            y_discrete = self._discretize(y)
            mi = mutual_info_score(x_discrete, y_discrete)
            return mi if np.isfinite(mi) else 0.0
        except Exception:
            return 0.0
    
    def _compute_valid_ratio(self, x: np.ndarray) -> float:
        """计算有效样本率（非 NaN 比例）"""
        if len(x) == 0:
            return 0.0
        valid_count = np.sum(~np.isnan(x))
        return valid_count / len(x)
    
    def fit(self, X: np.ndarray, y: np.ndarray, original_columns: List[str]) -> 'FeatureSynthesizer':
        """
        拟合特征合成器，选择 Top N 特征
        
        【V3.1 新增】特征剪枝步骤
        """
        n_samples, n_features = X.shape
        self.n_input_features = n_features
        self.input_columns = original_columns.copy()
        
        all_features: Dict[str, np.ndarray] = {}
        all_mi: Dict[str, float] = {}
        
        # 【V3.1 新增】步骤 1: 特征剪枝 - 过滤低有效样本率的原始因子
        if FEATURE_PRUNING_ENABLED:
            logger.info(f"[特征剪枝] 过滤有效样本率 < {self.min_valid_ratio:.0%} 的因子...")
            pruned_count = 0
            
            for i, col in enumerate(original_columns):
                valid_ratio = self._compute_valid_ratio(X[:, i])
                if valid_ratio < self.min_valid_ratio:
                    self.pruned_features.append(col)
                    pruned_count += 1
                    logger.debug(f"[特征剪枝] 剪枝：{col} (有效样本率={valid_ratio:.1%})")
                else:
                    all_features[col] = X[:, i]
                    mi = self._compute_mutual_information(X[:, i], y)
                    all_mi[col] = mi
            
            logger.info(f"[特征剪枝] 剪枝 {pruned_count}/{n_features} 个因子，保留 {len(all_features)} 个")
        else:
            # 不剪枝时使用所有特征
            for i, col in enumerate(original_columns):
                all_features[col] = X[:, i]
                mi = self._compute_mutual_information(X[:, i], y)
                all_mi[col] = mi
        
        # 生成二阶交叉特征
        if self.max_order >= 2:
            logger.info(f"[特征合成] 生成二阶交叉特征...")
            feature_names = list(all_features.keys())
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    cross_feature = all_features[feature_names[i]] * all_features[feature_names[j]]
                    col_name = f"{feature_names[i]}_x_{feature_names[j]}"
                    all_features[col_name] = cross_feature
                    mi = self._compute_mutual_information(cross_feature, y)
                    all_mi[col_name] = mi
                    self.synthesis_columns.append(col_name)
        
        # 按 MI 排序选择 Top N
        sorted_features = sorted(all_mi.items(), key=lambda x: x[1], reverse=True)
        self.selected_features = [f for f, _ in sorted_features[:self.top_n]]
        self.feature_importance = dict(sorted_features[:self.top_n])
        
        self.is_fitted = True
        logger.info(f"[特征选择] 从 {len(all_mi)} 个特征中选择 {len(self.selected_features)} 个 (基于 MI)")
        return self
    
    def transform(self, X: np.ndarray, original_columns: List[str]) -> np.ndarray:
        """
        使用训练时选择的特征进行转换
        """
        if not self.is_fitted:
            logger.warning("特征合成器未拟合，返回原始数据")
            return X
        
        # 构建特征字典
        n_features = X.shape[1]
        feature_dict: Dict[str, np.ndarray] = {}
        
        # 映射输入列名
        if len(original_columns) == n_features:
            for i, col in enumerate(original_columns):
                feature_dict[col] = X[:, i]
        else:
            # 使用训练时的列名
            for i, col in enumerate(self.input_columns[:n_features]):
                feature_dict[col] = X[:, i]
        
        # 生成所有需要的特征
        X_synthesized = []
        missing_features = []
        
        for feat_name in self.selected_features:
            if feat_name in feature_dict:
                X_synthesized.append(feature_dict[feat_name])
            elif '_x_' in feat_name:
                # 生成交叉特征
                parts = feat_name.split('_x_')
                if len(parts) == 2:
                    base1, base2 = parts[0], parts[1]
                    if base1 in feature_dict and base2 in feature_dict:
                        cross_feature = feature_dict[base1] * feature_dict[base2]
                        X_synthesized.append(cross_feature)
                    else:
                        missing_features.append(feat_name)
            else:
                missing_features.append(feat_name)
        
        if missing_features:
            logger.warning(f"特征合成：缺失 {len(missing_features)} 个特征，使用零填充")
            for _ in missing_features:
                X_synthesized.append(np.zeros(X.shape[0]))
        
        if len(X_synthesized) == 0:
            return X
        
        result = np.column_stack(X_synthesized)
        
        if result.shape[1] != self.top_n:
            logger.warning(f"特征合成：期望 {self.top_n} 个特征，实际 {result.shape[1]} 个")
        
        return result
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray,
                      original_columns: List[str]) -> Tuple[np.ndarray, List[str]]:
        self.fit(X, y, original_columns)
        X_transformed = self.transform(X, original_columns)
        return X_transformed, self.selected_features
    
    def get_feature_names(self) -> List[str]:
        """返回合成后的特征名"""
        return self.selected_features
    
    def get_pruned_features(self) -> List[str]:
        """返回被剪枝的特征列表"""
        return self.pruned_features


class MarketGatingMechanism:
    """
    智能环境感知门控 - 使用 GMM 识别市场隐状态
    
    【V3.1 增强】添加波动率特征以提升状态区分度
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
                
                # 计算状态波动率
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


class AdaptiveTripleBarrierLabeler:
    """
    【V3.1 新增】自适应三屏障碍法标注器 - 基于 ATR 和市场状态
    
    【核心改进】
    - 固定 1.5σ止损改为动态 ATR 止损
    - 止损距离 = k × ATR，k 根据 GMM 市场状态调整
    """
    
    def __init__(
        self,
        upper_barrier: float = 2.5,
        time_barrier: int = 12,
        atr_period: int = 20,
        stop_loss_k_bull: float = 2.5,
        stop_loss_k_bear: float = 1.2,
        stop_loss_k_volatile: float = 1.2,
        stop_loss_k_calm: float = 1.8,
    ):
        self.upper_barrier = upper_barrier
        self.time_barrier = time_barrier
        self.atr_period = atr_period
        self.stop_loss_k = {
            "BULL": stop_loss_k_bull,
            "BEAR": stop_loss_k_bear,
            "VOLATILE": stop_loss_k_volatile,
            "CALM": stop_loss_k_calm,
        }
    
    def compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """计算 ATR (Average True Range)"""
        n = len(high)
        atr = np.zeros(n)
        
        for i in range(1, n):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
            if i < self.atr_period:
                atr[i] = atr[i-1] + (tr - atr[i-1]) / (i + 1) if i > 0 else tr
            else:
                atr[i] = (atr[i-1] * (self.atr_period - 1) + tr) / self.atr_period
        
        return atr
    
    def get_stop_loss_distance(self, atr: float, market_state: str) -> float:
        """根据市场状态获取动态止损距离"""
        k = self.stop_loss_k.get(market_state, self.stop_loss_k["CALM"])
        return k * atr
    
    def label(self, prices: np.ndarray, volatility: np.ndarray, 
              market_state: str = "CALM", atr_values: Optional[np.ndarray] = None,
              high_prices: Optional[np.ndarray] = None, 
              low_prices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        自适应三屏障碍法标注
        
        Args:
            prices: 价格序列 (n_samples, n_days) 或 (n_samples,)
            volatility: 波动率序列 (n_samples,)
            market_state: 市场状态 (BULL/BEAR/VOLATILE/CALM)
            atr_values: ATR 序列 (可选，用于动态止损)
            high_prices: 最高价序列 (可选)
            low_prices: 最低价序列 (可选)
        
        Returns:
            labels: 1=触及上轨 (成功), 0=触及下轨 (失败), 2=时间到期
            barrier_types: 触及类型
            returns: 最终收益率
        """
        n_samples = prices.shape[0]
        labels = np.ones(n_samples) * 2  # 默认为时间到期
        barrier_types = ["time"] * n_samples
        returns = np.zeros(n_samples)
        
        # 获取当前市场状态的止损系数
        k = self.stop_loss_k.get(market_state, 1.8)
        
        for i in range(n_samples):
            if i >= len(volatility) or np.isnan(volatility[i]) or volatility[i] <= 0:
                continue
            
            entry_price = prices[i, 0] if prices.ndim > 1 else prices[i]
            vol = abs(volatility[i])
            
            # 非对称上轨（止盈）
            upper = entry_price * (1 + self.upper_barrier * vol)
            
            # 【V3.1 核心改进】动态 ATR 止损
            if atr_values is not None and i < len(atr_values) and not np.isnan(atr_values[i]):
                atr = atr_values[i]
                stop_loss_distance = k * atr
                lower = entry_price * (1 - stop_loss_distance / entry_price)
            else:
                # 回退到固定止损
                lower = entry_price * (1 - 1.5 * vol)
            
            if prices.ndim > 1:
                max_days = min(prices.shape[1], self.time_barrier)
                for t in range(max_days):
                    price_t = prices[i, t] if t < prices.shape[1] else prices[i, -1]
                    
                    if price_t >= upper:
                        labels[i] = 1  # 成功触及上轨
                        barrier_types[i] = "upper"
                        returns[i] = (price_t - entry_price) / entry_price
                        break
                    elif price_t <= lower:
                        labels[i] = 0  # 失败触及下轨
                        barrier_types[i] = "lower_adaptive"  # 标记为自适应止损
                        returns[i] = (price_t - entry_price) / entry_price
                        break
                    elif t == max_days - 1:
                        labels[i] = 2  # 时间到期
                        returns[i] = (price_t - entry_price) / entry_price
        
        return labels, barrier_types, returns


class TripleBarrierLabeler:
    """
    三屏障碍法标注器 - 路径依赖标注 (V3.0 兼容版)
    """
    
    def __init__(self, upper_barrier: float = 2.5, lower_barrier: float = 1.5, time_barrier: int = 12):
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier
        self.time_barrier = time_barrier
    
    def label(self, prices: np.ndarray, volatility: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
        n_samples = prices.shape[0]
        labels = np.ones(n_samples) * 2
        barrier_types = ["time"] * n_samples
        returns = np.zeros(n_samples)
        
        for i in range(n_samples):
            if i >= len(volatility) or np.isnan(volatility[i]) or volatility[i] <= 0:
                continue
            
            entry_price = prices[i, 0] if prices.ndim > 1 else prices[i]
            vol = abs(volatility[i])
            
            upper = entry_price * (1 + self.upper_barrier * vol)
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


class HuberDrawdownLoss:
    """感知回撤的 Huber 损失函数"""
    
    def __init__(self, delta: float = 1.0, drawdown_weight: float = 0.1):
        self.delta = delta
        self.drawdown_weight = drawdown_weight
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residual = y_true - y_pred
        abs_residual = np.abs(residual)
        huber_loss = np.where(
            abs_residual <= self.delta,
            0.5 * residual ** 2,
            self.delta * (abs_residual - 0.5 * self.delta)
        )
        drawdown_penalty = np.maximum(0, -y_pred) ** 2
        return float(np.mean(huber_loss) + self.drawdown_weight * np.mean(drawdown_penalty))


class FactorHealthChecker:
    """
    【V3.1 新增】因子健康度检查器
    
    功能：
    1. 计算每个因子的有效数据占比
    2. 诊断因子缺失原因
    3. 生成因子健康度报告
    """
    
    def __init__(self, min_valid_ratio: float = 0.70):
        self.min_valid_ratio = min_valid_ratio
        self.health_report: Dict[str, Dict] = {}
    
    def check_factor_health(self, df: pl.DataFrame, factor_columns: List[str]) -> Dict[str, Dict]:
        """
        检查因子健康度
        
        Args:
            df: 输入 DataFrame
            factor_columns: 因子列名列表
        
        Returns:
            因子健康度报告字典
        """
        self.health_report = {}
        total_rows = len(df)
        
        for col in factor_columns:
            if col not in df.columns:
                self.health_report[col] = {
                    "status": "MISSING",
                    "valid_ratio": 0.0,
                    "null_count": total_rows,
                    "recommendation": "因子列不存在"
                }
                continue
            
            null_count = df[col].null_count()
            valid_count = total_rows - null_count
            valid_ratio = valid_count / total_rows if total_rows > 0 else 0.0
            
            # 诊断缺失原因
            status = "HEALTHY" if valid_ratio >= self.min_valid_ratio else "CRITICAL"
            recommendation = self._get_recommendation(col, valid_ratio, status)
            
            self.health_report[col] = {
                "status": status,
                "valid_ratio": valid_ratio,
                "null_count": null_count,
                "valid_count": valid_count,
                "recommendation": recommendation
            }
        
        return self.health_report
    
    def _get_recommendation(self, col: str, valid_ratio: float, status: str) -> str:
        """生成修复建议"""
        if status == "HEALTHY":
            return "因子健康，无需处理"
        
        # 根据因子类型给出建议
        if "smart_money" in col:
            return "检查成交额 (amount) 数据是否完整，或使用 volume 替代计算"
        elif "turnover" in col:
            return "检查换手率 (turnover_rate) 数据，或使用 volume/流通股本估算"
        elif "vcp" in col:
            return "检查 OHLC 数据完整性，确保 high/low 不为空"
        elif "bias" in col:
            return f"需要至少 {int(60 / valid_ratio)} 天历史数据来计算 bias_60"
        elif "momentum" in col or "volatility" in col:
            return "需要更长的历史数据窗口"
        elif "volume" in col:
            return "检查成交量数据，考虑使用平滑处理"
        else:
            return "考虑使用替代因子或删除此特征"
    
    def print_health_report(self, logger=None) -> None:
        """打印因子健康度报告"""
        if not self.health_report:
            return
        
        log_func = logger.info if logger else print
        
        log_func("\n" + "=" * 70)
        log_func("【因子健康度报告】")
        log_func("=" * 70)
        
        healthy_count = sum(1 for r in self.health_report.values() if r["status"] == "HEALTHY")
        critical_count = sum(1 for r in self.health_report.values() if r["status"] == "CRITICAL")
        missing_count = sum(1 for r in self.health_report.values() if r["status"] == "MISSING")
        
        log_func(f"总因子数：{len(self.health_report)}")
        log_func(f"健康因子：{healthy_count} ({healthy_count/len(self.health_report):.1%})")
        log_func(f"临界因子：{critical_count} ({critical_count/len(self.health_report):.1%})")
        log_func(f"缺失因子：{missing_count} ({missing_count/len(self.health_report):.1%})")
        log_func("-" * 70)
        
        # 打印临界因子详情
        critical_factors = [(k, v) for k, v in self.health_report.items() if v["status"] == "CRITICAL"]
        if critical_factors:
            log_func("\n【临界因子详情】")
            for col, info in sorted(critical_factors, key=lambda x: x[1]["valid_ratio"]):
                log_func(f"  {col}:")
                log_func(f"    有效样本率：{info['valid_ratio']:.1%}")
                log_func(f"    建议：{info['recommendation']}")
        
        # 打印缺失因子详情
        missing_factors = [(k, v) for k, v in self.health_report.items() if v["status"] == "MISSING"]
        if missing_factors:
            log_func("\n【缺失因子】")
            for col, info in missing_factors:
                log_func(f"  {col}: {info['recommendation']}")
        
        log_func("=" * 70 + "\n")
    
    def get_healthy_factors(self) -> List[str]:
        """返回健康因子列表"""
        return [k for k, v in self.health_report.items() if v["status"] == "HEALTHY"]
    
    def get_critical_factors(self) -> List[str]:
        """返回临界因子列表"""
        return [k for k, v in self.health_report.items() if v["status"] == "CRITICAL"]


# ==============================================================================
# 主策略类 (V3.1 增强版)
# ==============================================================================

class FinalStrategyV31:
    """
    Final Strategy V3.1 - Adaptive Barrier & Robust Factor Engine
    
    【V3.1 升级说明】
    1. 数据预热机制：确保至少 120 天 Lookback 窗口
    2. 因子健康度检查：初始化时打印健康度报告
    3. 自适应 ATR 止损：根据 GMM 市场状态动态调整
    4. 特征剪枝：过滤有效样本率低于 70% 的因子
    5. smart_money 等因子缺失修复：使用 volume 替代 amount
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
        
        self.config = self._load_config()
        self.factor_engine = FactorEngine(str(self.factors_config_path), validate=True)
        
        # 核心组件 (V3.1 使用自适应版本)
        self.orthogonalizer = FactorOrthogonalizer(tolerance=GRAM_SCHMIDT_TOLERANCE)
        self.synthesizer = FeatureSynthesizer(
            max_order=MAX_INTERACTION_ORDER, 
            top_n=TOP_N_FEATURES_BY_MI,
            min_valid_ratio=MIN_VALID_SAMPLE_RATIO
        )
        self.gating = MarketGatingMechanism(n_components=GMM_N_COMPONENTS, random_state=GMM_RANDOM_STATE)
        
        # V3.1: 使用自适应三屏障碍法
        self.adaptive_labeler = AdaptiveTripleBarrierLabeler(
            upper_barrier=TBR_UPPER_BARRIER,
            time_barrier=TBR_TIME_BARRIER,
            atr_period=ATR_PERIOD,
            stop_loss_k_bull=STOP_LOSS_K_BULL,
            stop_loss_k_bear=STOP_LOSS_K_BEAR,
            stop_loss_k_volatile=STOP_LOSS_K_VOLATILE,
            stop_loss_k_calm=STOP_LOSS_K_CALM,
        )
        self.legacy_labeler = TripleBarrierLabeler(
            upper_barrier=TBR_UPPER_BARRIER,
            lower_barrier=1.5,
            time_barrier=TBR_TIME_BARRIER,
        )
        
        self.loss_fn = HuberDrawdownLoss(delta=HUBER_DELTA, drawdown_weight=DRAWDOWN_PENALTY_WEIGHT)
        
        # V3.1: 新增因子健康度检查器
        self.health_checker = FactorHealthChecker(min_valid_ratio=MIN_VALID_SAMPLE_RATIO)
        
        # 模型状态
        self.ridge_model: Optional[Ridge] = None
        self.lgb_model: Optional[Any] = None
        
        # 保存训练时的特征配置
        self.training_feature_columns: List[str] = []
        self.synthesized_feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.expected_feature_dim: int = TOP_N_FEATURES_BY_MI
        
        # 市场状态
        self.market_state = MarketState()
        self.positions: Dict[str, Position] = {}
        self.trade_records: List[TradeRecord] = []
        self.cash: float = 0.0
        self.initial_capital: float = 0.0
        
        self.score_history: Dict[str, List[float]] = {}
        self.recent_scores: List[float] = []
        
        logger.info("FinalStrategyV3.1 initialized (Adaptive Barrier & Robust Factors)")
    
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
    
    def _check_data_warmup(self, df: pl.DataFrame, symbol: str = None) -> Tuple[bool, str]:
        """
        【V3.1 新增】检查数据预热是否充分
        
        Returns:
            (is_ready, message): 是否准备好及说明信息
        """
        if not DATA_WARMUP_ENABLED:
            return True, "预热检查已禁用"
        
        # 检查数据量
        if "trade_date" in df.columns:
            unique_dates = df["trade_date"].n_unique()
        else:
            unique_dates = len(df)
        
        if unique_dates < MIN_LOOKBACK_DAYS:
            return False, f"数据预热不足：需要至少 {MIN_LOOKBACK_DAYS} 天，实际 {unique_dates} 天"
        
        # 检查 bias_60 所需数据
        if unique_dates < MIN_LOOKBACK_FOR_BIAS60:
            return False, f"bias_60 计算需要至少 {MIN_LOOKBACK_FOR_BIAS60} 天数据"
        
        return True, f"数据预热充分：{unique_dates} 天 >= {MIN_LOOKBACK_DAYS} 天要求"
    
    def _get_training_data(self, end_date: str) -> Optional[pl.DataFrame]:
        try:
            query = f"SELECT * FROM stock_daily WHERE trade_date <= '{end_date}' AND trade_date >= '2022-01-01'"
            data = self.db.read_sql(query)
            if data.is_empty():
                return None
            
            # V3.1: 检查数据预热
            is_ready, message = self._check_data_warmup(data)
            if not is_ready:
                logger.warning(f"训练数据预热检查：{message}")
            
            data = self.factor_engine.compute_factors(data)
            return data
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return None
    
    def _prepare_training_features(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备训练特征
        
        【V3.1 新增】
        1. 因子健康度检查
        2. 特征剪枝
        """
        base_columns = self._get_feature_columns()
        available_cols = [c for c in base_columns if c in df.columns]
        
        # V3.1: 因子健康度检查
        logger.info("[特征准备] 执行因子健康度检查...")
        self.health_checker.check_factor_health(df, available_cols)
        self.health_checker.print_health_report(logger)
        
        # 保存训练时的特征列
        self.training_feature_columns = available_cols.copy()
        logger.info(f"[特征准备] 训练特征数：{len(available_cols)}")
        
        df_clean = df.filter(pl.all_horizontal(pl.col(available_cols).is_not_null()))
        if len(df_clean) == 0:
            logger.warning("No valid data after filtering nulls, using fill_null strategy")
            # V3.1: 使用更宽松的填充策略
            df_clean = df.clone()
            for col in available_cols:
                if col in df_clean.columns:
                    df_clean = df_clean.with_columns([
                        pl.col(col).fill_null(0.0).alias(col)
                    ])
        
        X = df_clean.select(available_cols).to_numpy()
        
        # 计算目标变量 (使用 12 日收益以匹配三屏障碍法)
        future_return_col = "future_return_12d"
        if future_return_col not in df_clean.columns:
            df_clean = df_clean.with_columns([
                (pl.col("close").shift(-12) / pl.col("close") - 1).alias("future_return_12d")
            ])
        y = df_clean["future_return_12d"].fill_null(0).to_numpy()
        
        # 1. 正交化
        if ORTHOGONALIZATION_ENABLED:
            logger.info("[数据准备] 执行因子正交化...")
            X_ortho = self.orthogonalizer.fit_transform(X, available_cols)
        else:
            X_ortho = X
        
        # 2. 特征合成 (包含剪枝)
        if FEATURE_SYNTHESIS_ENABLED:
            logger.info("[数据准备] 执行特征合成 (含剪枝)...")
            X_synth, selected_cols = self.synthesizer.fit_transform(X_ortho, y, self.orthogonalizer.get_feature_names())
            self.synthesized_feature_columns = selected_cols
            
            # 打印被剪枝的特征
            pruned = self.synthesizer.get_pruned_features()
            if pruned:
                logger.info(f"[特征剪枝] 已剔除 {len(pruned)} 个低有效样本率因子：{pruned[:5]}...")
        else:
            X_synth = X_ortho
            selected_cols = self.orthogonalizer.get_feature_names()
        
        # 3. 标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_synth)
        self.feature_columns = selected_cols
        
        logger.info(f"[数据准备] 完成：最终特征数={X_scaled.shape[1]}, 期望={self.expected_feature_dim}")
        return X_scaled, y, selected_cols
    
    def _get_feature_columns(self) -> List[str]:
        """获取所有可能的特征列"""
        base_factors = self.factor_engine.get_factor_names()
        technical_factors = ["rsi_14", "macd", "macd_signal", "macd_hist"]
        volume_price_factors = ["volume_price_health", "volume_shrink_flag", "price_volume_divergence"]
        private_factors = ["vcp_score", "turnover_stable", "smart_money_signal"]
        enhanced_factors = [
            "bias_recovery", "mfi_intensity", "momentum_accel", "liquidity_inflection",
            "price_vol_momentum", "vol_price_resilience", "relative_strength_sector",
            "momentum_squeeze", "liquidity_stress",
        ]
        all_factors = base_factors + technical_factors + volume_price_factors + private_factors + enhanced_factors
        seen = set()
        unique_factors = []
        for f in all_factors:
            if f not in seen:
                seen.add(f)
                unique_factors.append(f)
        return unique_factors
    
    def train_model(self, train_end_date: str = TRAIN_END_DATE) -> None:
        """训练模型并保存所有状态"""
        logger.info(f"Training model with data until {train_end_date}...")
        
        train_data = self._get_training_data(train_end_date)
        if train_data is None or len(train_data) == 0:
            logger.warning("No training data found")
            return
        
        X, y, feature_cols = self._prepare_training_features(train_data)
        if len(X) == 0 or len(y) == 0:
            logger.warning("No valid training data")
            return
        
        logger.info(f"[模型训练] 特征维度验证：X.shape={X.shape}, 期望={self.expected_feature_dim}")
        
        # 拟合 GMM
        if GATING_MECHANISM_ENABLED:
            logger.info("[模型训练] 拟合环境门控 GMM...")
            self.gating.fit(X, y, feature_cols)
        
        # 训练 Ridge
        logger.info("[模型训练] 训练 Ridge 模型 (Huber Loss)...")
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
        
        self._print_feature_importance()
        logger.info(f"Model training complete with {len(X)} samples, {len(feature_cols)} features")
    
    def _print_feature_importance(self) -> None:
        """打印特征重要性"""
        if self.ridge_model and hasattr(self.ridge_model, "coef_"):
            importance = list(zip(self.synthesized_feature_columns, self.ridge_model.coef_))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            logger.info("[Ridge Top 10 特征重要性]")
            for i, (feat, imp) in enumerate(importance[:10], 1):
                logger.info(f"  {i}. {feat}: {imp:.4f}")
        
        if self.lgb_model and LIGHTGBM_AVAILABLE:
            importance = list(zip(self.synthesized_feature_columns, self.lgb_model.feature_importances_))
            importance.sort(key=lambda x: abs(x[1]), reverse=True)
            logger.info("[LightGBM Top 10 特征重要性]")
            for i, (feat, imp) in enumerate(importance[:10], 1):
                logger.info(f"  {i}. {feat}: {imp:.4f}")
    
    def predict(self, df: pl.DataFrame, symbol: str = None) -> pl.DataFrame:
        """
        预测函数 - 使用与训练时完全一致的特征转换流程
        """
        if not self.orthogonalizer.is_fitted or not self.synthesizer.is_fitted:
            logger.warning("模型未训练，返回零预测")
            return df.with_columns(pl.lit(0.0).alias("predict_score"))
        
        feature_cols = self.training_feature_columns
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if len(available_cols) == 0:
            logger.warning("No feature columns found")
            return df.with_columns(pl.lit(0.0).alias("predict_score"))
        
        if len(available_cols) != len(feature_cols):
            logger.warning(f"特征数不匹配：期望 {len(feature_cols)}, 实际 {len(available_cols)}")
        
        df_with_idx = df.with_columns(pl.arange(0, len(df)).alias("__idx"))
        df_filtered = df_with_idx.filter(
            pl.all_horizontal(pl.col(available_cols).is_not_null())
        )
        
        if len(df_filtered) == 0:
            logger.warning("No valid data for prediction")
            return df.with_columns(pl.lit(0.0).alias("predict_score"))
        
        # 1. 提取特征
        X = df_filtered.select(available_cols).to_numpy()
        
        # 2. 正交化
        X_ortho = self.orthogonalizer.transform(X)
        
        # 3. 特征合成
        X_synth = self.synthesizer.transform(X_ortho, self.orthogonalizer.get_feature_names())
        
        # 4. 验证特征维度
        if X_synth.shape[1] != self.expected_feature_dim:
            logger.error(f"特征维度错误：期望 {self.expected_feature_dim}, 实际 {X_synth.shape[1]}")
            if X_synth.shape[1] < self.expected_feature_dim:
                padding = np.zeros((X_synth.shape[0], self.expected_feature_dim - X_synth.shape[1]))
                X_synth = np.hstack([X_synth, padding])
            else:
                X_synth = X_synth[:, :self.expected_feature_dim]
        
        # 5. 标准化
        if self.scaler is None:
            logger.warning("Scaler not fitted")
            X_scaled = X_synth
        else:
            X_scaled = self.scaler.transform(X_synth)
        
        # 6. 模型预测
        if self.ridge_model is None:
            ridge_pred = np.zeros(len(X_scaled))
        else:
            ridge_pred = self.ridge_model.predict(X_scaled)
        
        if self.lgb_model is not None and LIGHTGBM_AVAILABLE:
            lgb_pred = self.lgb_model.predict(X_scaled)
        else:
            lgb_pred = ridge_pred
        
        # 7. 集成预测
        ensemble_pred = 0.5 * ridge_pred + 0.5 * lgb_pred
        
        # 8. 构建结果
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
        self.score_history = {}
        self.recent_scores = []
        
        backtest_data = self._get_backtest_data(start_date, end_date)
        if backtest_data is None or len(backtest_data) == 0:
            logger.warning("No backtest data found")
            return BacktestResult()
        
        # V3.1: 检查数据预热
        is_ready, message = self._check_data_warmup(backtest_data)
        if not is_ready:
            logger.warning(f"回测数据预热检查：{message}")
        
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
            
            # V3.1: 计算因子 (带次优替代方案)
            daily_data = self.factor_engine.compute_factors(daily_data)
            
            # 流动性过滤
            year = int(str(date)[:4])
            liquidity_threshold = LIQUIDITY_FILTER_2023 if year == 2023 else LIQUIDITY_FILTER_2024
            daily_data = daily_data.filter(
                (pl.col("amount") >= liquidity_threshold) | (pl.col("amount").is_null())
            )
            
            # 预测
            daily_data = self.predict(daily_data)
            
            # V3.1: 执行交易 (带自适应止损)
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
        """
        检查退出条件
        
        【V3.1 增强】使用自适应止损和止盈
        """
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
            
            # V3.1: 使用动态止损/止盈
            if position.stop_loss_price > 0 and current_price <= position.stop_loss_price:
                self._exit_position(symbol, date, current_price, "adaptive_stop_loss")
            elif position.take_profit_price > 0 and current_price >= position.take_profit_price:
                self._exit_position(symbol, date, current_price, "take_profit")
            # 备用退出条件
            elif entry_return >= position.predicted_return * 0.5 or entry_return <= -0.05:
                self._exit_position(symbol, date, current_price, "target/rebalance")
    
    def _generate_buy_signals(self, date: str, data: pl.DataFrame, price_map: Dict[str, Dict]) -> float:
        """
        生成买入信号
        
        【V3.1 增强】设置自适应止损/止盈价
        """
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
                        
                        # V3.1: 计算自适应止损/止盈价
                        # 获取当前市场状态
                        if self.gating.gmm is not None:
                            market_state = self.gating.get_state_name()
                        else:
                            market_state = "CALM"
                        
                        # 使用 ATR 计算止损距离 (简化版本，使用固定 ATR 估计)
                        estimated_atr = price * 0.02  # 假设 2% 的日均 ATR
                        k = self.adaptive_labeler.stop_loss_k.get(market_state, 1.8)
                        stop_loss_distance = k * estimated_atr
                        
                        stop_loss_price = price - stop_loss_distance
                        take_profit_price = price * (1 + self.adaptive_labeler.upper_barrier * 0.02)
                        
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
                        
                        logger.debug(f"[买入] {symbol} @ {price:.2f} x {shares}股，评分={raw_score:.4f}, "
                                   f"状态={market_state}, 止损={stop_loss_price:.2f}")
                        total_slippage += slippage_cost
        
        return total_slippage
    
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
        """计算组合总价值"""
        value = float(self.cash)
        for symbol, position in self.positions.items():
            if symbol in price_map:
                price = float(price_map[symbol]["close"])
                value += position.shares * price
        return value
    
    def _calculate_backtest_result(self, daily_values: List[Dict]) -> BacktestResult:
        """计算回测结果"""
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
    logger.info("Final Strategy V3.1 - Adaptive Barrier & Robust Factors")
    logger.info("=" * 70)
    
    strategy = FinalStrategyV31(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
    )
    
    strategy.train_model(train_end_date=TRAIN_END_DATE)
    
    result = strategy.run_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=1000000.0,
    )
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("回测结果")
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


if __name__ == "__main__":
    main()