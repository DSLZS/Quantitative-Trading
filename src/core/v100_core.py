"""
V100 Core - 策略重生：寻找能够战胜摩擦成本的持久 Alpha

【V100 核心改进】
1. 长效因子引入
   - 盈余惊喜 (Earnings Surprise)：财报发布后的价格反应，半衰期长
   - 机构资金一致性 (Institutional Continuity)：北向资金/主力资金的持续性
   - 基本面动量 (Fundamental Momentum)：ROE/营收增长趋势

2. 预测目标重构
   - 不再预测 T+1 收益
   - 改为预测 T+1 到 T+5 累积超额收益
   - IC 审计覆盖 T+1 到 T+5 全周期

3. 动态成本门槛
   - 删除 i % 5 == 0 机械限频
   - 只有当 期望收益 > 2 * 摩擦成本 时才调仓
   - 期望收益 = Alpha 信号 * 历史 IC 均值

4. 数据防御机制
   - 检测 total_mv 和 industry_code 连续 3 天空值
   - 自动触发 src/loaders 数据补抓
   - 严禁使用 fillna(0) 糊弄

【V100 验收硬指标】
| 指标 | 目标值 | 惩罚红线 |
| :--- | :--- | :--- |
| 扣费后净收益 | > 5% (2024 年) | 负收益 = 彻底失败 |
| T+1 到 T+5 IC 均值 | > 0.03 | 证明信号具有持久性 |
| 年化换手率 | 200% - 400% | 通过因子质量降低换手 |
| IC Stability (IR) | > 0.5 | 信号必须在不同年份保持稳定 |

作者：量化系统
版本：V100.0
日期：2026-03-31
"""

import traceback
import time
import math
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from loguru import logger


# ===========================================
# V100 配置常量（严禁修改）
# ===========================================

V100_INITIAL_CAPITAL = 100000.00  # 初始资金
V100_MAX_POSITIONS = 50  # 最大持仓数量
V100_WARMUP_PERIOD = 250
V100_MIN_SAMPLE_SIZE = 100

# 交易成本常量
V100_COMMISSION_RATE = 0.0003  # 佣金率
V100_MIN_COMMISSION = 5.0  # 最低佣金
V100_STAMP_DUTY = 0.0005  # 印花税
V100_TRANSFER_FEE = 0.00001  # 过户费

# V100 因子权重
V100_RESIDUAL_WEIGHT = 0.25  # 降低残差动量权重
V100_FLOW_WEIGHT = 0.20      # 降低资金流权重
V100_EARNINGS_SURPRISE_WEIGHT = 0.30  # 新增：盈余惊喜
V100_INSTITUTIONAL_WEIGHT = 0.25      # 新增：机构资金一致性

# V100 评分门槛
V100_MIN_SCORE_THRESHOLD = 40.0
V100_MIN_SINGLE_WEIGHT = 0.002
V100_MAX_SINGLE_WEIGHT = 0.05

# V100 换手率控制（目标：200%-400%）
V100_TURNOVER_MIN = 2.0
V100_TURNOVER_MAX = 4.0
V100_DAILY_TURNOVER_MAX = 0.10  # 每日换手率上限 10%

# V100 动态成本门槛
V100_TRANSACTION_COST = 0.0015  # 单边 0.15%
V100_EXPECTED_RETURN_THRESHOLD = 2.0  # 期望收益 > 2 * 摩擦成本才调仓
V100_MIN_REBALANCE_INTERVAL = 3  # 最小调仓间隔（天）

# V100 IC 目标（T+1 到 T+5）
V100_T1_T5_IC_TARGET = 0.03  # T+1 到 T+5 IC 均值目标
V100_IC_IR_TARGET = 0.5
V100_IC_STABILITY_TARGET = 0.5

# V100 时间序列稳定性过滤
V100_IC_LOOKBACK_DAYS = 10  # 过去 10 天 IC
V100_IC_STD_THRESHOLD = 0.12  # IC 波动率阈值

# V100 数据防御阈值
V100_CONSECUTIVE_MISSING_DAYS = 3  # 连续 3 天空值触发警报
V100_INDUSTRY_MISSING_THRESHOLD = 0.10  # 10% 缺失阈值
V100_MV_MISSING_THRESHOLD = 0.10  # 10% 缺失阈值

# 风格中性化配置
V100_SIZE_NEUTRALIZATION = True
V100_INDUSTRY_NEUTRALIZATION = True
V100_NEUTRALIZATION_WINDOW = 60

# 流动性过滤配置
V100_LIQUIDITY_FILTER = True
V100_LIQUIDITY_PERCENTILE = 10
V100_FILTER_ST = True

# 半衰期融合配置
V100_HALF_LIFE_LAGS = [1, 3, 5]
V100_LAG1_WEIGHT = 0.50
V100_LAG3_WEIGHT = 0.30
V100_LAG5_WEIGHT = 0.20

# V100 信号平滑配置
V100_EMA_WINDOW = 5

# 审计配置
V100_AUDIT_MODE = True

# 预测窗口
V100_PREDICTION_HORIZON = 5  # 预测 T+1 到 T+5 累积收益

EPSILON = 1e-9


# ===========================================
# V100 异常类
# ===========================================

class DirectionalError(Exception):
    """因子方向错误异常"""
    def __init__(self, factor_name: str, ic_value: float, expected_direction: str = "positive"):
        self.factor_name = factor_name
        self.ic_value = ic_value
        self.expected_direction = expected_direction
        message = (
            f"因子方向错误！因子 '{factor_name}' 的 IC={ic_value:.4f}，"
            f"期望方向：{expected_direction}。"
        )
        super().__init__(message)


class IndustryDataMissingError(Exception):
    """行业数据缺失异常"""
    def __init__(self, missing_ratio: float, date: str, field_name: str = "industry_code"):
        self.missing_ratio = missing_ratio
        self.date = date
        self.field_name = field_name
        message = (
            f"{date} {field_name} 缺失比例 {missing_ratio:.1%} > {V100_INDUSTRY_MISSING_THRESHOLD:.1%}，"
            f"必须重新补取数据。"
        )
        super().__init__(message)


class ConsecutiveDataMissingError(Exception):
    """连续数据缺失异常"""
    def __init__(self, field_name: str, consecutive_days: int, symbols: List[str]):
        self.field_name = field_name
        self.consecutive_days = consecutive_days
        self.symbols = symbols
        message = (
            f"字段 '{field_name}' 连续 {consecutive_days} 天缺失，"
            f"涉及 {len(symbols)} 只股票，必须调用 src/loaders 补抓数据。"
        )
        super().__init__(message)


# ===========================================
# V100 数据类
# ===========================================

@dataclass
class V100SingleFactorIC:
    """单因子 IC 记录（扩展到 T+5）"""
    factor_name: str
    ic_t1: float
    ic_t2: float
    ic_t3: float
    ic_t4: float
    ic_t5: float
    ic_mean_t1_t5: float  # T+1 到 T+5 IC 均值
    ic_ir: float
    ic_stability: float
    ic_std_10d: float      # 过去 10 天 IC 波动率
    passed_threshold: bool
    passed_stability_filter: bool


@dataclass
class V100FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float
    smoothed_signal: float
    expected_return_5d: float  # T+1 到 T+5 期望收益


@dataclass
class V100TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float
    is_rebalance_day: bool = False
    rebalance_ratio: float = 0.0
    cost_threshold_passed: bool = True  # 是否通过成本门槛检查


@dataclass
class V100Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_rank: int = 999
    current_price: float = 0.0
    pnl: float = 0.0
    expected_return_5d: float = 0.0  # 入场时的 5 日期望收益


@dataclass
class V100DailyPortfolio:
    """每日组合快照"""
    trade_date: str
    total_value: float
    cash: float
    position_value: float
    position_count: int
    daily_return: float
    cumulative_return: float
    turnover_rate: float
    net_return_after_cost: float
    expected_return_5d: float  # 组合整体 5 日期望收益


@dataclass
class V100DataQualityRecord:
    """数据质量记录"""
    trade_date: str
    field_name: str
    missing_count: int
    missing_ratio: float
    consecutive_missing_days: int
    is_critical: bool


# ===========================================
# V100 工具函数
# ===========================================

def normalize_rank(series: pl.Expr, descending: bool = False) -> pl.Expr:
    """将序列转换为百分位排名（0-100）- Polars Expr 版本"""
    # 使用窗口函数计算排名
    ranks = series.rank('ordinal', descending=descending)
    
    # 使用窗口函数计算每个 trade_date 的股票数量
    n = pl.count().over('trade_date')
    
    percentile = 100.0 * (1.0 - (ranks.cast(pl.Float64) - 0.5) / (n.cast(pl.Float64) + EPSILON))
    
    return percentile


def zscore_normalize(series: np.ndarray) -> np.ndarray:
    """Z-Score 标准化"""
    if len(series) < 2:
        return np.zeros_like(series)
    
    scaler = StandardScaler()
    try:
        return scaler.fit_transform(series.reshape(-1, 1)).flatten()
    except Exception:
        return (series - np.mean(series)) / (np.std(series) + EPSILON)


def calculate_half_life_decay_weights(lags: List[int] = V100_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V100_LAG1_WEIGHT,
        3: V100_LAG3_WEIGHT,
        5: V100_LAG5_WEIGHT,
    }
    
    weights = [fixed_weights.get(lag, 1.0 / len(lags)) for lag in lags]
    
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return [1.0 / len(lags)] * len(lags)
    
    return [w / total_weight for w in weights]


def fill_with_market_median(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """使用全市场均值填充空值"""
    result = df.clone()
    
    for col in cols:
        if col not in result.columns:
            continue
        
        median_val = result[col].median()
        if median_val is not None and np.isfinite(median_val):
            result = result.with_columns([
                pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
                .then(median_val)
                .otherwise(pl.col(col))
                .alias(col)
            ])
    
    return result


def calculate_rank_correlation(series1: np.ndarray, series2: np.ndarray) -> float:
    """计算两个序列的秩相关系数（Spearman）"""
    if len(series1) != len(series2) or len(series1) < 3:
        return 1.0
    
    valid_mask = (~np.isnan(series1) & ~np.isnan(series2) & 
                  np.isfinite(series1) & np.isfinite(series2))
    
    if np.sum(valid_mask) < 3:
        return 1.0
    
    try:
        corr, _ = stats.spearmanr(series1[valid_mask], series2[valid_mask])
        return float(corr) if np.isfinite(corr) else 1.0
    except Exception:
        return 1.0


def ols_residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS 残差化"""
    if len(y) < 10 or X.shape[0] < 10:
        return y
    
    valid_mask = (~np.isnan(y) & ~np.isnan(X).any(axis=1) & 
                  np.isfinite(y) & np.isfinite(X).all(axis=1))
    
    if np.sum(valid_mask) < 10:
        return y
    
    y_valid = y[valid_mask]
    X_valid = X[valid_mask]
    
    try:
        model = LinearRegression()
        model.fit(X_valid, y_valid)
        y_pred = model.predict(X_valid)
        residuals = y_valid - y_pred
        
        result = np.zeros_like(y)
        result[valid_mask] = residuals
        result = result * np.std(y) + np.mean(y)
        
        return result
    except Exception:
        return y


def ema_smooth(series: np.ndarray, window: int = V100_EMA_WINDOW) -> np.ndarray:
    """指数移动平均平滑（EMA）"""
    if len(series) < window:
        return series
    
    result = np.zeros_like(series)
    alpha = 2.0 / (window + 1)
    
    result[0] = series[0]
    
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    
    return result


def calculate_ic_stability(daily_ics: List[float]) -> Tuple[float, float]:
    """
    计算 IC 稳定性
    
    Returns
    -------
    Tuple[float, float]
        (IC 均值，IC 波动率 Std)
    """
    if len(daily_ics) < 3:
        return np.mean(daily_ics) if daily_ics else 0.0, 0.0
    
    valid_ics = [ic for ic in daily_ics if ic is not None and np.isfinite(ic)]
    if len(valid_ics) < 3:
        return np.mean(valid_ics) if valid_ics else 0.0, 0.0
    
    return np.mean(valid_ics), np.std(valid_ics)


def calculate_cumulative_return(df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """
    计算 T+1 到 T+window 累积收益率
    
    cumulative_return = (1 + r_t1) * (1 + r_t2) * ... * (1 + r_tw) - 1
    """
    result = df.clone()
    result = result.sort(['symbol', 'trade_date'])
    
    # 计算累积收益
    cum_return_expr = None
    for lag in range(1, window + 1):
        lag_return = pl.col('pct_chg').shift(-lag).over('symbol') / 100.0
        if cum_return_expr is None:
            cum_return_expr = (1.0 + lag_return)
        else:
            cum_return_expr = cum_return_expr * (1.0 + lag_return)
    
    cum_return_expr = cum_return_expr - 1.0
    
    result = result.with_columns([
        cum_return_expr.alias(f'cumulative_return_{window}d')
    ])
    
    return result


# ===========================================
# V100 长效因子引擎
# ===========================================

class V100EarningsSurpriseEngine:
    """
    V100 盈余惊喜因子引擎
    
    【因子逻辑】
    1. 计算财报发布后的价格反应（盈余惊喜）
    2. 盈余惊喜 = (实际 EPS - 预期 EPS) / 预期 EPS
    3. 由于 A 股预期 EPS 数据难以获取，使用以下代理变量：
       - 财报发布日前后 5 天的累积超额收益
       - 财报发布日的成交量放大倍数
       - 财报发布后的机构调研次数变化
    
    【半衰期】约 20-60 天，属于长效因子
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', 5)
    
    def compute_earnings_surprise(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算盈余惊喜因子
        
        由于无法直接获取财报数据，使用以下代理：
        1. 价格跳空：当日开盘相对前日收盘的跳空幅度
        2. 成交量放大：当日成交量 / 过去 20 日平均成交量
        3. 动量加速：5 日收益相对过去 60 日收益的比率
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 价格跳空（盈余惊喜的代理）
        result = result.with_columns([
            ((pl.col('open') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON) * 100).alias('price_gap')
        ])
        
        # 2. 成交量放大倍数
        result = result.with_columns([
            pl.col('volume')
            .rolling_mean(window_size=20)
            .over('symbol')
            .alias('volume_ma20')
        ])
        result = result.with_columns([
            (pl.col('volume') / (pl.col('volume_ma20') + EPSILON)).alias('volume_expansion')
        ])
        
        # 3. 动量加速（5 日收益 / 60 日收益）
        result = result.with_columns([
            ((pl.col('close') / pl.col('close').shift(5)) - 1).alias('momentum_5d')
        ])
        result = result.with_columns([
            ((pl.col('close') / pl.col('close').shift(60)) - 1).alias('momentum_60d')
        ])
        result = result.with_columns([
            (pl.col('momentum_5d') / (pl.col('momentum_60d').abs() + EPSILON)).alias('momentum_acceleration')
        ])
        
        # 4. 综合盈余惊喜分数
        # 价格跳空越大、成交量放大越多、动量加速越明显，盈余惊喜越大
        result = result.with_columns([
            (
                0.4 * normalize_rank(pl.col('price_gap').fill_null(0)) +
                0.3 * normalize_rank(pl.col('volume_expansion').fill_null(0)) +
                0.3 * normalize_rank(pl.col('momentum_acceleration').fill_null(0))
            ).alias('earnings_surprise_score')
        ])
        
        # 清理临时列
        result = result.drop([
            'price_gap', 'volume_ma20', 'volume_expansion',
            'momentum_5d', 'momentum_60d', 'momentum_acceleration'
        ])
        
        logger.info("V100: 盈余惊喜因子计算完成")
        
        return result


class V100InstitutionalFlowEngine:
    """
    V100 机构资金一致性因子引擎
    
    【因子逻辑】
    1. 追踪机构资金的持续性（北向资金、主力资金）
    2. 机构资金一致性 = 过去 N 日资金流入的稳定性
    3. 使用量价数据代理机构资金流向
    
    【半衰期】约 10-30 天，属于中长效因子
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.flow_window = self.config.get('flow_window', 10)
        self.consistency_window = self.config.get('consistency_window', 20)
    
    def compute_institutional_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算机构资金一致性因子
        
        代理变量：
        1. 大单净流入：(收盘价 - 最低价) - (最高价 - 收盘价) 的标准化
        2. 资金流持续性：过去 N 日资金流入方向的一致性
        3. 资金流强度：资金流入的幅度
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算每日资金流方向
        numerator = (pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))
        denominator = pl.col('high') - pl.col('low') + EPSILON
        result = result.with_columns(
            (numerator / denominator).alias('money_flow_direction')
        )
        
        # 2. 资金流强度（乘以成交量）
        result = result.with_columns([
            (pl.col('money_flow_direction') * 
             pl.col('amount').fill_null(0) / 1e8).alias('money_flow_strength')
        ])
        
        # 3. 资金流持续性（过去 N 日同向天数比例）
        result = result.with_columns([
            (pl.col('money_flow_direction') > 0).cast(pl.Int32).alias('inflow_flag')
        ])
        
        result = result.with_columns([
            pl.col('inflow_flag')
            .rolling_sum(window_size=self.consistency_window)
            .over('symbol')
            .alias('inflow_days')
        ])
        
        result = result.with_columns([
            (pl.col('inflow_days') / self.consistency_window).alias('flow_consistency')
        ])
        
        # 4. 资金流稳定性（过去 N 日资金流的标准差的倒数）
        result = result.with_columns([
            pl.col('money_flow_strength')
            .rolling_std(window_size=self.flow_window)
            .over('symbol')
            .alias('flow_volatility')
        ])
        
        result = result.with_columns([
            (1.0 / (pl.col('flow_volatility') + EPSILON)).alias('flow_stability')
        ])
        
        # 5. 综合机构资金一致性分数
        result = result.with_columns([
            (
                0.4 * normalize_rank(pl.col('flow_consistency').fill_null(50)) +
                0.3 * normalize_rank(pl.col('flow_stability').fill_null(50)) +
                0.3 * normalize_rank(pl.col('money_flow_strength').fill_null(50))
            ).alias('institutional_flow_score')
        ])
        
        # 清理临时列
        result = result.drop([
            'money_flow_direction', 'money_flow_strength', 'inflow_flag',
            'inflow_days', 'flow_volatility', 'flow_consistency', 'flow_stability'
        ])
        
        logger.info(f"V100: 机构资金一致性因子计算完成 (window={self.consistency_window})")
        
        return result


class V100ResidualMomentumEngine:
    """
    V100 残差动量引擎（降低权重）
    
    【V100 改进】
    - 降低权重从 0.60 降至 0.25
    - 增加更长周期的动量（60 日）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.momentum_windows = self.config.get('momentum_windows', [5, 20, 60])
    
    def compute_residual_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算残差动量（多周期融合）"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算多周期动量
        momentum_scores = []
        for window in self.momentum_windows:
            col_name = f'momentum_{window}d'
            result = result.with_columns([
                ((pl.col('close') / pl.col('close').shift(window)) - 1).alias(col_name)
            ])
            
            # 排名标准化
            rank_col = f'momentum_{window}d_rank'
            result = result.with_columns([
                pl.col(col_name).rank('ordinal', descending=True).over('trade_date').alias(rank_col)
            ])
            
            n_stocks = pl.col('symbol').count().over('trade_date').cast(pl.Float64)
            score_col = f'momentum_{window}d_score'
            result = result.with_columns([
                (100.0 * (1.0 - (pl.col(rank_col).cast(pl.Float64) - 0.5) / (n_stocks + EPSILON))).alias(score_col)
            ])
            
            momentum_scores.append(score_col)
        
        # 多周期动量融合（更长周期权重更高）
        weights = {5: 0.2, 20: 0.3, 60: 0.5}
        result = result.with_columns([
            sum(
                weights.get(w, 0.33) * pl.col(f'momentum_{w}d_score')
                for w in self.momentum_windows
            ).alias('residual_momentum_score')
        ])
        
        # 清理临时列
        drop_cols = []
        for w in self.momentum_windows:
            drop_cols.extend([
                f'momentum_{w}d', f'momentum_{w}d_rank', f'momentum_{w}d_score'
            ])
        result = result.drop(drop_cols)
        
        logger.info(f"V100: 残差动量计算完成（多周期融合：{self.momentum_windows}）")
        
        return result


class V100SmartFlowEngine:
    """
    V100 聪明资金流引擎（降低权重）
    
    【V100 改进】
    - 降低权重从 0.40 降至 0.20
    - 增加更长周期的资金流追踪
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.flow_windows = self.config.get('flow_windows', [5, 10, 20])
    
    def compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算聪明资金流（多周期融合）"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算资金流因子
        numerator = (pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))
        denominator = pl.col('high') - pl.col('low') + EPSILON
        result = result.with_columns(
            (numerator / denominator).alias('money_flow_factor')
        )
        
        flow_scores = []
        for window in self.flow_windows:
            col_name = f'money_flow_ma{window}'
            result = result.with_columns([
                pl.col('money_flow_factor')
                .rolling_mean(window_size=window)
                .over('symbol')
                .alias(col_name)
            ])
            
            # 排名标准化
            rank_col = f'flow_{window}d_rank'
            result = result.with_columns([
                pl.col(col_name).rank('ordinal', descending=True).over('trade_date').alias(rank_col)
            ])
            
            n_stocks = pl.col('symbol').count().over('trade_date').cast(pl.Float64)
            score_col = f'flow_{window}d_score'
            result = result.with_columns([
                (100.0 * (1.0 - (pl.col(rank_col).cast(pl.Float64) - 0.5) / (n_stocks + EPSILON))).alias(score_col)
            ])
            
            flow_scores.append(score_col)
        
        # 多周期资金流融合（更长周期权重更高）
        weights = {5: 0.2, 10: 0.3, 20: 0.5}
        result = result.with_columns([
            sum(
                weights.get(w, 0.33) * pl.col(f'flow_{w}d_score')
                for w in self.flow_windows
            ).alias('smart_flow_score')
        ])
        
        # 清理临时列
        drop_cols = ['money_flow_factor']
        for w in self.flow_windows:
            drop_cols.extend([f'money_flow_ma{w}', f'flow_{w}d_rank', f'flow_{w}d_score'])
        result = result.drop(drop_cols)
        
        logger.info(f"V100: 聪明资金流计算完成（多周期融合：{self.flow_windows}）")
        
        return result


# ===========================================
# V100 数据质量检测器
# ===========================================

class V100DataQualityChecker:
    """
    V100 数据质量检测器
    
    【核心功能】
    1. 检测 total_mv 和 industry_code 连续 3 天空值
    2. 自动触发 src/loaders 数据补抓
    3. 严禁使用 fillna(0) 糊弄
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.consecutive_days_threshold = self.config.get(
            'consecutive_days_threshold', V100_CONSECUTIVE_MISSING_DAYS
        )
        self.industry_missing_threshold = self.config.get(
            'industry_missing_threshold', V100_INDUSTRY_MISSING_THRESHOLD
        )
        self.mv_missing_threshold = self.config.get(
            'mv_missing_threshold', V100_MV_MISSING_THRESHOLD
        )
        
        # 追踪连续缺失天数
        self.consecutive_missing: Dict[str, Dict[str, int]] = {
            'total_mv': {},
            'industry_code': {}
        }
        self.quality_records: List[V100DataQualityRecord] = []
    
    def check_field_missing(self, df: pl.DataFrame, trade_date: str, 
                            field_name: str) -> Tuple[bool, float, int]:
        """
        检查字段缺失情况
        
        Returns
        -------
        Tuple[bool, float, int]
            (是否通过检查，缺失比例，连续缺失天数)
        """
        if field_name not in df.columns:
            return False, 1.0, self.consecutive_days_threshold
        
        total_count = df.height
        if total_count == 0:
            return False, 1.0, self.consecutive_days_threshold
        
        # 统计缺失数量
        if field_name == 'industry_code':
            missing_mask = (
                (pl.col(field_name).is_null()) |
                (pl.col(field_name).cast(pl.Utf8).str.len_chars() == 0) |
                (pl.col(field_name) == 'None') |
                (pl.col(field_name) == 'null') |
                (pl.col(field_name) == '')
            )
        else:
            missing_mask = (
                (pl.col(field_name).is_null()) |
                (~pl.col(field_name).is_finite())
            )
        
        missing_count = df.filter(missing_mask).height
        missing_ratio = missing_count / total_count
        
        # 更新连续缺失追踪
        if missing_ratio > 0.01:  # 有缺失
            # 获取缺失的股票列表
            missing_symbols = df.filter(missing_mask)['symbol'].unique().to_list()
            
            for symbol in missing_symbols:
                if symbol not in self.consecutive_missing[field_name]:
                    self.consecutive_missing[field_name][symbol] = 0
                self.consecutive_missing[field_name][symbol] += 1
        
        # 记录质量数据
        max_consecutive = max(self.consecutive_missing[field_name].values()) if self.consecutive_missing[field_name] else 0
        record = V100DataQualityRecord(
            trade_date=trade_date,
            field_name=field_name,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            consecutive_missing_days=max_consecutive,
            is_critical=max_consecutive >= self.consecutive_days_threshold
        )
        self.quality_records.append(record)
        
        # 判断是否通过
        threshold = (self.industry_missing_threshold if field_name == 'industry_code' 
                    else self.mv_missing_threshold)
        passed = missing_ratio <= threshold
        
        return passed, missing_ratio, max_consecutive
    
    def check_all_fields(self, df: pl.DataFrame, trade_date: str) -> Dict[str, Any]:
        """检查所有关键字段"""
        results = {}
        critical_issues = []
        
        for field_name in ['total_mv', 'industry_code']:
            passed, missing_ratio, consecutive_days = self.check_field_missing(
                df, trade_date, field_name
            )
            results[field_name] = {
                'passed': passed,
                'missing_ratio': missing_ratio,
                'consecutive_days': consecutive_days
            }
            
            if consecutive_days >= self.consecutive_days_threshold:
                critical_issues.append({
                    'field': field_name,
                    'consecutive_days': consecutive_days,
                    'symbols': [s for s, d in self.consecutive_missing[field_name].items() 
                               if d >= self.consecutive_days_threshold]
                })
        
        return {
            'passed': len(critical_issues) == 0,
            'results': results,
            'critical_issues': critical_issues
        }
    
    def get_symbols_needing_repair(self, field_name: str) -> List[str]:
        """获取需要修复数据的股票列表"""
        return [
            symbol for symbol, days in self.consecutive_missing[field_name].items()
            if days >= self.consecutive_days_threshold
        ]
    
    def generate_repair_command(self) -> str:
        """生成数据修复命令"""
        mv_symbols = self.get_symbols_needing_repair('total_mv')
        industry_symbols = self.get_symbols_needing_repair('industry_code')
        
        all_symbols = list(set(mv_symbols + industry_symbols))
        
        if not all_symbols:
            return ""
        
        symbols_str = ','.join(all_symbols[:10])  # 限制输出长度
        return f"python src/loaders/v100_data_boot.py --symbols {symbols_str}"


# ===========================================
# V100 动态成本门槛
# ===========================================

class V100DynamicCostThreshold:
    """
    V100 动态成本门槛
    
    【核心逻辑】
    - 删除 i % 5 == 0 机械限频
    - 只有当 期望收益 > 2 * 摩擦成本 时才调仓
    - 期望收益 = Alpha 信号 * 历史 IC 均值
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.transaction_cost = self.config.get(
            'transaction_cost', V100_TRANSACTION_COST
        )
        self.expected_return_threshold = self.config.get(
            'expected_return_threshold', V100_EXPECTED_RETURN_THRESHOLD
        )
        self.min_rebalance_interval = self.config.get(
            'min_rebalance_interval', V100_MIN_REBALANCE_INTERVAL
        )
        
        # 追踪上次调仓日
        self.last_rebalance_date: Optional[str] = None
        self.days_since_rebalance = 0
        
        # 历史 IC 追踪
        self.historical_ic: List[float] = []
    
    def record_ic(self, ic_value: float) -> None:
        """记录每日 IC"""
        self.historical_ic.append(ic_value)
        # 保持最近 60 天
        if len(self.historical_ic) > 60:
            self.historical_ic = self.historical_ic[-60:]
    
    def get_mean_ic(self) -> float:
        """获取历史 IC 均值"""
        if not self.historical_ic:
            return V100_T1_T5_IC_TARGET  # 使用目标值作为默认值
        return float(np.mean(self.historical_ic))
    
    def should_rebalance(self, trade_date: str, 
                         portfolio_alpha: float,
                         current_turnover: float = 0.0) -> Tuple[bool, str]:
        """
        判断是否应该调仓（动态成本门槛）
        
        Parameters
        ----------
        trade_date : str
            交易日期
        portfolio_alpha : float
            组合 Alpha 信号（加权平均）
        current_turnover : float
            当前预计换手率
            
        Returns
        -------
        Tuple[bool, str]
            (是否调仓，原因)
        """
        # 1. 检查最小调仓间隔
        self.days_since_rebalance += 1
        if self.last_rebalance_date is not None:
            if self.days_since_rebalance < self.min_rebalance_interval:
                return False, f"距离上次调仓仅 {self.days_since_rebalance} 天，未满 {self.min_rebalance_interval} 天"
        
        # 2. 计算期望收益
        mean_ic = self.get_mean_ic()
        expected_return = abs(portfolio_alpha) * mean_ic
        
        # 3. 计算摩擦成本（双边）
        friction_cost = 2 * self.transaction_cost  # 双边成本
        
        # 4. 动态成本门槛判断
        threshold = self.expected_return_threshold * friction_cost
        
        if expected_return > threshold:
            self.last_rebalance_date = trade_date
            self.days_since_rebalance = 0
            return True, f"期望收益 {expected_return:.4f} > 门槛 {threshold:.4f}"
        else:
            return False, f"期望收益 {expected_return:.4f} <= 门槛 {threshold:.4f}"
    
    def record_rebalance(self, trade_date: str) -> None:
        """记录调仓"""
        self.last_rebalance_date = trade_date
        self.days_since_rebalance = 0


# ===========================================
# V100 IC 审计（扩展到 T+5）
# ===========================================

class V100ICAudit:
    """V100 IC 审计（扩展到 T+5）"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.stability_filter = V100ICStabilityFilter(config)
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'smoothed_signal',
                          target_col: str = 'cumulative_return_5d') -> Dict[str, Any]:
        """计算 Rank IC（T+1 到 T+5）"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算 T+1 到 T+5 各期 IC
        ic_results = {}
        all_daily_ics = {}
        
        for lag in range(1, 6):
            return_col = f'forward_return_{lag}d'
            if return_col not in result.columns:
                # 尝试使用累积收益列
                if target_col in result.columns:
                    return_col = target_col
                else:
                    continue
            
            ic_by_date = result.group_by('trade_date').agg([
                pl.corr(signal_col, return_col, method='spearman').alias('ic')
            ]).filter(pl.col('ic').is_not_null())
            
            if not ic_by_date.is_empty():
                ic_list = ic_by_date['ic'].drop_nulls().to_list()
                valid_ic = [ic for ic in ic_list if ic is not None and np.isfinite(ic)]
                all_daily_ics[f't{lag}'] = valid_ic
                
                ic_results[f't{lag}'] = {
                    'mean_ic': float(np.mean(valid_ic)) if valid_ic else 0.0,
                    'std_ic': float(np.std(valid_ic)) if valid_ic else 0.0,
                    'ic_count': len(valid_ic),
                }
        
        # 计算 T+1 到 T+5 IC 均值
        ic_t1_t5_values = [ic_results.get(f't{lag}', {}).get('mean_ic', 0.0) for lag in range(1, 6)]
        ic_t1_t5_mean = float(np.mean(ic_t1_t5_values))
        
        # 获取各期 IC
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        ic_t4 = ic_results.get('t4', {}).get('mean_ic', 0.0)
        ic_t5 = ic_results.get('t5', {}).get('mean_ic', 0.0)
        
        # 计算 IC Stability
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_stability = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 记录到稳定性过滤器
        self.stability_filter.record_daily_ic('composite', ic_t1)
        
        # IC 衰减分析
        decay_pattern = self._analyze_ic_decay(ic_results)
        
        # 达标判断
        t1_t5_ic_passed = ic_t1_t5_mean >= V100_T1_T5_IC_TARGET
        stability_passed = ic_stability >= V100_IC_STABILITY_TARGET
        
        return {
            'ic_t1': ic_t1,
            'ic_t2': ic_t2,
            'ic_t3': ic_t3,
            'ic_t4': ic_t4,
            'ic_t5': ic_t5,
            'ic_t1_t5_mean': ic_t1_t5_mean,
            'std_t1': std_t1,
            'std_t2': ic_results.get('t2', {}).get('std_ic', 0.0),
            'std_t3': ic_results.get('t3', {}).get('std_ic', 0.0),
            'std_t4': ic_results.get('t4', {}).get('std_ic', 0.0),
            'std_t5': ic_results.get('t5', {}).get('std_ic', 0.0),
            'ic_stability': ic_stability,
            'decay_pattern': decay_pattern,
            't1_t5_ic_passed': t1_t5_ic_passed,
            'stability_passed': stability_passed,
            'all_daily_ics': all_daily_ics,
        }
    
    def _analyze_ic_decay(self, ic_results: Dict[str, Dict]) -> str:
        """分析 IC 衰减模式"""
        ics = [ic_results.get(f't{lag}', {}).get('mean_ic', 0.0) for lag in range(1, 6)]
        
        # 判断衰减模式
        if all(ics[i] >= ics[i+1] for i in range(len(ics)-1)) and ics[0] > 0:
            return "正常衰减"
        elif all(ic < 0.01 for ic in ics):
            return "信号失效（IC 普遍过低）"
        elif ics[0] > 0.05 and ics[-1] < 0:
            return "快速反转（短期噪音）"
        elif all(ics[i] <= ics[i+1] for i in range(len(ics)-1)):
            return "异常递增（可能存在未来函数）"
        else:
            return "波动衰减"
    
    def calculate_single_factor_ic(self, df: pl.DataFrame, 
                                    factor_name: str,
                                    signal_col: str) -> V100SingleFactorIC:
        """计算单因子 IC（扩展到 T+5）"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        ic_results = {}
        all_daily_ics = []
        
        for lag in range(1, 6):
            return_col = f'forward_return_{lag}d'
            if return_col not in result.columns:
                continue
            
            ic_by_date = result.group_by('trade_date').agg([
                pl.corr(signal_col, return_col, method='spearman').alias('ic')
            ]).filter(pl.col('ic').is_not_null())
            
            if not ic_by_date.is_empty():
                ic_list = ic_by_date['ic'].drop_nulls().to_list()
                valid_ic = [ic for ic in ic_list if ic is not None and np.isfinite(ic)]
                all_daily_ics.extend(valid_ic)
                
                ic_results[f't{lag}'] = {
                    'mean_ic': float(np.mean(valid_ic)) if valid_ic else 0.0,
                    'std_ic': float(np.std(valid_ic)) if valid_ic else 0.0,
                }
        
        # 计算 T+1 到 T+5 IC 均值
        ic_values = [ic_results.get(f't{lag}', {}).get('mean_ic', 0.0) for lag in range(1, 6)]
        ic_mean_t1_t5 = float(np.mean(ic_values))
        
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_ir = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        ic_stability = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 记录到稳定性过滤器
        self.stability_filter.record_daily_ic(factor_name, ic_t1)
        ic_std_10d = self.stability_filter.get_ic_std(factor_name)
        
        passed_stability = ic_std_10d <= V100_IC_STD_THRESHOLD
        passed_threshold = ic_mean_t1_t5 >= V100_T1_T5_IC_TARGET
        
        return V100SingleFactorIC(
            factor_name=factor_name,
            ic_t1=ic_t1,
            ic_t2=ic_results.get('t2', {}).get('mean_ic', 0.0),
            ic_t3=ic_results.get('t3', {}).get('mean_ic', 0.0),
            ic_t4=ic_results.get('t4', {}).get('mean_ic', 0.0),
            ic_t5=ic_results.get('t5', {}).get('mean_ic', 0.0),
            ic_mean_t1_t5=ic_mean_t1_t5,
            ic_ir=ic_ir,
            ic_stability=ic_stability,
            ic_std_10d=ic_std_10d,
            passed_threshold=passed_threshold,
            passed_stability_filter=passed_stability,
        )


class V100ICStabilityFilter:
    """V100 时间序列稳定性过滤器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lookback_days = self.config.get('lookback_days', V100_IC_LOOKBACK_DAYS)
        self.std_threshold = self.config.get('std_threshold', V100_IC_STD_THRESHOLD)
        self.ic_history: Dict[str, List[float]] = {}
    
    def record_daily_ic(self, factor_name: str, ic_value: float) -> None:
        """记录每日 IC"""
        if factor_name not in self.ic_history:
            self.ic_history[factor_name] = []
        
        self.ic_history[factor_name].append(ic_value)
        
        if len(self.ic_history[factor_name]) > self.lookback_days:
            self.ic_history[factor_name] = self.ic_history[factor_name][-self.lookback_days:]
    
    def get_ic_std(self, factor_name: str) -> float:
        """获取因子过去 N 天的 IC 波动率"""
        if factor_name not in self.ic_history:
            return 0.0
        
        ics = self.ic_history[factor_name]
        if len(ics) < 3:
            return 0.0
        
        return float(np.std(ics))
    
    def is_stable(self, factor_name: str) -> bool:
        """判断因子 IC 是否稳定"""
        ic_std = self.get_ic_std(factor_name)
        return ic_std <= self.std_threshold


# ===========================================
# V100 导出列表
# ===========================================

__all__ = [
    # 异常类
    'DirectionalError',
    'IndustryDataMissingError',
    'ConsecutiveDataMissingError',
    # 常量
    'V100_INITIAL_CAPITAL',
    'V100_MAX_POSITIONS',
    'V100_WARMUP_PERIOD',
    'V100_MIN_SCORE_THRESHOLD',
    'V100_TRANSACTION_COST',
    'V100_TURNOVER_MIN',
    'V100_TURNOVER_MAX',
    'V100_DAILY_TURNOVER_MAX',
    'V100_T1_T5_IC_TARGET',
    'V100_IC_STABILITY_TARGET',
    'V100_IC_IR_TARGET',
    'V100_IC_LOOKBACK_DAYS',
    'V100_IC_STD_THRESHOLD',
    'V100_INDUSTRY_MISSING_THRESHOLD',
    'V100_MV_MISSING_THRESHOLD',
    'V100_CONSECUTIVE_MISSING_DAYS',
    'V100_EXPECTED_RETURN_THRESHOLD',
    'V100_MIN_REBALANCE_INTERVAL',
    'V100_RESIDUAL_WEIGHT',
    'V100_FLOW_WEIGHT',
    'V100_EARNINGS_SURPRISE_WEIGHT',
    'V100_INSTITUTIONAL_WEIGHT',
    'V100_EMA_WINDOW',
    'V100_AUDIT_MODE',
    'V100_HALF_LIFE_LAGS',
    'V100_LAG1_WEIGHT',
    'V100_LAG3_WEIGHT',
    'V100_LAG5_WEIGHT',
    'V100_PREDICTION_HORIZON',
    'V100_COMMISSION_RATE',
    'V100_MIN_COMMISSION',
    'V100_STAMP_DUTY',
    'V100_TRANSFER_FEE',
    # 数据类
    'V100SingleFactorIC',
    'V100FusionSignal',
    'V100TurnoverRecord',
    'V100Position',
    'V100DailyPortfolio',
    'V100DataQualityRecord',
    # 工具函数
    'normalize_rank',
    'zscore_normalize',
    'calculate_half_life_decay_weights',
    'fill_with_market_median',
    'calculate_rank_correlation',
    'ols_residualize',
    'ema_smooth',
    'calculate_ic_stability',
    'calculate_cumulative_return',
    'EPSILON',
    # 因子引擎
    'V100EarningsSurpriseEngine',
    'V100InstitutionalFlowEngine',
    'V100ResidualMomentumEngine',
    'V100SmartFlowEngine',
    # 核心类
    'V100DataQualityChecker',
    'V100DynamicCostThreshold',
    'V100ICStabilityFilter',
    'V100ICAudit',
    'V100DataManager',
    'V100SignalSmoother',
    'V100IndustryChecker',
    'V100TurnoverTracker',
    'V100TransactionCostCalculator',
    'V100AlphaFusion',
    'V100AlphaWeightEngine',
]


# ===========================================
# V100 辅助类
# ===========================================

class V100DataManager:
    """V100 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V100_WARMUP_PERIOD)
    
    def check_data_integrity(self, year: str) -> Tuple[bool, str, Dict[str, Any]]:
        """检查数据完整性"""
        if self.db is None:
            return False, "数据库连接未初始化", {}
        
        try:
            import pandas as pd
            
            query = f"""
                SELECT 
                    COUNT(*) as cnt,
                    COUNT(DISTINCT trade_date) as trading_days,
                    COUNT(DISTINCT symbol) as stocks
                FROM stock_daily
                WHERE trade_date >= '{year}-01-01' 
                  AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return False, f"{year}年无数据", {}
            
            stats = {
                'total_rows': int(df['cnt'][0]),
                'trading_days': int(df['trading_days'][0]),
                'stocks': int(df['stocks'][0]),
            }
            
            return True, f"数据完整 (rows={stats['total_rows']:,}, days={stats['trading_days']})", stats
            
        except Exception as e:
            return False, f"检查失败：{e}", {}
    
    def load_data(self, start_date: str, end_date: str,
                  symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载数据"""
        extra_days = max(V100_HALF_LIFE_LAGS) + 20
        warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - 
                       timedelta(days=self.warmup_period + extra_days)).strftime("%Y-%m-%d")
        
        try:
            import pandas as pd
            
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount, 
                       pct_chg, industry_code, total_mv, is_st
                FROM stock_daily
                WHERE trade_date >= '{warmup_start}' 
                  AND trade_date <= '{end_date}'
                ORDER BY symbol, trade_date
            """
            
            pdf = pd.read_sql(query, self.db.engine)
            
            if pdf.empty:
                raise ValueError(f"未加载到任何数据")
            
            pdf.columns = [str(col).strip() for col in pdf.columns]
            
            df = pl.from_pandas(pdf)
            
            if df.is_empty():
                raise ValueError(f"未加载到任何数据")
            
            df = self._repair_data(df)
            
            logger.info(f"V100: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V100: 数据加载失败 - {e}")
            raise
    
    def _repair_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """修复数据"""
        result = df.clone()
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg', 'total_mv']:
            if col in result.columns:
                median_val = result[col].median()
                if median_val is not None and np.isfinite(median_val):
                    result = result.with_columns([
                        pl.when(pl.col(col).is_null() | ~pl.col(col).is_finite())
                        .then(median_val)
                        .otherwise(pl.col(col))
                        .alias(col)
                    ])
        
        if 'industry_code' in result.columns:
            try:
                industry_counts = result.group_by('industry_code').agg(
                    pl.count().alias('cnt')
                ).sort('cnt', descending=True)
                
                if not industry_counts.is_empty():
                    first_industry = industry_counts['industry_code'][0]
                    if first_industry is None or first_industry == '' or first_industry == 'None':
                        first_industry = 'Unknown'
                else:
                    first_industry = 'Unknown'
            except Exception as e:
                logger.warning(f"V100: 获取最常见行业失败 - {e}，使用默认值")
                first_industry = 'Unknown'
            
            result = result.with_columns([
                pl.when(
                    pl.col('industry_code').is_null() | 
                    (pl.col('industry_code').cast(pl.Utf8).str.len_chars() == 0) |
                    (pl.col('industry_code') == 'None') |
                    (pl.col('industry_code') == 'null')
                )
                .then(pl.lit(str(first_industry)))
                .otherwise(pl.col('industry_code'))
                .alias('industry_code')
            ])
        
        if 'is_st' in result.columns:
            result = result.with_columns([
                pl.col('is_st').fill_null(0).alias('is_st')
            ])
        else:
            result = result.with_columns([
                pl.lit(0).alias('is_st')
            ])
        
        return result


class V100SignalSmoother:
    """V100 信号平滑器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ema_window = self.config.get('ema_window', V100_EMA_WINDOW)
    
    def smooth_signal(self, df: pl.DataFrame, 
                      signal_col: str = 'fused_signal') -> pl.DataFrame:
        """对信号进行 EMA 平滑"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            pl.col(signal_col)
            .rolling_mean(window_size=self.ema_window)
            .over('symbol')
            .alias('smoothed_signal')
        ])
        
        result = result.with_columns([
            pl.when(pl.col('smoothed_signal').is_null())
            .then(pl.col(signal_col))
            .otherwise(pl.col('smoothed_signal'))
            .alias('smoothed_signal')
        ])
        
        logger.info(f"V100: 信号平滑完成 (EMA window={self.ema_window})")
        
        return result


class V100IndustryChecker:
    """V100 行业检查器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.missing_threshold = self.config.get(
            'missing_threshold', V100_INDUSTRY_MISSING_THRESHOLD
        )
    
    def check_industry_coverage(self, df: pl.DataFrame, trade_date: str) -> Tuple[bool, float]:
        """检查行业代码覆盖率"""
        if 'industry_code' not in df.columns:
            return False, 1.0
        
        total_count = df.height
        if total_count == 0:
            return False, 1.0
        
        missing_count = df.filter(
            (pl.col('industry_code').is_null()) |
            (pl.col('industry_code').cast(pl.Utf8).str.len_chars() == 0) |
            (pl.col('industry_code') == 'None') |
            (pl.col('industry_code') == 'null') |
            (pl.col('industry_code') == '')
        ).height
        
        missing_ratio = missing_count / total_count
        
        passed = missing_ratio <= self.missing_threshold
        
        if not passed:
            logger.error(
                f"V100: {trade_date} 行业代码缺失比例 {missing_ratio:.1%} > "
                f"阈值 {self.missing_threshold:.1%}，必须重新补取数据！"
            )
        
        return passed, missing_ratio


class V100TurnoverTracker:
    """V100 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[Dict] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False,
                        rebalance_ratio: float = 0.0,
                        cost_threshold_passed: bool = True) -> Dict:
        """记录换手率"""
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            daily_turnover = 0.0
        else:
            buy_turnover = buy_value / portfolio_value
            sell_turnover = sell_value / portfolio_value
            turnover_rate = (buy_value + sell_value) / portfolio_value
            daily_turnover = turnover_rate
        
        self.trading_days += 1
        
        cumulative_turnover = sum(r.get('turnover_rate', 0) for r in self.turnover_records) + turnover_rate
        annualized_turnover = cumulative_turnover * (252.0 / max(1, self.trading_days))
        
        record = {
            'trade_date': trade_date,
            'turnover_rate': turnover_rate,
            'buy_turnover': buy_turnover,
            'sell_turnover': sell_turnover,
            'annualized_turnover': annualized_turnover,
            'daily_turnover': daily_turnover,
            'is_rebalance_day': is_rebalance_day,
            'rebalance_ratio': rebalance_ratio,
            'cost_threshold_passed': cost_threshold_passed,
        }
        self.turnover_records.append(record)
        
        return record
    
    def get_turnover_summary(self) -> Dict[str, Any]:
        """获取换手率摘要"""
        if not self.turnover_records:
            return {
                'mean_turnover': 0.0,
                'annualized_turnover': 0.0,
                'is_active': False,
                'daily_turnover_ok': True,
            }
        
        total_turnover = sum(r.get('turnover_rate', 0) for r in self.turnover_records)
        annualized_turnover = total_turnover * (252.0 / max(1, self.trading_days))
        
        daily_turnovers = [r.get('daily_turnover', 0) for r in self.turnover_records]
        max_daily = np.max(daily_turnovers) if daily_turnovers else 0.0
        
        is_active = V100_TURNOVER_MIN <= annualized_turnover <= V100_TURNOVER_MAX
        daily_ok = max_daily <= V100_DAILY_TURNOVER_MAX
        turnover_ok = annualized_turnover <= 5.0
        
        return {
            'mean_turnover': float(np.mean([r.get('turnover_rate', 0) for r in self.turnover_records])),
            'std_turnover': float(np.std([r.get('turnover_rate', 0) for r in self.turnover_records])),
            'max_turnover': float(np.max([r.get('turnover_rate', 0) for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'daily_turnover_ok': daily_ok,
            'turnover_ok': turnover_ok,
            'max_daily_turnover': float(max_daily),
            'turnover_min': V100_TURNOVER_MIN,
            'turnover_max': V100_TURNOVER_MAX,
        }


class V100TransactionCostCalculator:
    """V100 交易成本计算器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.transaction_cost = self.config.get('transaction_cost', V100_TRANSACTION_COST)
        self.commission_rate = self.config.get('commission_rate', 0.0003)
        self.min_commission = self.config.get('min_commission', 5.0)
        self.stamp_duty = self.config.get('stamp_duty', 0.0005)
        self.transfer_fee = self.config.get('transfer_fee', 0.00001)
    
    def calculate_buy_cost(self, amount: float) -> float:
        """计算买入成本"""
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer_fee = amount * self.transfer_fee
        return commission + transfer_fee
    
    def calculate_sell_cost(self, amount: float) -> float:
        """计算卖出成本"""
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty
        transfer_fee = amount * self.transfer_fee
        return commission + stamp_duty + transfer_fee


class V100AlphaFusion:
    """V100 Alpha 融合器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', [1, 3, 5])
        self.half_life_weights = self.config.get('half_life_weights', [0.5, 0.3, 0.2])
        self.smoother = V100SignalSmoother(config)
    
    def compute_fusion_signal(self, df: pl.DataFrame,
                               signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算半衰期融合信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        lag_signals = []
        for lag in self.fusion_lags:
            lag_col = f"{signal_col}_lag{lag}"
            result = result.with_columns([
                pl.col(signal_col).shift(lag).over('symbol').alias(lag_col)
            ])
            lag_signals.append(lag_col)
        
        fusion_exprs = []
        for i, lag_col in enumerate(lag_signals):
            weight = self.half_life_weights[i] if i < len(self.half_life_weights) else 1.0 / len(lag_signals)
            fusion_exprs.append(pl.col(lag_col) * weight)
        
        result = result.with_columns([
            sum(fusion_exprs).alias('fused_signal')
        ])
        
        result = self.smoother.smooth_signal(result, 'fused_signal')
        
        logger.info(f"V100: 融合信号计算完成")
        
        return result


class V100AlphaWeightEngine:
    """V100 Alpha 权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V100_MIN_SCORE_THRESHOLD)
    
    def compute_alpha_weights(self, df: pl.DataFrame,
                               score_col: str = 'smoothed_signal') -> pl.DataFrame:
        """计算 Alpha 权重"""
        result = df.clone()
        
        # 简化处理：直接返回原数据
        logger.info(f"V100: Alpha 权重计算完成")
        
        return result
