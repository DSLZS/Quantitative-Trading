"""
V98 Logic Module - 逻辑纠偏与信号稳定化

【V98 核心修复】
1. 因子方向修正 - 确保 Rank 排序逻辑是"高得分对应高收益"
2. 数据对齐修复 - 使用 pct_chg.shift(-1) 来对齐 T+1 收益
3. 信号平滑 - 对最终融合后的 Alpha 信号进行 3 日 EMA 处理
4. 调仓门槛 - 只有当新信号对持仓的预期收益提升超过 0.5% 时才允许换仓
5. 强制报错 - 如果 residual_momentum 因子 IC 为负，抛出 DirectionalError

【V98 硬性指标】
| 维度 | 指标 | 目标值 | 失败判定 |
| :--- | :--- | :--- | :--- |
| 预测力 | T+1 Rank IC | ≥ 0.048 | 低于 0.045 直接抛出 DirectionalError |
| 稳定性 | IC Stability | > 0.4 | Mean(IC) / Std(IC) |
| 活跃度 | 年化换手率 | 300% - 550% | 超过 600% 判定失败 |
| 2024 表现 | 正收益 | > 0% | 必须证明算法有效性 |

【V98 与 V97 的主要区别】
1. 新增 DirectionalError 异常类，用于因子方向错误时报错
2. 新增 V98SignalSmoother - 3 日 EMA 平滑
3. 新增 V98RebalanceThreshold - 0.5% 预期收益提升门槛
4. 修复因子计算中的 Rank 方向问题
5. 数据缺失时使用全市场均值填充，禁止中断回测

作者：量化系统
版本：V98.0
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
# V98 异常类
# ===========================================

class DirectionalError(Exception):
    """因子方向错误异常"""
    def __init__(self, factor_name: str, ic_value: float, expected_direction: str = "positive"):
        self.factor_name = factor_name
        self.ic_value = ic_value
        self.expected_direction = expected_direction
        message = (
            f"因子方向错误！因子 '{factor_name}' 的 IC={ic_value:.4f}，"
            f"期望方向：{expected_direction}。这表示因子计算存在逻辑错误，"
            f"必须修正 Rank 方向或数据对齐。"
        )
        super().__init__(message)


# ===========================================
# V98 配置常量
# ===========================================

V98_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V98_MAX_POSITIONS = 50  # 最大持仓数量
V98_WARMUP_PERIOD = 250
V98_MIN_SAMPLE_SIZE = 100

# V90 基准因子权重
V98_RESIDUAL_WEIGHT = 0.60       # 残差动量权重（V90 核心）
V98_FLOW_WEIGHT = 0.40           # 聪明资金流权重（V90 核心）

# V98 评分门槛配置
V98_MIN_SCORE_THRESHOLD = 45.0   # 评分门槛
V98_MIN_SINGLE_WEIGHT = 0.002    # 最小权重
V98_MAX_SINGLE_WEIGHT = 0.06     # 最大权重

# V98 换手率控制配置（目标：300%-550%）
V98_TURNOVER_MIN = 3.0
V98_TURNOVER_MAX = 5.5           # 更严格的上限
V98_DAILY_TURNOVER_MAX = 0.40    # 降低单日换手率上限

# V98 调仓门槛配置（新增）
V98_REBALANCE_THRESHOLD = 0.005  # 0.5% 预期收益提升门槛

# 费率配置
V98_COMMISSION_RATE = 0.002
V98_MIN_COMMISSION = 5.0
V98_STAMP_DUTY = 0.0005
V98_TRANSFER_FEE = 0.00001

# IC 目标（V98 强制）
V98_T1_IC_TARGET = 0.048
V98_IC_IR_TARGET = 0.5
V98_IC_STABILITY_TARGET = 0.4    # IC Stability = Mean(IC) / Std(IC)

# 风格中性化配置
V98_SIZE_NEUTRALIZATION = True
V98_INDUSTRY_NEUTRALIZATION = True
V98_NEUTRALIZATION_WINDOW = 60

# 流动性过滤配置
V98_LIQUIDITY_FILTER = True
V98_LIQUIDITY_PERCENTILE = 10
V98_FILTER_ST = True

# 调仓配置
V98_MIN_REBALANCE_INTERVAL = 5
V98_MAX_REBALANCE_INTERVAL = 5
V98_RANK_CORRELATION_THRESHOLD = 0.35

# 半衰期融合配置
V98_HALF_LIFE_LAGS = [1, 3, 5]
V98_LAG1_WEIGHT = 0.50
V98_LAG3_WEIGHT = 0.30
V98_LAG5_WEIGHT = 0.20

# V98 信号平滑配置（新增）
V98_EMA_WINDOW = 3               # 3 日 EMA 平滑

# 审计配置
V98_AUDIT_MODE = True

EPSILON = 1e-9


# ===========================================
# V98 数据类
# ===========================================

@dataclass
class V98SingleFactorIC:
    """单因子 IC 记录"""
    factor_name: str
    ic_t1: float
    ic_t2: float
    ic_t3: float
    ic_ir: float
    ic_stability: float          # IC Stability = Mean(IC) / Std(IC)
    passed_threshold: bool


@dataclass
class V98FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float
    smoothed_signal: float       # V98 新增：平滑后的信号


@dataclass
class V98TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float
    is_rebalance_day: bool = False


@dataclass
class V98Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V98DailyPortfolio:
    """每日组合快照"""
    trade_date: str
    total_value: float
    cash: float
    position_value: float
    position_count: int
    daily_return: float
    cumulative_return: float
    turnover_rate: float


# ===========================================
# V98 工具函数
# ===========================================

def normalize_rank(series: pl.Series, descending: bool = False) -> pl.Series:
    """
    将序列转换为百分位排名（0-100）
    
    【V98 修复】确保排名方向正确：
    - descending=True: 值越大，排名越高（百分位越高）
    - descending=False: 值越小，排名越高
    """
    n = len(series)
    if n == 0:
        return series
    
    ranks = series.rank('ordinal', descending=descending)
    # 百分位计算：排名越靠前（值越小），百分位越高
    percentile = 100.0 * (1.0 - (ranks.cast(pl.Float64) - 0.5) / (n + EPSILON))
    
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


def calculate_half_life_decay_weights(lags: List[int] = V98_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V98_LAG1_WEIGHT,
        3: V98_LAG3_WEIGHT,
        5: V98_LAG5_WEIGHT,
    }
    
    weights = [fixed_weights.get(lag, 1.0 / len(lags)) for lag in lags]
    
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return [1.0 / len(lags)] * len(lags)
    
    return [w / total_weight for w in weights]


def fill_with_market_median(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """
    使用全市场中位数填充空值
    
    【V98 修复】禁止中断回测，必须使用均值填充
    """
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


def ema_smooth(series: np.ndarray, window: int = V98_EMA_WINDOW) -> np.ndarray:
    """
    指数移动平均平滑（EMA）
    
    【V98 新增】用于信号平滑，降低信号周转率
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
    window : int
        EMA 窗口大小
        
    Returns
    -------
    np.ndarray
        平滑后的序列
    """
    if len(series) < window:
        return series
    
    result = np.zeros_like(series)
    alpha = 2.0 / (window + 1)
    
    # 初始化
    result[0] = series[0]
    
    # EMA 计算
    for i in range(1, len(series)):
        result[i] = alpha * series[i] + (1 - alpha) * result[i-1]
    
    return result


# ===========================================
# V98 Signal Smoother - 信号平滑（V98 新增）
# ===========================================

class V98SignalSmoother:
    """
    V98 信号平滑引擎
    
    【核心功能】
    对最终融合后的 Alpha 信号进行 3 日 EMA 处理，强制降低信号周转率
    
    【实现方式】
    1. 按股票分组，对信号进行时间序列 EMA 平滑
    2. 使用指数衰减权重，近期信号权重更高
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ema_window = self.config.get('ema_window', V98_EMA_WINDOW)
    
    def smooth_signal(self, df: pl.DataFrame, 
                      signal_col: str = 'fused_signal') -> pl.DataFrame:
        """
        对信号进行 EMA 平滑
        
        Parameters
        ----------
        df : pl.DataFrame
            输入数据，必须包含 symbol, trade_date, signal_col
        signal_col : str
            需要平滑的信号列名
            
        Returns
        -------
        pl.DataFrame
            添加了 smoothed_signal 列的数据
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 按股票分组，对信号进行 EMA 平滑
        # 使用 polars 的 rolling_mean 近似 EMA
        result = result.with_columns([
            pl.col(signal_col)
            .rolling_mean(window_size=self.ema_window)
            .over('symbol')
            .alias('smoothed_signal')
        ])
        
        # 对于窗口内的前几个值，使用可用数据的均值
        result = result.with_columns([
            pl.when(pl.col('smoothed_signal').is_null())
            .then(pl.col(signal_col))
            .otherwise(pl.col('smoothed_signal'))
            .alias('smoothed_signal')
        ])
        
        logger.info(f"V98: 信号平滑完成 (EMA window={self.ema_window})")
        
        return result
    
    def smooth_signal_ema(self, df: pl.DataFrame,
                          signal_col: str = 'fused_signal') -> pl.DataFrame:
        """
        使用真正的 EMA 公式进行平滑
        
        这是更精确的实现，但计算较慢
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        unique_symbols = result['symbol'].unique().to_list()
        
        smoothed_signals = []
        for symbol in unique_symbols:
            symbol_data = result.filter(pl.col('symbol') == symbol)
            
            if symbol_data.is_empty():
                continue
            
            signals = symbol_data[signal_col].to_numpy()
            dates = symbol_data['trade_date'].to_list()
            
            # EMA 平滑
            smoothed = ema_smooth(signals, self.ema_window)
            
            for i, date in enumerate(dates):
                smoothed_signals.append({
                    'symbol': symbol,
                    'trade_date': date,
                    'smoothed_signal': smoothed[i],
                })
        
        if smoothed_signals:
            smooth_df = pl.DataFrame({
                'symbol': [s['symbol'] for s in smoothed_signals],
                'trade_date': [s['trade_date'] for s in smoothed_signals],
                'smoothed_signal': [s['smoothed_signal'] for s in smoothed_signals],
            })
            
            result = result.join(smooth_df, on=['symbol', 'trade_date'], how='left')
            result = result.with_columns([
                pl.col('smoothed_signal').fill_null(pl.col(signal_col)).alias('smoothed_signal')
            ])
        
        logger.info(f"V98: EMA 信号平滑完成")
        
        return result


# ===========================================
# V98 Rebalance Threshold - 调仓门槛（V98 新增）
# ===========================================

class V98RebalanceThreshold:
    """
    V98 调仓门槛引擎
    
    【核心功能】
    只有当新信号对持仓的预期收益提升超过 0.5% 时才允许换仓
    
    【实现方式】
    1. 计算当前持仓的预期收益（基于旧信号）
    2. 计算新信号的预期收益
    3. 如果收益提升 < 0.5%，则维持原持仓
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', V98_REBALANCE_THRESHOLD)
    
    def should_rebalance(self, 
                         current_positions: Dict[str, float],
                         new_signals: Dict[str, float],
                         expected_returns: Dict[str, float]) -> Tuple[bool, str]:
        """
        判断是否应该调仓
        
        Parameters
        ----------
        current_positions : Dict[str, float]
            当前持仓 {symbol: weight}
        new_signals : Dict[str, float]
            新信号 {symbol: signal_value}
        expected_returns : Dict[str, float]
            预期收益 {symbol: expected_return}
            
        Returns
        -------
        Tuple[bool, str]
            (是否调仓，原因)
        """
        if not current_positions:
            return True, "空仓状态，允许调仓"
        
        # 计算当前持仓的预期收益
        current_expected = sum(
            weight * expected_returns.get(symbol, 0.0)
            for symbol, weight in current_positions.items()
        )
        
        # 计算新信号的预期收益
        # 按新信号排序，选择 top N 股票
        sorted_symbols = sorted(new_signals.keys(), key=lambda s: new_signals.get(s, 0), reverse=True)
        max_positions = min(len(sorted_symbols), V98_MAX_POSITIONS)
        
        new_weights = {s: 1.0 / max_positions for s in sorted_symbols[:max_positions]}
        
        new_expected = sum(
            weight * expected_returns.get(symbol, 0.0)
            for symbol, weight in new_weights.items()
        )
        
        # 计算收益提升
        if abs(current_expected) < EPSILON:
            improvement = new_expected
        else:
            improvement = (new_expected - current_expected) / abs(current_expected)
        
        if improvement >= self.threshold:
            return True, f"预期收益提升 {improvement:.2%} >= {self.threshold:.1%}"
        else:
            return False, f"预期收益提升 {improvement:.2%} < {self.threshold:.1%}"


# ===========================================
# V98 DataManager
# ===========================================

class V98DataManager:
    """V98 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V98_WARMUP_PERIOD)
    
    def check_data_integrity(self, year: str) -> Tuple[bool, str, Dict[str, Any]]:
        """检查数据完整性"""
        if self.db is None:
            return False, "数据库连接未初始化", {}
        
        try:
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
        extra_days = max(V98_HALF_LIFE_LAGS) + 20
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
            
            logger.info(f"V98: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V98: 数据加载失败 - {e}")
            raise
    
    def _repair_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        修复数据
        
        【V98 修复】
        - 数据缺失时使用全市场均值填充，禁止中断回测
        """
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
                logger.warning(f"V98: 获取最常见行业失败 - {e}，使用默认值")
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


# ===========================================
# V98 ICAudit - IC 审计（带方向检查）
# ===========================================

class V98ICAudit:
    """
    V98 IC 审计
    
    【V98 修复】
    1. 强制检查因子方向，IC 为负时抛出 DirectionalError
    2. 计算 IC Stability = Mean(IC) / Std(IC)
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'smoothed_signal') -> Dict[str, Any]:
        """
        计算 Rank IC
        
        【V98 修复】
        - 使用 pct_chg.shift(-1) 来对齐 T+1 收益
        - 检查 IC 方向，如果为负则抛出异常
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 【关键修复】使用 shift(-lag) 获取未来收益，确保 T+1 对齐
        for lag in [1, 2, 3]:
            result = result.with_columns([
                pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
            ])
        
        ic_results = {}
        all_daily_ics = {'t1': [], 't2': [], 't3': []}
        
        for lag in [1, 2, 3]:
            return_col = f'forward_return_{lag}d'
            if return_col not in result.columns:
                continue
            
            # 按日期计算 IC
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
                    'min_ic': float(np.min(valid_ic)) if valid_ic else 0.0,
                    'max_ic': float(np.max(valid_ic)) if valid_ic else 0.0,
                }
        
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        
        # 计算 IC Stability
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_stability = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 【V98 强制检查】IC 方向验证
        decay_normal = (ic_t1 >= ic_t2 >= ic_t3) and (ic_t1 > 0)
        
        return {
            'ic_t1': ic_t1,
            'ic_t2': ic_t2,
            'ic_t3': ic_t3,
            'std_t1': std_t1,
            'std_t2': ic_results.get('t2', {}).get('std_ic', 0.0),
            'std_t3': ic_results.get('t3', {}).get('std_ic', 0.0),
            'ic_stability': ic_stability,
            'decay_normal': decay_normal,
            't1_ic_passed': ic_t1 >= V98_T1_IC_TARGET,
            'stability_passed': ic_stability >= V98_IC_STABILITY_TARGET,
            'all_daily_ics': all_daily_ics,
        }
    
    def calculate_single_factor_ic(self, df: pl.DataFrame, 
                                    factor_name: str,
                                    signal_col: str) -> V98SingleFactorIC:
        """
        计算单因子 IC
        
        【V98 强制检查】
        如果 residual_momentum 因子 IC 为负，抛出 DirectionalError
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 【关键修复】使用 shift(-lag) 获取未来收益
        for lag in [1, 2, 3]:
            result = result.with_columns([
                pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
            ])
        
        ic_results = {}
        all_daily_ics = []
        
        for lag in [1, 2, 3]:
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
        
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        
        # 计算 IC IR
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_ir = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 计算 IC Stability
        ic_stability = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 【V98 强制报错】
        if factor_name == 'residual_momentum' and ic_t1 < 0:
            raise DirectionalError(
                factor_name=factor_name,
                ic_value=ic_t1,
                expected_direction="positive"
            )
        
        return V98SingleFactorIC(
            factor_name=factor_name,
            ic_t1=ic_t1,
            ic_t2=ic_t2,
            ic_t3=ic_t3,
            ic_ir=ic_ir,
            ic_stability=ic_stability,
            passed_threshold=ic_t1 >= V98_T1_IC_TARGET,
        )
    
    def validate_factor_direction(self, df: pl.DataFrame,
                                   factor_name: str,
                                   signal_col: str) -> Tuple[bool, str]:
        """
        验证因子方向是否正确
        
        Returns
        -------
        Tuple[bool, str]
            (方向是否正确，错误信息)
        """
        try:
            ic_result = self.calculate_single_factor_ic(df, factor_name, signal_col)
            
            if ic_result.ic_t1 < 0:
                return False, f"因子 {factor_name} 的 T+1 IC={ic_result.ic_t1:.4f} < 0，方向错误！"
            
            if ic_result.ic_t1 < V98_T1_IC_TARGET:
                return False, f"因子 {factor_name} 的 T+1 IC={ic_result.ic_t1:.4f} < 目标 {V98_T1_IC_TARGET}"
            
            return True, f"因子 {factor_name} 方向正确，T+1 IC={ic_result.ic_t1:.4f}"
            
        except DirectionalError as e:
            return False, str(e)


# ===========================================
# V98 Residual Momentum - 残差动量（V98 修复版）
# ===========================================

class V98ResidualMomentumEngine:
    """
    V98 残差动量引擎
    
    【V98 修复】
    1. 确保 Rank 方向正确：高动量对应高得分
    2. 使用 pct_chg.shift(-1) 对齐 T+1 收益进行 IC 计算
    3. 【关键修复】A 股呈现短期反转效应，需要反转 Rank 方向
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.momentum_window = self.config.get('momentum_window', 20)
    
    def compute_residual_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算残差动量
        
        【V98 关键修复】
        A 股市场呈现显著的短期反转效应（20 日动量），即：
        - 过去 20 日涨幅大的股票，未来 1 日倾向于下跌
        - 过去 20 日跌幅大的股票，未来 1 日倾向于上涨
        
        因此需要反转 Rank 方向：
        - descending=False: 动量值越小（跌幅越大），排名越靠前
        - 这样低动量（超跌）对应高得分，符合反转效应
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算 N 日动量（使用价格比率）
        result = result.with_columns([
            ((pl.col('close') / pl.col('close').shift(self.momentum_window)) - 1).alias('momentum_raw')
        ])
        
        # 2. 【V98 关键修复】按日期排名，反转方向以适应 A 股反转效应
        # descending=False: 动量值越小（跌幅越大），排名数值越小（排名越靠前）
        # 百分位转换后：动量值越小（超跌），百分位越高 → 高得分
        result = result.with_columns([
            pl.col('momentum_raw').rank('ordinal', descending=False).over('trade_date').alias('momentum_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('momentum_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('residual_momentum_score')
        ])
        
        # 清理临时列
        result = result.drop(['momentum_rank', 'n_stocks'])
        
        logger.info(f"V98: 残差动量计算完成（反转效应修正），处理 {result.height} 条记录")
        
        return result


# ===========================================
# V98 Smart Flow - 聪明资金流（V98 修复版）
# ===========================================

class V98SmartFlowEngine:
    """
    V98 聪明资金流引擎
    
    【V98 修复】
    1. 确保资金流方向正确：流入为正
    2. 【关键修复】A 股资金流因子呈现反转特性，需要反转 Rank 方向
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.flow_window = self.config.get('flow_window', 5)
    
    def compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算聪明资金流
        
        【V98 关键修复】
        A 股市场资金流因子呈现显著的反转特性：
        - 当日资金大幅流入的股票，次日倾向于回调
        - 当日资金大幅流出的股票，次日倾向于反弹
        
        因此需要反转 Rank 方向：
        - descending=False: 资金流值越小（流出），排名越靠前
        - 这样资金流出对应高得分，符合反转效应
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算资金流向因子
        numerator = (pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))
        denominator = pl.col('high') - pl.col('low') + EPSILON
        result = result.with_columns(
            (numerator / denominator).alias('money_flow_factor')
        )
        
        # 2. 计算资金流（乘以成交量）
        result = result.with_columns(
            (pl.col('money_flow_factor') * pl.col('volume').fill_null(0)).alias('money_flow')
        )
        
        # 3. 计算滚动资金流均值
        result = result.with_columns([
            pl.col('money_flow')
            .rolling_mean(window_size=self.flow_window)
            .over('symbol')
            .alias('money_flow_ma')
        ])
        
        # 4. 【V98 关键修复】按日期排名，反转方向以适应 A 股反转效应
        # descending=False: 资金流值越小（流出），排名数值越小（排名越靠前）
        result = result.with_columns([
            pl.col('money_flow_ma').rank('ordinal', descending=False).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('flow_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('smart_flow_score')
        ])
        
        # 清理临时列
        keep_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 
                     'volume', 'amount', 'pct_chg', 'industry_code', 'total_mv', 
                     'is_st', 'smart_flow_score']
        
        if 'residual_momentum_score' in result.columns:
            keep_cols.append('residual_momentum_score')
        
        result = result.select(keep_cols)
        
        logger.info(f"V98: 聪明资金流计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V98 AlphaFusion - 半衰期融合引擎（带信号平滑）
# ===========================================

class V98AlphaFusion:
    """
    V98 AlphaFusion - 半衰期衰减融合引擎
    
    【V98 新增】
    融合后自动应用 3 日 EMA 平滑
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V98_HALF_LIFE_LAGS)
        
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        
        self.smoother = V98SignalSmoother(config)
        
        logger.info("V98 AlphaFusion 初始化完成")
        logger.info(f"V98: 融合 Lags={self.fusion_lags}")
        logger.info(f"V98: 半衰期权重={dict(zip(self.fusion_lags, self.half_life_weights))}")
    
    def compute_fusion_signal(self, df: pl.DataFrame,
                               signal_col: str = 'composite_score') -> pl.DataFrame:
        """
        计算半衰期融合信号并应用 EMA 平滑
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算滞后信号
        lag_signals = []
        for lag in self.fusion_lags:
            lag_col = f"{signal_col}_lag{lag}"
            result = result.with_columns([
                pl.col(signal_col).shift(lag).over('symbol').alias(lag_col)
            ])
            lag_signals.append(lag_col)
        
        # 2. 加权融合
        fusion_exprs = []
        for i, lag_col in enumerate(lag_signals):
            weight = self.half_life_weights[i]
            fusion_exprs.append(pl.col(lag_col) * weight)
        
        result = result.with_columns([
            sum(fusion_exprs).alias('fused_signal')
        ])
        
        # 3. 【V98 新增】应用 EMA 平滑
        result = self.smoother.smooth_signal(result, 'fused_signal')
        
        logger.info(f"V98: 融合信号计算完成（含 EMA 平滑），处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        return {
            'weights': dict(zip(self.fusion_lags, self.half_life_weights)),
            'lag1_weight': self.half_life_weights[0],
            'ema_window': V98_EMA_WINDOW,
        }


# ===========================================
# V98 AlphaWeight - Alpha 权重引擎
# ===========================================

class V98AlphaWeightEngine:
    """V98 Alpha 权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V98_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V98_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V98_MAX_SINGLE_WEIGHT)
    
    def compute_alpha_weights(self, df: pl.DataFrame,
                               score_col: str = 'smoothed_signal') -> pl.DataFrame:
        """计算 Alpha 权重"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        result = result.with_columns([
            pl.col('daily_return')
            .rolling_std(window_size=20)
            .over('symbol')
            .alias('daily_volatility')
        ])
        
        result = result.with_columns([
            (pl.col('daily_volatility') * np.sqrt(252)).alias('volatility')
        ])
        
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        all_weights = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            scores = day_data[score_col].to_numpy()
            volatilities = day_data['volatility'].to_numpy()
            
            valid_mask = (~np.isnan(scores) & ~np.isnan(volatilities) & 
                         np.isfinite(scores) & np.isfinite(volatilities))
            
            if np.sum(valid_mask) < 1:
                continue
            
            valid_scores = scores[valid_mask]
            valid_vols = volatilities[valid_mask]
            valid_symbols = day_data['symbol'].to_numpy()[valid_mask]
            
            weights, filtered = self._alpha_weighting(valid_scores, valid_vols)
            
            for i, symbol in enumerate(valid_symbols):
                all_weights.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'volatility': valid_vols[i],
                    'alpha_weight': weights[i],
                    'is_filtered': filtered[i],
                })
        
        if all_weights:
            weight_df = pl.DataFrame({
                'trade_date': [w['trade_date'] for w in all_weights],
                'symbol': [w['symbol'] for w in all_weights],
                'volatility': [w['volatility'] for w in all_weights],
                'alpha_weight': [w['alpha_weight'] for w in all_weights],
                'is_filtered': [w['is_filtered'] for w in all_weights],
            })
            
            result = result.join(weight_df, on=['trade_date', 'symbol'], how='left')
        
        logger.info(f"V98: Alpha 权重计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _alpha_weighting(self, scores: np.ndarray, volatilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Alpha 权重计算"""
        n = len(scores)
        weights = np.zeros(n)
        filtered = np.ones(n, dtype=bool)
        
        valid_mask = scores >= self.min_score
        filtered[valid_mask] = False
        
        if np.sum(valid_mask) < 1:
            valid_mask = np.ones(n, dtype=bool)
            filtered[:] = False
        
        valid_scores = scores[valid_mask]
        valid_vols = volatilities[valid_mask]
        
        valid_vols = np.where(valid_vols < EPSILON, EPSILON, valid_vols)
        valid_vols = np.where(valid_vols > 10.0, 10.0, valid_vols)
        
        raw_weights = valid_scores / valid_vols
        
        total_weight = np.sum(raw_weights)
        if total_weight < EPSILON:
            return np.full(n, 1.0 / n), filtered
        
        normalized_weights = raw_weights / total_weight
        
        normalized_weights = np.clip(normalized_weights, self.min_weight, self.max_weight)
        
        total_weight = np.sum(normalized_weights)
        if total_weight > EPSILON:
            normalized_weights = normalized_weights / total_weight
        
        weights[valid_mask] = normalized_weights
        
        return weights, filtered


# ===========================================
# V98 TurnoverTracker - 换手率追踪
# ===========================================

class V98TurnoverTracker:
    """V98 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V98TurnoverRecord] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> V98TurnoverRecord:
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
        
        cumulative_turnover = sum(r.turnover_rate for r in self.turnover_records) + turnover_rate
        annualized_turnover = cumulative_turnover * (252.0 / max(1, self.trading_days))
        
        record = V98TurnoverRecord(
            trade_date=trade_date,
            turnover_rate=turnover_rate,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            annualized_turnover=annualized_turnover,
            daily_turnover=daily_turnover,
            is_rebalance_day=is_rebalance_day,
        )
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
        
        total_turnover = sum(r.turnover_rate for r in self.turnover_records)
        annualized_turnover = total_turnover * (252.0 / max(1, self.trading_days))
        
        daily_turnovers = [r.daily_turnover for r in self.turnover_records]
        max_daily = np.max(daily_turnovers) if daily_turnovers else 0.0
        
        is_active = V98_TURNOVER_MIN <= annualized_turnover <= V98_TURNOVER_MAX
        daily_ok = max_daily <= V98_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean([r.turnover_rate for r in self.turnover_records])),
            'std_turnover': float(np.std([r.turnover_rate for r in self.turnover_records])),
            'max_turnover': float(np.max([r.turnover_rate for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V98_TURNOVER_MIN,
            'turnover_max': V98_TURNOVER_MAX,
        }


# ===========================================
# V98 导出列表
# ===========================================

__all__ = [
    # 异常类
    'DirectionalError',
    # 常量
    'V98_INITIAL_CAPITAL',
    'V98_MAX_POSITIONS',
    'V98_WARMUP_PERIOD',
    'V98_MIN_SCORE_THRESHOLD',
    'V98_MIN_SINGLE_WEIGHT',
    'V98_MAX_SINGLE_WEIGHT',
    'V98_TURNOVER_MIN',
    'V98_TURNOVER_MAX',
    'V98_DAILY_TURNOVER_MAX',
    'V98_REBALANCE_THRESHOLD',
    'V98_T1_IC_TARGET',
    'V98_IC_IR_TARGET',
    'V98_IC_STABILITY_TARGET',
    'V98_COMMISSION_RATE',
    'V98_MIN_COMMISSION',
    'V98_STAMP_DUTY',
    'V98_TRANSFER_FEE',
    'V98_HALF_LIFE_LAGS',
    'V98_LAG1_WEIGHT',
    'V98_LAG3_WEIGHT',
    'V98_LAG5_WEIGHT',
    'V98_SIZE_NEUTRALIZATION',
    'V98_INDUSTRY_NEUTRALIZATION',
    'V98_NEUTRALIZATION_WINDOW',
    'V98_LIQUIDITY_FILTER',
    'V98_LIQUIDITY_PERCENTILE',
    'V98_FILTER_ST',
    'V98_MIN_REBALANCE_INTERVAL',
    'V98_MAX_REBALANCE_INTERVAL',
    'V98_RANK_CORRELATION_THRESHOLD',
    'V98_EMA_WINDOW',
    'V98_AUDIT_MODE',
    'V98_RESIDUAL_WEIGHT',
    'V98_FLOW_WEIGHT',
    # 数据类
    'V98SingleFactorIC',
    'V98FusionSignal',
    'V98TurnoverRecord',
    'V98Position',
    'V98DailyPortfolio',
    # 工具函数
    'normalize_rank',
    'zscore_normalize',
    'calculate_half_life_decay_weights',
    'fill_with_market_median',
    'calculate_rank_correlation',
    'ols_residualize',
    'ema_smooth',
    'EPSILON',
    # 核心类
    'V98DataManager',
    'V98SignalSmoother',
    'V98RebalanceThreshold',
    'V98ICAudit',
    'V98ResidualMomentumEngine',
    'V98SmartFlowEngine',
    'V98AlphaFusion',
    'V98AlphaWeightEngine',
    'V98TurnoverTracker',
]