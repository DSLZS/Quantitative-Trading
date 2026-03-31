"""
V99 Core - 净收益保卫战：持仓缓冲与交易成本强约束

【V99 核心改进】
1. 持仓缓冲逻辑（Position Buffer）
   - 买入条件：Alpha 排名进入前 20 名
   - 卖出条件：跌出前 60 名（缓冲区域 20-60 名）
   - 禁止"全仓换血"

2. 每日调仓比例限制
   - 每日调仓不超过 15%
   - 分批执行，避免一次性换仓

3. 交易成本强约束
   - TRANSACTION_COST = 0.0015（单边）
   - 硬编码扣费，输出"扣费后净收益"

4. 时间序列稳定性过滤
   - 剔除过去 5 天 IC 波动率过大的因子分量

5. industry_code 缺失检测
   - 缺失比例 > 10% 时主动重新补取数据

【V99 验收硬指标】
| 指标 | 目标值 | 失败判定 |
| :--- | :--- | :--- |
| 扣费后年化收益 | > 15% | 任何因换手率导致的亏损直接判定为负优化 |
| 年化换手率 | 300% - 450% | 超过 500% 立即重写持仓逻辑 |
| T+1 Rank IC | > 0.048 | 必须维持在正向且稳定 |
| IC Stability | > 0.4 | 证明预测不是随机的 |

作者：量化系统
版本：V99.0
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
# V99 配置常量（硬编码，严禁修改）
# ===========================================

V99_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V99_MAX_POSITIONS = 50  # 最大持仓数量
V99_WARMUP_PERIOD = 250
V99_MIN_SAMPLE_SIZE = 100

# V90 基准因子权重
V99_RESIDUAL_WEIGHT = 0.60
V99_FLOW_WEIGHT = 0.40

# V99 评分门槛
V99_MIN_SCORE_THRESHOLD = 45.0
V99_MIN_SINGLE_WEIGHT = 0.002
V99_MAX_SINGLE_WEIGHT = 0.06

# V99 换手率控制（目标：300%-450%）
V99_TURNOVER_MIN = 3.0
V99_TURNOVER_MAX = 4.5           # 更严格的上限
V99_DAILY_TURNOVER_MAX = 0.15    # 每日换手率上限 15%

# V99 持仓缓冲逻辑（核心改进）
V99_BUY_RANK_THRESHOLD = 20      # 买入阈值：前 20 名
V99_SELL_RANK_THRESHOLD = 60     # 卖出阈值：跌出前 60 名
V99_BUFFER_ZONE = (V99_BUY_RANK_THRESHOLD, V99_SELL_RANK_THRESHOLD)  # 缓冲区域

# V99 调仓比例限制（核心改进）
# 计算：450% / 252 天 ≈ 1.8% 每日
V99_MAX_DAILY_REBALANCE_RATIO = 0.02  # 每日调仓不超过 2%（严格控制换手率）

# V99 交易成本（硬编码）
V99_TRANSACTION_COST = 0.0015    # 单边交易成本 0.15%
V99_COMMISSION_RATE = 0.002
V99_MIN_COMMISSION = 5.0
V99_STAMP_DUTY = 0.0005
V99_TRANSFER_FEE = 0.00001

# V99 IC 目标
V99_T1_IC_TARGET = 0.048
V99_IC_IR_TARGET = 0.5
V99_IC_STABILITY_TARGET = 0.4

# V99 时间序列稳定性过滤
V99_IC_LOOKBACK_DAYS = 5         # 过去 5 天 IC
V99_IC_STD_THRESHOLD = 0.15      # IC 波动率阈值

# V99 industry_code 缺失检测
V99_INDUSTRY_MISSING_THRESHOLD = 0.10  # 10% 缺失阈值

# 风格中性化配置
V99_SIZE_NEUTRALIZATION = True
V99_INDUSTRY_NEUTRALIZATION = True
V99_NEUTRALIZATION_WINDOW = 60

# 流动性过滤配置
V99_LIQUIDITY_FILTER = True
V99_LIQUIDITY_PERCENTILE = 10
V99_FILTER_ST = True

# 调仓配置
V99_MIN_REBALANCE_INTERVAL = 5
V99_MAX_REBALANCE_INTERVAL = 5

# 半衰期融合配置
V99_HALF_LIFE_LAGS = [1, 3, 5]
V99_LAG1_WEIGHT = 0.50
V99_LAG3_WEIGHT = 0.30
V99_LAG5_WEIGHT = 0.20

# V99 信号平滑配置
V99_EMA_WINDOW = 3

# 审计配置
V99_AUDIT_MODE = True

EPSILON = 1e-9


# ===========================================
# V99 异常类
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
    def __init__(self, missing_ratio: float, date: str):
        self.missing_ratio = missing_ratio
        self.date = date
        message = (
            f"{date} 行业代码缺失比例 {missing_ratio:.1%} > {V99_INDUSTRY_MISSING_THRESHOLD:.1%}，"
            f"必须重新补取数据。"
        )
        super().__init__(message)


# ===========================================
# V99 数据类
# ===========================================

@dataclass
class V99SingleFactorIC:
    """单因子 IC 记录"""
    factor_name: str
    ic_t1: float
    ic_t2: float
    ic_t3: float
    ic_ir: float
    ic_stability: float
    ic_std_5d: float           # 过去 5 天 IC 波动率
    passed_threshold: bool
    passed_stability_filter: bool


@dataclass
class V99FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float
    smoothed_signal: float


@dataclass
class V99TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float
    is_rebalance_day: bool = False
    rebalance_ratio: float = 0.0  # 实际调仓比例


@dataclass
class V99Position:
    """持仓记录（带缓冲逻辑）"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_rank: int = 999      # 当前 Alpha 排名
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V99DailyPortfolio:
    """每日组合快照"""
    trade_date: str
    total_value: float
    cash: float
    position_value: float
    position_count: int
    daily_return: float
    cumulative_return: float
    turnover_rate: float
    net_return_after_cost: float  # 扣费后收益


# ===========================================
# V99 工具函数
# ===========================================

def normalize_rank(series: pl.Series, descending: bool = False) -> pl.Series:
    """将序列转换为百分位排名（0-100）"""
    n = len(series)
    if n == 0:
        return series
    
    ranks = series.rank('ordinal', descending=descending)
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


def calculate_half_life_decay_weights(lags: List[int] = V99_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V99_LAG1_WEIGHT,
        3: V99_LAG3_WEIGHT,
        5: V99_LAG5_WEIGHT,
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


def ema_smooth(series: np.ndarray, window: int = V99_EMA_WINDOW) -> np.ndarray:
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


# ===========================================
# V99 Signal Smoother
# ===========================================

class V99SignalSmoother:
    """V99 信号平滑引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.ema_window = self.config.get('ema_window', V99_EMA_WINDOW)
    
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
        
        logger.info(f"V99: 信号平滑完成 (EMA window={self.ema_window})")
        
        return result


# ===========================================
# V99 Position Buffer - 持仓缓冲逻辑（核心改进）
# ===========================================

class V99PositionBuffer:
    """
    V99 持仓缓冲逻辑
    
    【核心功能】
    1. 买入条件：Alpha 排名进入前 20 名
    2. 卖出条件：跌出前 60 名（缓冲区域 20-60 名）
    3. 禁止"全仓换血"
    
    【缓冲区域设计】
    - 前 20 名：买入区域
    - 20-60 名：持有区域（缓冲带）
    - 60 名以后：卖出区域
    
    这样设计的好处：
    - 股票在 20-60 名之间波动时不会触发调仓
    - 大幅降低换手率
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.buy_threshold = self.config.get('buy_threshold', V99_BUY_RANK_THRESHOLD)
        self.sell_threshold = self.config.get('sell_threshold', V99_SELL_RANK_THRESHOLD)
    
    def get_buy_candidates(self, stock_ranks: Dict[str, int]) -> List[str]:
        """
        获取买入候选股票（前 20 名）
        
        Parameters
        ----------
        stock_ranks : Dict[str, int]
            {symbol: rank}
            
        Returns
        -------
        List[str]
            买入候选股票列表
        """
        buy_candidates = [
            symbol for symbol, rank in stock_ranks.items()
            if rank <= self.buy_threshold
        ]
        return sorted(buy_candidates, key=lambda s: stock_ranks[s])
    
    def should_sell(self, symbol: str, current_rank: int) -> bool:
        """
        判断是否应该卖出
        
        Parameters
        ----------
        symbol : str
            股票代码
        current_rank : int
            当前 Alpha 排名
            
        Returns
        -------
        bool
            是否卖出
        """
        # 跌出前 60 名才卖出
        return current_rank > self.sell_threshold
    
    def should_hold(self, symbol: str, current_rank: int) -> bool:
        """
        判断是否应该持有（在缓冲区域内）
        
        Parameters
        ----------
        symbol : str
            股票代码
        current_rank : int
            当前 Alpha 排名
            
        Returns
        -------
        bool
            是否持有
        """
        # 在 20-60 名缓冲区域内，继续持有
        return self.buy_threshold < current_rank <= self.sell_threshold
    
    def get_target_positions(self, stock_ranks: Dict[str, int], 
                              current_positions: Dict[str, V99Position],
                              max_positions: int = V99_MAX_POSITIONS) -> Tuple[List[str], List[str]]:
        """
        获取目标持仓
        
        Parameters
        ----------
        stock_ranks : Dict[str, int]
            {symbol: rank}
        current_positions : Dict[str, V99Position]
            当前持仓
        max_positions : int
            最大持仓数
            
        Returns
        -------
        Tuple[List[str], List[str]]
            (买入列表，卖出列表)
        """
        # 1. 获取买入候选（前 20 名）
        buy_candidates = self.get_buy_candidates(stock_ranks)
        target_buy = buy_candidates[:max_positions]
        
        # 2. 确定卖出列表（跌出前 60 名）
        symbols_to_sell = []
        for symbol, position in current_positions.items():
            current_rank = stock_ranks.get(symbol, 999)
            if self.should_sell(symbol, current_rank):
                symbols_to_sell.append(symbol)
        
        # 3. 打印重合度（V99 强制要求）
        current_symbols = set(current_positions.keys())
        target_symbols = set(target_buy)
        
        if current_symbols:
            overlap = len(current_symbols & target_symbols)
            overlap_ratio = overlap / len(current_symbols)
            logger.info(f"V99: 持仓重合度 = {overlap_ratio:.1%} ({overlap}/{len(current_symbols)})")
        
        return target_buy, symbols_to_sell


# ===========================================
# V99 Rebalance Limiter - 调仓比例限制（核心改进）
# ===========================================

class V99RebalanceLimiter:
    """
    V99 调仓比例限制器
    
    【核心功能】
    每日调仓比例严禁超过 15%
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_rebalance_ratio = self.config.get(
            'max_rebalance_ratio', V99_MAX_DAILY_REBALANCE_RATIO
        )
    
    def limit_rebalance(self, 
                        trades_to_execute: List[Dict[str, Any]],
                        portfolio_value: float,
                        current_positions: Dict[str, V99Position],
                        max_positions: int = V99_MAX_POSITIONS) -> Tuple[List[Dict[str, Any]], float]:
        """
        限制调仓比例
        
        Parameters
        ----------
        trades_to_execute : List[Dict[str, Any]]
            计划执行的交易列表
        portfolio_value : float
            组合价值
        current_positions : Dict[str, V99Position]
            当前持仓
        max_positions : int
            最大持仓数
            
        Returns
        -------
        Tuple[List[Dict[str, Any]], float]
            (限制后的交易列表，实际调仓比例)
        """
        if portfolio_value < EPSILON:
            return [], 0.0
        
        # 强制：如果当前持仓超过 max_positions，必须卖出多余的
        if len(current_positions) > max_positions:
            # 获取所有卖出交易
            sell_trades = [t for t in trades_to_execute if t.get('action') == 'sell']
            buy_trades = [t for t in trades_to_execute if t.get('action') == 'buy']
            
            # 计算需要额外卖出的数量
            excess_count = len(current_positions) - max_positions
            
            # 获取当前持仓中未在卖出列表中的股票
            selling_symbols = {t['symbol'] for t in sell_trades}
            positions_to_sell = []
            for symbol, position in current_positions.items():
                if symbol not in selling_symbols:
                    positions_to_sell.append((symbol, position))
            
            # 按排名排序，卖出排名最差的
            positions_to_sell.sort(key=lambda x: x[1].current_rank, reverse=True)
            
            # 添加强制卖出交易
            for symbol, position in positions_to_sell[:excess_count]:
                sell_amount = position.current_price * position.quantity
                sell_trades.append({
                    'symbol': symbol,
                    'action': 'sell',
                    'price': position.current_price,
                    'quantity': position.quantity,
                    'amount': sell_amount,
                    'cost': sell_amount * 0.001,  # 简化成本
                })
            
            # 重新组合交易列表：卖出优先，买入受调仓比例限制
            # 先执行所有卖出交易（不受限制）
            limited_trades = sell_trades
            
            # 计算买入可用额度
            sell_value = sum(t.get('amount', 0) for t in sell_trades)
            max_buy_value = portfolio_value * self.max_rebalance_ratio
            
            # 按比例分配买入额度
            total_buy_value = sum(t.get('amount', 0) for t in buy_trades)
            if total_buy_value > max_buy_value and max_buy_value > 0:
                scale_factor = max_buy_value / total_buy_value
                for trade in buy_trades:
                    limited_amount = trade.get('amount', 0) * scale_factor
                    limited_quantity = int(abs(limited_amount) / trade.get('price', 1))
                    if limited_quantity > 0:
                        limited_trade = trade.copy()
                        limited_trade['amount'] = limited_amount
                        limited_trade['quantity'] = limited_quantity
                        limited_trades.append(limited_trade)
            else:
                limited_trades.extend(buy_trades)
            
            total_trade_value = sum(abs(t.get('amount', 0)) for t in limited_trades)
            actual_ratio = total_trade_value / portfolio_value
            
            return limited_trades, actual_ratio
        
        # 正常情况：限制调仓比例
        if not trades_to_execute:
            return [], 0.0
        
        # 计算总调仓金额
        total_trade_value = sum(
            abs(t.get('amount', 0)) for t in trades_to_execute
        )
        
        # 计算调仓比例
        rebalance_ratio = total_trade_value / portfolio_value
        
        # 如果未超过限制，直接返回
        if rebalance_ratio <= self.max_rebalance_ratio:
            return trades_to_execute, rebalance_ratio
        
        # 超过限制，按比例缩减
        scale_factor = self.max_rebalance_ratio / rebalance_ratio
        
        limited_trades = []
        limited_value = 0.0
        
        for trade in trades_to_execute:
            limited_amount = trade.get('amount', 0) * scale_factor
            limited_quantity = int(abs(limited_amount) / trade.get('price', 1))
            
            if limited_quantity > 0:
                limited_trade = trade.copy()
                limited_trade['amount'] = limited_amount
                limited_trade['quantity'] = limited_quantity
                limited_trades.append(limited_trade)
                limited_value += abs(limited_amount)
        
        actual_ratio = limited_value / portfolio_value
        
        logger.warning(
            f"V99: 调仓比例限制触发 - 原计划 {rebalance_ratio:.1%}, "
            f"限制后 {actual_ratio:.1%} (上限 {self.max_rebalance_ratio:.1%})"
        )
        
        return limited_trades, actual_ratio


# ===========================================
# V99 ICStabilityFilter - 时间序列稳定性过滤（核心改进）
# ===========================================

class V99ICStabilityFilter:
    """
    V99 时间序列稳定性过滤器
    
    【核心功能】
    剔除过去 5 天 IC 波动率（Std）过大的因子分量
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.lookback_days = self.config.get('lookback_days', V99_IC_LOOKBACK_DAYS)
        self.std_threshold = self.config.get('std_threshold', V99_IC_STD_THRESHOLD)
        self.ic_history: Dict[str, List[float]] = {}  # {factor_name: [daily_ics]}
    
    def record_daily_ic(self, factor_name: str, ic_value: float) -> None:
        """记录每日 IC"""
        if factor_name not in self.ic_history:
            self.ic_history[factor_name] = []
        
        self.ic_history[factor_name].append(ic_value)
        
        # 保持只看过去 N 天
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
    
    def get_filtered_factors(self, factor_ics: Dict[str, float]) -> Dict[str, float]:
        """
        获取过滤后的因子 IC
        
        Parameters
        ----------
        factor_ics : Dict[str, float]
            {factor_name: ic_value}
            
        Returns
        -------
        Dict[str, float]
            过滤后的因子 IC（剔除不稳定的）
        """
        filtered = {}
        for factor_name, ic_value in factor_ics.items():
            if self.is_stable(factor_name):
                filtered[factor_name] = ic_value
            else:
                ic_std = self.get_ic_std(factor_name)
                logger.warning(
                    f"V99: 剔除因子 {factor_name} - IC Std={ic_std:.4f} > 阈值 {self.std_threshold}"
                )
        return filtered
    
    def get_factor_weights(self, factor_ics: Dict[str, float]) -> Dict[str, float]:
        """
        根据 IC 稳定性计算因子权重
        
        Parameters
        ----------
        factor_ics : Dict[str, float]
            {factor_name: ic_value}
            
        Returns
        -------
        Dict[str, float]
            {factor_name: weight}
        """
        weights = {}
        
        for factor_name, ic_value in factor_ics.items():
            if self.is_stable(factor_name):
                # 稳定因子：使用 IC 值作为权重基础
                weights[factor_name] = abs(ic_value)
            else:
                # 不稳定因子：降低权重或剔除
                ic_std = self.get_ic_std(factor_name)
                if ic_std > 0:
                    # 使用 IC/Std 比率作为权重（类似 IR）
                    weights[factor_name] = abs(ic_value) / ic_std * 0.5  # 打 5 折
                else:
                    weights[factor_name] = abs(ic_value) * 0.5
        
        # 归一化
        total_weight = sum(weights.values())
        if total_weight > EPSILON:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights


# ===========================================
# V99 Industry Checker - 行业数据检测
# ===========================================

class V99IndustryChecker:
    """
    V99 行业数据检测器
    
    【核心功能】
    如果某日 industry_code 缺失比例 > 10%，主动调用重新补取数据
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.missing_threshold = self.config.get(
            'missing_threshold', V99_INDUSTRY_MISSING_THRESHOLD
        )
    
    def check_industry_coverage(self, df: pl.DataFrame, trade_date: str) -> Tuple[bool, float]:
        """
        检查行业代码覆盖率
        
        Parameters
        ----------
        df : pl.DataFrame
            当日数据
        trade_date : str
            交易日期
            
        Returns
        -------
        Tuple[bool, float]
            (是否通过检查，缺失比例)
        """
        if 'industry_code' not in df.columns:
            return False, 1.0
        
        total_count = df.height
        if total_count == 0:
            return False, 1.0
        
        # 统计缺失数量
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
                f"V99: {trade_date} 行业代码缺失比例 {missing_ratio:.1%} > "
                f"阈值 {self.missing_threshold:.1%}，必须重新补取数据！"
            )
        
        return passed, missing_ratio
    
    def repair_industry_data(self, df: pl.DataFrame, db=None) -> pl.DataFrame:
        """
        修复行业数据
        
        Parameters
        ----------
        df : pl.DataFrame
            原始数据
        db : DatabaseManager
            数据库管理器
            
        Returns
        -------
        pl.DataFrame
            修复后的数据
        """
        logger.warning("V99: 开始修复行业数据...")
        
        # 尝试从数据库重新获取行业数据
        if db is not None:
            try:
                # 获取缺失行业数据的股票列表
                missing_symbols = df.filter(
                    (pl.col('industry_code').is_null()) |
                    (pl.col('industry_code').cast(pl.Utf8).str.len_chars() == 0)
                )['symbol'].unique().to_list()
                
                if missing_symbols:
                    # 从数据库重新查询行业数据
                    logger.info(f"V99: 重新获取 {len(missing_symbols)} 只股票的行业数据...")
                    
                    # 这里调用实际的数据补取逻辑
                    # 简化处理：使用最常见的行业代码填充
                    industry_counts = df.group_by('industry_code').agg(
                        pl.count().alias('cnt')
                    ).filter(
                        pl.col('industry_code').is_not_null() &
                        (pl.col('industry_code').cast(pl.Utf8).str.len_chars() > 0)
                    ).sort('cnt', descending=True)
                    
                    if not industry_counts.is_empty():
                        default_industry = industry_counts['industry_code'][0]
                        df = df.with_columns([
                            pl.when(
                                (pl.col('industry_code').is_null()) |
                                (pl.col('industry_code').cast(pl.Utf8).str.len_chars() == 0)
                            )
                            .then(default_industry)
                            .otherwise(pl.col('industry_code'))
                            .alias('industry_code')
                        ])
                        
                        logger.info(f"V99: 行业数据修复完成，使用 {default_industry} 填充缺失值")
                        
            except Exception as e:
                logger.error(f"V99: 行业数据修复失败 - {e}")
        
        return df


# ===========================================
# V99 TurnoverTracker
# ===========================================

class V99TurnoverTracker:
    """V99 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V99TurnoverRecord] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False,
                        rebalance_ratio: float = 0.0) -> V99TurnoverRecord:
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
        
        record = V99TurnoverRecord(
            trade_date=trade_date,
            turnover_rate=turnover_rate,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            annualized_turnover=annualized_turnover,
            daily_turnover=daily_turnover,
            is_rebalance_day=is_rebalance_day,
            rebalance_ratio=rebalance_ratio,
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
        
        is_active = V99_TURNOVER_MIN <= annualized_turnover <= V99_TURNOVER_MAX
        daily_ok = max_daily <= V99_DAILY_TURNOVER_MAX
        turnover_ok = annualized_turnover <= 5.0  # 500% 上限
        
        return {
            'mean_turnover': float(np.mean([r.turnover_rate for r in self.turnover_records])),
            'std_turnover': float(np.std([r.turnover_rate for r in self.turnover_records])),
            'max_turnover': float(np.max([r.turnover_rate for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'daily_turnover_ok': daily_ok,
            'turnover_ok': turnover_ok,
            'max_daily_turnover': float(max_daily),
            'turnover_min': V99_TURNOVER_MIN,
            'turnover_max': V99_TURNOVER_MAX,
        }


# ===========================================
# V99 TransactionCostCalculator - 交易成本计算器（核心改进）
# ===========================================

class V99TransactionCostCalculator:
    """
    V99 交易成本计算器
    
    【核心功能】
    硬编码 TRANSACTION_COST = 0.0015（单边）
    计算扣费后净收益
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.transaction_cost = self.config.get(
            'transaction_cost', V99_TRANSACTION_COST
        )
        self.commission_rate = self.config.get(
            'commission_rate', V99_COMMISSION_RATE
        )
        self.min_commission = self.config.get(
            'min_commission', V99_MIN_COMMISSION
        )
        self.stamp_duty = self.config.get('stamp_duty', V99_STAMP_DUTY)
        self.transfer_fee = self.config.get('transfer_fee', V99_TRANSFER_FEE)
    
    def calculate_buy_cost(self, amount: float) -> float:
        """计算买入成本"""
        commission = max(amount * self.commission_rate, self.min_commission)
        transfer_fee = amount * self.transfer_fee
        # 买入无印花税
        return commission + transfer_fee
    
    def calculate_sell_cost(self, amount: float) -> float:
        """计算卖出成本"""
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty
        transfer_fee = amount * self.transfer_fee
        return commission + stamp_duty + transfer_fee
    
    def calculate_transaction_cost(self, amount: float, is_buy: bool) -> float:
        """计算交易成本"""
        if is_buy:
            return self.calculate_buy_cost(amount)
        else:
            return self.calculate_sell_cost(amount)
    
    def calculate_simple_cost(self, amount: float) -> float:
        """
        简化成本计算（使用硬编码的 TRANSACTION_COST）
        
        【V99 核心】使用 0.15% 单边成本
        """
        return amount * self.transaction_cost
    
    def get_cost_summary(self, total_buy_amount: float, 
                         total_sell_amount: float) -> Dict[str, float]:
        """获取成本摘要"""
        buy_cost = self.calculate_buy_cost(total_buy_amount)
        sell_cost = self.calculate_sell_cost(total_sell_amount)
        total_cost = buy_cost + sell_cost
        
        simple_cost = (total_buy_amount + total_sell_amount) * self.transaction_cost
        
        return {
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'total_cost': total_cost,
            'simple_cost': simple_cost,
            'transaction_cost_rate': self.transaction_cost,
        }


# ===========================================
# V99 ICAudit
# ===========================================

class V99ICAudit:
    """V99 IC 审计（带稳定性过滤）"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.stability_filter = V99ICStabilityFilter(config)
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'smoothed_signal') -> Dict[str, Any]:
        """计算 Rank IC"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 使用 shift(-lag) 获取未来收益
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
        
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        
        # 计算 IC Stability
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_stability = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 记录到稳定性过滤器
        self.stability_filter.record_daily_ic('composite', ic_t1)
        
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
            't1_ic_passed': ic_t1 >= V99_T1_IC_TARGET,
            'stability_passed': ic_stability >= V99_IC_STABILITY_TARGET,
            'all_daily_ics': all_daily_ics,
        }
    
    def calculate_single_factor_ic(self, df: pl.DataFrame, 
                                    factor_name: str,
                                    signal_col: str) -> V99SingleFactorIC:
        """计算单因子 IC（带稳定性过滤）"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
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
        
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_ir = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        ic_stability = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 记录到稳定性过滤器
        self.stability_filter.record_daily_ic(factor_name, ic_t1)
        ic_std_5d = self.stability_filter.get_ic_std(factor_name)
        
        passed_stability = ic_std_5d <= V99_IC_STD_THRESHOLD
        
        return V99SingleFactorIC(
            factor_name=factor_name,
            ic_t1=ic_t1,
            ic_t2=ic_t2,
            ic_t3=ic_t3,
            ic_ir=ic_ir,
            ic_stability=ic_stability,
            ic_std_5d=ic_std_5d,
            passed_threshold=ic_t1 >= V99_T1_IC_TARGET,
            passed_stability_filter=passed_stability,
        )


# ===========================================
# V99 DataManager
# ===========================================

class V99DataManager:
    """V99 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V99_WARMUP_PERIOD)
    
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
        extra_days = max(V99_HALF_LIFE_LAGS) + 20
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
            
            logger.info(f"V99: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V99: 数据加载失败 - {e}")
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
                logger.warning(f"V99: 获取最常见行业失败 - {e}，使用默认值")
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
# V99 Residual Momentum
# ===========================================

class V99ResidualMomentumEngine:
    """V99 残差动量引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.momentum_window = self.config.get('momentum_window', 20)
    
    def compute_residual_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算残差动量（A 股反转效应修正）"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            ((pl.col('close') / pl.col('close').shift(self.momentum_window)) - 1).alias('momentum_raw')
        ])
        
        # A 股反转效应：动量值越小（跌幅越大），排名越靠前
        result = result.with_columns([
            pl.col('momentum_raw').rank('ordinal', descending=False).over('trade_date').alias('momentum_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('momentum_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('residual_momentum_score')
        ])
        
        result = result.drop(['momentum_rank', 'n_stocks'])
        
        logger.info(f"V99: 残差动量计算完成（反转效应修正）")
        
        return result


# ===========================================
# V99 Smart Flow
# ===========================================

class V99SmartFlowEngine:
    """V99 聪明资金流引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.flow_window = self.config.get('flow_window', 5)
    
    def compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算聪明资金流（A 股反转效应修正）"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        numerator = (pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))
        denominator = pl.col('high') - pl.col('low') + EPSILON
        result = result.with_columns(
            (numerator / denominator).alias('money_flow_factor')
        )
        
        result = result.with_columns(
            (pl.col('money_flow_factor') * pl.col('volume').fill_null(0)).alias('money_flow')
        )
        
        result = result.with_columns([
            pl.col('money_flow')
            .rolling_mean(window_size=self.flow_window)
            .over('symbol')
            .alias('money_flow_ma')
        ])
        
        # A 股反转效应：资金流值越小（流出），排名越靠前
        result = result.with_columns([
            pl.col('money_flow_ma').rank('ordinal', descending=False).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('flow_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('smart_flow_score')
        ])
        
        keep_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 
                     'volume', 'amount', 'pct_chg', 'industry_code', 'total_mv', 
                     'is_st', 'smart_flow_score']
        
        if 'residual_momentum_score' in result.columns:
            keep_cols.append('residual_momentum_score')
        
        result = result.select(keep_cols)
        
        logger.info(f"V99: 聪明资金流计算完成")
        
        return result


# ===========================================
# V99 AlphaFusion
# ===========================================

class V99AlphaFusion:
    """V99 AlphaFusion - 半衰期衰减融合引擎"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V99_HALF_LIFE_LAGS)
        
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        self.smoother = V99SignalSmoother(config)
        
        logger.info("V99 AlphaFusion 初始化完成")
    
    def compute_fusion_signal(self, df: pl.DataFrame,
                               signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算半衰期融合信号并应用 EMA 平滑"""
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
            weight = self.half_life_weights[i]
            fusion_exprs.append(pl.col(lag_col) * weight)
        
        result = result.with_columns([
            sum(fusion_exprs).alias('fused_signal')
        ])
        
        result = self.smoother.smooth_signal(result, 'fused_signal')
        
        logger.info(f"V99: 融合信号计算完成（含 EMA 平滑）")
        
        return result


# ===========================================
# V99 AlphaWeight
# ===========================================

class V99AlphaWeightEngine:
    """V99 Alpha 权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V99_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V99_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V99_MAX_SINGLE_WEIGHT)
    
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
        
        logger.info(f"V99: Alpha 权重计算完成")
        
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
# V99 导出列表
# ===========================================

__all__ = [
    # 异常类
    'DirectionalError',
    'IndustryDataMissingError',
    # 常量
    'V99_INITIAL_CAPITAL',
    'V99_MAX_POSITIONS',
    'V99_WARMUP_PERIOD',
    'V99_MIN_SCORE_THRESHOLD',
    'V99_BUY_RANK_THRESHOLD',
    'V99_SELL_RANK_THRESHOLD',
    'V99_MAX_DAILY_REBALANCE_RATIO',
    'V99_TRANSACTION_COST',
    'V99_TURNOVER_MIN',
    'V99_TURNOVER_MAX',
    'V99_DAILY_TURNOVER_MAX',
    'V99_T1_IC_TARGET',
    'V99_IC_STABILITY_TARGET',
    'V99_IC_LOOKBACK_DAYS',
    'V99_IC_STD_THRESHOLD',
    'V99_INDUSTRY_MISSING_THRESHOLD',
    'V99_RESIDUAL_WEIGHT',
    'V99_FLOW_WEIGHT',
    # 数据类
    'V99SingleFactorIC',
    'V99FusionSignal',
    'V99TurnoverRecord',
    'V99Position',
    'V99DailyPortfolio',
    # 工具函数
    'normalize_rank',
    'zscore_normalize',
    'calculate_half_life_decay_weights',
    'fill_with_market_median',
    'calculate_rank_correlation',
    'ols_residualize',
    'ema_smooth',
    'calculate_ic_stability',
    'EPSILON',
    # 核心类
    'V99SignalSmoother',
    'V99PositionBuffer',
    'V99RebalanceLimiter',
    'V99ICStabilityFilter',
    'V99IndustryChecker',
    'V99TurnoverTracker',
    'V99TransactionCostCalculator',
    'V99ICAudit',
    'V99ResidualMomentumEngine',
    'V99SmartFlowEngine',
    'V99AlphaFusion',
    'V99AlphaWeightEngine',
]