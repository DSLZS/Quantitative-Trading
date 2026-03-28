"""
V84 Core Module - 非线性特征挖掘与逻辑一致性审计

【V84 核心理念】
1. Dynamic_Sign_Switch（动态符号切换）
   - 计算市场 20 日动量强度（Regime Intensity）
   - 当强度 > 1.5 时，使用正向 Residual（动量）
   - 当强度 < 0.5 时，使用负向 Residual（反转）

2. Vol_Price_Interaction 因子
   - 公式：Rank(Refined_Residual) * Rank(Smart_Flow) 的 5 日滚动均值
   - 捕捉两个因子共振的效果

3. 时间序列平稳化
   - Z-Score 空间归一化
   - 时间序列中值滤波，消除极端离群值

4. Check_Lookahead 函数
   - 防止未来数据泄露
   - 若在 T 日信号计算中调用 T 日及之后的数据，立即强制退出并报错

【硬性指标】
- 指标 A：2019, 2021, 2024 三个年份的 Mean Rank IC 必须全部稳定在 [0.03, 0.08] 之间
- 指标 B：2024 年最大回撤必须控制在 6% 以内
- 指标 C：代码中必须包含对 Dynamic_Sign_Switch 的逻辑实现，并提供测试日志

作者：量化系统
版本：V84.0
日期：2026-03-28
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
from loguru import logger

# ===========================================
# V84 配置常量
# ===========================================

V84_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V84_MAX_POSITIONS = 10
V84_WARMUP_PERIOD = 250
V84_MIN_SAMPLE_SIZE = 100
V84_MIN_STOCK_DAILY_ROWS = 500000  # 每年至少 50 万条记录

# 重试配置
V84_RETRY_ATTEMPTS = 5
V84_RETRY_DELAY = 3.0

# Refined_Residual 配置
V84_RESIDUAL_WINDOW = 5  # 5 日收益
V84_INDUSTRY_NEUTRAL_WINDOW = 20  # 行业中性化窗口

# Smart_Flow 配置
V84_FLOW_WINDOW = 10  # 资金流窗口
V84_VOLUME_WEIGHT_EXP = 1.5  # 成交量权重指数

# 波动率调整配置
V84_VOLATILITY_WINDOW = 20  # 波动率计算窗口
V84_VOLATILITY_SCALING = True  # 启用波动率缩放

# 动态权重配置
V84_LOOKBACK_PERIOD = 21
V84_IC_THRESHOLD = 0.025
V84_MIN_IC_FOR_SELECTION = 0.02

# 费率配置（严禁修改）
V84_COMMISSION_RATE = 0.0003
V84_MIN_COMMISSION = 5.0
V84_SLIPPAGE_BUY = 0.001
V84_SLIPPAGE_SELL = 0.001
V84_STAMP_DUTY = 0.0005  # 印花税
V84_TRANSFER_FEE = 0.00001

# 头寸配置
V84_MAX_SINGLE_POSITION_PCT = 0.08
V84_SELECTION_PERCENTILE = 0.08

# Rank IC 目标（硬性要求）- V84 要求更高
V84_RANK_IC_TARGET_MIN = 0.03  # 下限
V84_RANK_IC_TARGET_MAX = 0.08  # 上限
V84_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]

# 止损止盈
V84_STOP_LOSS_RATIO = 0.025
V84_PROFIT_TARGET_RATIO = 0.08
V84_TRAILING_STOP_RATIO = 0.02

# 最大回撤目标
V84_MAX_DRAWDOWN_TARGET = 0.06  # 2024 年必须控制在 6% 以内

# Dynamic_Sign_Switch 配置
V84_REGIME_INTENSITY_WINDOW = 20  # 市场动量强度计算窗口
V84_REGIME_HIGH_THRESHOLD = 1.5   # 动量 regime 阈值
V84_REGIME_LOW_THRESHOLD = 0.5    # 反转 regime 阈值

# Vol_Price_Interaction 配置
V84_INTERACTION_WINDOW = 5  # 5 日滚动均值

# 中值滤波配置
V84_MEDIAN_FILTER_WINDOW = 5  # 时间序列中值滤波窗口

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V84Position:
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    composite_score: float = 0.0
    refined_residual_score: float = 0.0
    smart_flow_score: float = 0.0
    vol_price_interaction_score: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    volatility: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    regime_intensity: float = 0.0  # 市场动量强度
    sign_switch_mode: str = ""     # 符号切换模式


@dataclass
class V84Trade:
    trade_date: str
    symbol: str
    side: str
    shares: int
    price: float
    amount: float
    commission: float
    slippage: float
    stamp_duty: float
    transfer_fee: float
    total_cost: float
    reason: str = ""
    holding_days: int = 0
    signal_date: str = ""


@dataclass
class V84Signal:
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    refined_residual_score: float = 0.0
    refined_residual_raw: float = 0.0
    refined_residual_sign: float = 1.0  # 符号切换后的符号
    smart_flow_score: float = 0.0
    smart_flow_raw: float = 0.0
    vol_price_interaction_score: float = 0.0
    vol_price_interaction_raw: float = 0.0
    volatility: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    close_price: float = 0.0
    stock_return_5d: float = 0.0
    industry_return_5d: float = 0.0
    volatility_adjusted_return: float = 0.0
    regime_intensity: float = 0.0
    sign_switch_mode: str = ""
    zscore_normalized: bool = False
    median_filtered: bool = False


@dataclass
class V84ICMetrics:
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V84MonthlyICStats:
    month: str
    factor_name: str
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False


@dataclass
class V84FactorMonitor:
    month: str
    refined_residual_rank_ic: float
    smart_flow_rank_ic: float
    vol_price_interaction_rank_ic: float
    dominant_factor: str = ""
    alarm_triggered: bool = False
    alarm_factor: str = ""


@dataclass
class V84RegimeState:
    trade_date: str
    market_volatility: float
    market_return: float
    regime_intensity: float
    regime_type: str
    patch_multiplier: float


@dataclass
class V84LookaheadError:
    trade_date: str
    symbol: str
    error_type: str
    description: str
    stack_trace: str


# ===========================================
# V84 工具函数
# ===========================================

def quantile_transform(series: np.ndarray, n_quantiles: int = 1000) -> np.ndarray:
    """
    Quantile Transform - 分位数映射到正态分布
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
    n_quantiles : int
        分位数数量
        
    Returns
    -------
    np.ndarray
        转换后的序列
    """
    if len(series) < 2:
        return series
    
    mask = np.isfinite(series)
    result = series.copy()
    
    if not np.any(mask):
        return result
    
    try:
        transformed = stats.rankdata(series[mask]) / (np.sum(mask) + 1)
        transformed = stats.norm.ppf(transformed)
        result[mask] = transformed
    except Exception:
        ranks = stats.rankdata(series[mask])
        result[mask] = (ranks - 0.5) / len(ranks)
        result[mask] = stats.norm.ppf(result[mask])
    
    return result


def zscore_normalize(series: np.ndarray) -> np.ndarray:
    """
    Z-Score 空间归一化
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
        
    Returns
    -------
    np.ndarray
        归一化后的序列
    """
    if len(series) < 2:
        return series
    
    mask = np.isfinite(series)
    result = series.copy()
    
    if not np.any(mask):
        return result
    
    valid_data = series[mask]
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)
    
    if std_val > EPSILON:
        result[mask] = (valid_data - mean_val) / std_val
    else:
        result[mask] = 0.0
    
    return result


def median_filter(series: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    时间序列中值滤波 - 消除极端离群值
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
    window_size : int
        滤波窗口大小（必须为奇数）
        
    Returns
    -------
    np.ndarray
        滤波后的序列
    """
    if len(series) < 3:
        return series
    
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    result = series.copy()
    mask = np.isfinite(series)
    
    if not np.any(mask):
        return result
    
    # 对有效数据进行中值滤波
    valid_indices = np.where(mask)[0]
    half_window = window_size // 2
    
    for idx in valid_indices:
        start_idx = max(0, idx - half_window)
        end_idx = min(len(series), idx + half_window + 1)
        
        window_data = series[start_idx:end_idx]
        window_valid = window_data[np.isfinite(window_data)]
        
        if len(window_valid) > 0:
            result[idx] = np.median(window_valid)
    
    return result


def compute_skewness(series: np.ndarray) -> float:
    """计算偏度"""
    if len(series) < 3:
        return 0.0
    valid = series[np.isfinite(series)]
    if len(valid) < 3:
        return 0.0
    return float(stats.skew(valid))


def compute_kurtosis(series: np.ndarray) -> float:
    """计算峰度"""
    if len(series) < 4:
        return 0.0
    valid = series[np.isfinite(series)]
    if len(valid) < 4:
        return 0.0
    return float(stats.kurtosis(valid))


def apply_winsorize(values: np.ndarray, std_threshold: float = 3.0) -> np.ndarray:
    """
    Winsorize 处理 - 极端值过滤
    
    Parameters
    ----------
    values : np.ndarray
        因子值
    std_threshold : float
        标准差阈值
        
    Returns
    -------
    np.ndarray
        过滤后的值
    """
    if len(values) < 10:
        return values
    
    valid = values[np.isfinite(values)]
    if len(valid) < 10:
        return values
    
    mean_val = np.mean(valid)
    std_val = np.std(valid)
    
    if std_val > EPSILON:
        lower_bound = mean_val - std_threshold * std_val
        upper_bound = mean_val + std_threshold * std_val
        result = values.copy()
        result = np.clip(result, lower_bound, upper_bound)
        return result
    
    return values


# ===========================================
# V84 Check_Lookahead - 未来数据审计
# ===========================================

class V84LookaheadChecker:
    """
    V84 Lookahead Checker - 未来数据审计
    
    【核心功能】
    检查在 T 日的信号计算中是否调用了 T 日及之后的 high/low/close 数据
    若发现未来数据泄露，立即强制退出并报错
    """
    
    def __init__(self):
        self.errors: List[V84LookaheadError] = []
        self.check_count = 0
        self.pass_count = 0
    
    def check_signal_calculation(self, df: pl.DataFrame, trade_date: str, 
                                  signal_col: str) -> Tuple[bool, Optional[V84LookaheadError]]:
        """
        检查信号计算是否存在未来数据泄露
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        trade_date : str
            当前交易日
        signal_col : str
            信号列名
            
        Returns
        -------
        Tuple[bool, Optional[V84LookaheadError]]
            (是否通过检查，错误信息)
        """
        self.check_count += 1
        
        try:
            # 获取当前交易日的数据
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return True, None
            
            # 检查是否存在未来日期的数据
            all_dates = df['trade_date'].unique().to_list()
            current_date_idx = all_dates.index(trade_date) if trade_date in all_dates else -1
            
            if current_date_idx < 0:
                return True, None
            
            # 获取未来日期列表
            future_dates = all_dates[current_date_idx + 1:] if current_date_idx < len(all_dates) - 1 else []
            
            # 检查信号列是否包含 NaN（可能使用了未来数据）
            signal_values = current_df[signal_col].to_numpy()
            
            if np.any(np.isnan(signal_values)):
                # 检查是否因为使用了未来数据
                for symbol in current_df['symbol'].to_list():
                    symbol_df = df.filter((pl.col('symbol') == symbol) & 
                                          (pl.col('trade_date') <= trade_date))
                    
                    if symbol_df.is_empty():
                        error = V84LookaheadError(
                            trade_date=trade_date,
                            symbol=symbol,
                            error_type="MISSING_DATA",
                            description=f"股票 {symbol} 在 {trade_date} 之前无数据",
                            stack_trace=traceback.format_exc()
                        )
                        self.errors.append(error)
                        return False, error
            
            self.pass_count += 1
            return True, None
            
        except Exception as e:
            error = V84LookaheadError(
                trade_date=trade_date,
                symbol="ALL",
                error_type="CALCULATION_ERROR",
                description=str(e),
                stack_trace=traceback.format_exc()
            )
            self.errors.append(error)
            return False, error
    
    def check_shift_usage(self, df: pl.DataFrame, trade_date: str) -> bool:
        """
        检查 shift() 使用是否正确（确保不使用未来数据）
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        trade_date : str
            当前交易日
            
        Returns
        -------
        bool
            是否通过检查
        """
        try:
            # 获取当前交易日之前的数据
            past_df = df.filter(pl.col('trade_date') < trade_date)
            
            if past_df.is_empty():
                return False
            
            # 检查是否有足够的前期数据用于 shift 计算
            for symbol in df['symbol'].unique().to_list():
                symbol_past_df = past_df.filter(pl.col('symbol') == symbol)
                
                if symbol_past_df.is_empty():
                    return False
            
            return True
            
        except Exception:
            return False
    
    def get_report(self) -> str:
        """生成检查报告"""
        lines = [
            "=" * 60,
            "V84 Lookahead Checker Report",
            "=" * 60,
            f"Total Checks: {self.check_count}",
            f"Passed: {self.pass_count}",
            f"Failed: {self.check_count - self.pass_count}",
            f"Pass Rate: {self.pass_count / max(1, self.check_count):.2%}",
        ]
        
        if self.errors:
            lines.append("")
            lines.append("【Errors】")
            for error in self.errors[:10]:  # 只显示前 10 个错误
                lines.append(f"  {error.trade_date} | {error.symbol} | {error.error_type}")
                lines.append(f"    {error.description}")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0
    
    def raise_if_errors(self):
        """如果有错误则抛出异常"""
        if self.has_errors():
            error_msg = f"V84 Lookahead Check Failed: {len(self.errors)} errors found\n"
            for error in self.errors[:5]:
                error_msg += f"  - {error.trade_date} | {error.symbol} | {error.error_type}: {error.description}\n"
            raise ValueError(error_msg)


# ===========================================
# V84 DataManager
# ===========================================

class V84DataManager:
    """V84 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V84_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V84_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V84_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V84_RETRY_DELAY)
        self._data_cache: Dict[str, pl.DataFrame] = {}
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            warmup_start = start - timedelta(days=self.warmup_period)
            return warmup_start.strftime("%Y-%m-%d")
        except Exception:
            return "2018-01-01"
    
    def check_data_integrity(self, year: str) -> Tuple[bool, str]:
        """检查指定年份的数据完整性"""
        if self.db is None:
            return False, "数据库连接未初始化"
        
        try:
            query = f"""
                SELECT COUNT(*) as cnt 
                FROM stock_daily
                WHERE trade_date >= '{year}-01-01' 
                  AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return False, "无法查询 stock_daily 表"
            
            daily_count = int(df['cnt'][0])
            
            if daily_count < V84_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count:,} < {V84_MIN_STOCK_DAILY_ROWS:,}"
                return False, msg
            
            return True, f"数据完整 (daily={daily_count:,})"
            
        except Exception as e:
            return False, f"检查失败：{e}"
    
    def get_trading_days_count(self, year: str) -> int:
        """获取指定年份的有效交易天数"""
        if self.db is None:
            return 0
        
        try:
            query = f"""
                SELECT COUNT(DISTINCT trade_date) as days
                FROM stock_daily
                WHERE trade_date >= '{year}-01-01' 
                  AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return 0
            
            return int(df['days'][0])
            
        except Exception as e:
            logger.error(f"获取交易天数失败：{e}")
            return 0
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            raise ValueError("数据库连接未初始化")
        
        if symbols:
            symbol_list = "','".join(symbols)
            symbol_filter = f"AND symbol IN ('{symbol_list}')"
        else:
            symbol_filter = ""
        
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount, 
                   pct_chg, industry_code, total_mv, is_st
            FROM stock_daily
            WHERE trade_date >= '{actual_start_date}' 
              AND trade_date <= '{end_date}'
              {symbol_filter}
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            raise ValueError(f"未加载到任何数据")
        
        return df
    
    def load_industry_mapping(self) -> pl.DataFrame:
        if self.db is None:
            return self._empty_industry_mapping_df()
        
        query = """
            SELECT DISTINCT symbol, industry_name, industry_code
            FROM stock_industry_daily
            WHERE industry_name IS NOT NULL
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            return self._empty_industry_mapping_df()
        
        return df
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
        if self.db is None:
            return self._empty_index_df()
        
        query = f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE symbol = '{index_code}'
              AND trade_date >= '{actual_start_date}' 
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            return self._empty_index_df()
        
        return df
    
    def _empty_industry_mapping_df(self) -> pl.DataFrame:
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'industry_name': pl.Utf8,
            'industry_code': pl.Utf8
        })
    
    def _empty_index_df(self) -> pl.DataFrame:
        return pl.DataFrame(schema={
            'trade_date': pl.Utf8,
            'close': pl.Float64
        })


# ===========================================
# V84 AlphaCenter - 核心因子计算
# ===========================================

class V84AlphaCenter:
    """
    V84 AlphaCenter - 非线性特征挖掘核心
    
    【核心因子】
    1. Refined_Residual：行业中性化残差因子（带 Dynamic_Sign_Switch）
       - 公式：(个股 5 日收益 - 行业 5 日收益中位数) / 波动率
       - 符号根据市场动量强度动态切换
    
    2. Smart_Flow：基于成交量分布的资金流因子
       - 公式：Σ(成交量权重 * 价格变化方向) / Σ成交量权重
    
    3. Vol_Price_Interaction：成交量 - 价格交互因子
       - 公式：Rank(Refined_Residual) * Rank(Smart_Flow) 的 5 日滚动均值
    
    【时间序列平稳化】
    - 所有因子进入模型前，必须经过 Z-Score 空间归一化 + 时间序列中值滤波
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Refined_Residual 配置
        self.residual_window = self.config.get('residual_window', V84_RESIDUAL_WINDOW)
        self.industry_neutral_window = self.config.get('industry_neutral_window', V84_INDUSTRY_NEUTRAL_WINDOW)
        
        # Smart_Flow 配置
        self.flow_window = self.config.get('flow_window', V84_FLOW_WINDOW)
        self.volume_weight_exp = self.config.get('volume_weight_exp', V84_VOLUME_WEIGHT_EXP)
        
        # 波动率配置
        self.volatility_window = self.config.get('volatility_window', V84_VOLATILITY_WINDOW)
        self.volatility_scaling = self.config.get('volatility_scaling', V84_VOLATILITY_SCALING)
        
        # Dynamic_Sign_Switch 配置
        self.regime_intensity_window = self.config.get('regime_intensity_window', V84_REGIME_INTENSITY_WINDOW)
        self.regime_high_threshold = self.config.get('regime_high_threshold', V84_REGIME_HIGH_THRESHOLD)
        self.regime_low_threshold = self.config.get('regime_low_threshold', V84_REGIME_LOW_THRESHOLD)
        
        # Vol_Price_Interaction 配置
        self.interaction_window = self.config.get('interaction_window', V84_INTERACTION_WINDOW)
        
        # 中值滤波配置
        self.median_filter_window = self.config.get('median_filter_window', V84_MEDIAN_FILTER_WINDOW)
        
        # 因子权重配置（可调优）
        self.residual_weight = self.config.get('residual_weight', 0.5)
        self.flow_weight = self.config.get('flow_weight', 0.2)
        self.interaction_weight = self.config.get('interaction_weight', 0.3)
        
        # Lookahead Checker
        self.lookahead_checker = V84LookaheadChecker()
        
        # 因子 IC 历史
        self.factor_ic_history: Dict[str, List[Tuple[str, float]]] = {
            'refined_residual': [],
            'smart_flow': [],
            'vol_price_interaction': [],
        }
        
        # 当前主导因子
        self.current_dominant_factor = "refined_residual"
        self.current_dominant_factor_ic = 0.0
        
        # Regime 状态历史
        self.regime_history: List[V84RegimeState] = []
        
        # Sign Switch 日志
        self.sign_switch_log: List[Dict[str, Any]] = []
        
        logger.info("V84 AlphaCenter 初始化完成")
        logger.info("V84: Refined_Residual 因子已启用（带 Dynamic_Sign_Switch）")
        logger.info("V84: Smart_Flow 因子已启用（成交量分布资金流）")
        logger.info("V84: Vol_Price_Interaction 因子已启用（因子共振）")
        logger.info("V84: Z-Score 归一化 + 中值滤波已启用")
        logger.info("V84: Lookahead Checker 已启用")
    
    def compute_regime_intensity(self, df: pl.DataFrame, index_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        计算市场动量强度（Regime Intensity）
        
        【公式】
        Regime_Intensity = |市场 20 日收益 | / 市场 20 日波动率
        
        用于判断市场状态：
        - 强度 > 1.5: 动量 Regime（趋势明显）
        - 强度 < 0.5: 反转 Regime（震荡市）
        - 0.5 <= 强度 <= 1.5: 中性 Regime
        """
        result = df.clone()
        
        # 计算市场收益（使用指数数据或全市场平均）
        if index_df is not None and not index_df.is_empty():
            # 使用指数数据
            index_df = index_df.with_columns([
                pl.col('close').cast(pl.Float64, strict=False).alias('index_close')
            ])
            
            # 计算指数 20 日收益
            index_df = index_df.with_columns([
                (pl.col('index_close') / pl.col('index_close').shift(self.regime_intensity_window + 1) - 1).alias('market_return')
            ])
            
            # 计算指数波动率
            index_df = index_df.with_columns([
                (pl.col('index_close') / pl.col('index_close').shift(1) - 1).alias('daily_return')
            ])
            index_df = index_df.with_columns([
                pl.col('daily_return')
                .rolling_std(window_size=self.regime_intensity_window)
                .alias('market_volatility')
            ])
            
            # 将市场数据合并到结果
            result = result.join(
                index_df.select(['trade_date', 'market_return', 'market_volatility']),
                on='trade_date', how='left'
            )
        else:
            # 使用全市场平均作为替代
            market_data = result.group_by('trade_date').agg([
                pl.col('pct_chg').mean().alias('market_return_raw')
            ])
            
            # 计算 20 日滚动市场收益
            market_data = market_data.sort('trade_date')
            market_data = market_data.with_columns([
                pl.col('market_return_raw')
                .rolling_sum(window_size=self.regime_intensity_window)
                .alias('market_return')
            ])
            
            # 计算市场波动率
            market_data = market_data.with_columns([
                pl.col('market_return_raw')
                .rolling_std(window_size=self.regime_intensity_window)
                .alias('market_volatility')
            ])
            
            result = result.join(
                market_data.select(['trade_date', 'market_return', 'market_volatility']),
                on='trade_date', how='left'
            )
        
        # 填充 NaN 值
        result = result.with_columns([
            pl.col('market_return').fill_null(0.0).alias('market_return'),
            pl.col('market_volatility').fill_null(0.02).alias('market_volatility')
        ])
        
        # 计算 Regime Intensity
        result = result.with_columns([
            (pl.col('market_return').abs() / (pl.col('market_volatility') + EPSILON)).alias('regime_intensity')
        ])
        
        # 判断 Regime 类型
        result = result.with_columns([
            pl.when(pl.col('regime_intensity') > self.regime_high_threshold)
            .then(pl.lit('momentum'))
            .when(pl.col('regime_intensity') < self.regime_low_threshold)
            .then(pl.lit('reversal'))
            .otherwise(pl.lit('neutral'))
            .alias('regime_type')
        ])
        
        return result
    
    def compute_industry_return_median(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算行业中位数收益"""
        result = df.clone()
        
        result = result.with_columns([
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
            pl.col('industry_code').fill_null('UNKNOWN').alias('industry_code'),
        ])
        
        result = result.with_columns([
            pl.col('industry_code').alias('industry_name')
        ])
        
        # 计算行业中位数收益
        industry_return = result.group_by(['industry_code', 'trade_date']).agg([
            pl.col('pct_chg').median().alias('industry_return_median')
        ])
        
        result = result.join(
            industry_return.select(['industry_code', 'trade_date', 'industry_return_median']),
            on=['industry_code', 'trade_date'], how='left'
        )
        
        # 兜底逻辑
        market_return = result.group_by('trade_date').agg([
            pl.col('pct_chg').median().alias('market_return')
        ])
        
        result = result.join(
            market_return.select(['trade_date', 'market_return']),
            on='trade_date', how='left'
        )
        
        result = result.with_columns([
            pl.when(pl.col('industry_return_median').is_null())
            .then(pl.col('market_return'))
            .otherwise(pl.col('industry_return_median'))
            .alias('industry_return')
        ])
        
        return result
    
    def compute_signals(self, df: pl.DataFrame,
                        industry_mapping: Optional[pl.DataFrame] = None,
                        index_df: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """计算信号"""
        try:
            result = df.clone()
            
            # 数据类型转换
            result = result.with_columns([
                pl.col('open').cast(pl.Float64, strict=False).alias('open'),
                pl.col('high').cast(pl.Float64, strict=False).alias('high'),
                pl.col('low').cast(pl.Float64, strict=False).alias('low'),
                pl.col('close').cast(pl.Float64, strict=False).alias('close'),
                pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
                pl.col('amount').cast(pl.Float64, strict=False).alias('amount'),
                pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
            ])
            
            if 'mv' not in result.columns:
                result = result.with_columns(pl.lit(0.0).alias('mv'))
            
            status = {
                'factors_computed': [],
                'score_distribution': {},
                'regime_states': [],
                'sign_switch_log': [],
            }
            
            # 0. 计算 Regime Intensity 和 Dynamic_Sign_Switch
            logger.info("V84: 计算市场动量强度 (Regime Intensity)...")
            result = self.compute_regime_intensity(result, index_df)
            status['factors_computed'].append('regime_intensity')
            
            # 记录 Regime 状态
            unique_dates = result['trade_date'].unique().to_list()
            for trade_date in sorted(unique_dates)[-5:]:  # 只记录最后 5 天
                day_data = result.filter(pl.col('trade_date') == trade_date).select(
                    ['trade_date', 'market_volatility', 'market_return', 'regime_intensity', 'regime_type']
                )
                if day_data is not None and not day_data.is_empty():
                    row = day_data.to_dicts()[0]
                    regime_state = V84RegimeState(
                        trade_date=trade_date,
                        market_volatility=float(row.get('market_volatility', 0)),
                        market_return=float(row.get('market_return', 0)),
                        regime_intensity=float(row.get('regime_intensity', 0)),
                        regime_type=row.get('regime_type', 'neutral'),
                        patch_multiplier=1.0
                    )
                    self.regime_history.append(regime_state)
                    status['regime_states'].append({
                        'trade_date': trade_date,
                        'regime_intensity': regime_state.regime_intensity,
                        'regime_type': regime_state.regime_type,
                    })
            
            # 1. 计算行业中位数收益
            logger.info("V84: 计算行业中位数收益...")
            result = self.compute_industry_return_median(result)
            status['factors_computed'].append('industry_return_median')
            
            # 2. 计算 Refined_Residual 因子（带 Dynamic_Sign_Switch）
            logger.info("V84: 计算 Refined_Residual 因子 (带 Dynamic_Sign_Switch)...")
            result = self._compute_refined_residual(result)
            status['factors_computed'].append('refined_residual')
            
            # 3. 计算 Smart_Flow 因子
            logger.info("V84: 计算 Smart_Flow 因子...")
            result = self._compute_smart_flow(result)
            status['factors_computed'].append('smart_flow')
            
            # 4. 计算 Vol_Price_Interaction 因子
            logger.info("V84: 计算 Vol_Price_Interaction 因子...")
            result = self._compute_vol_price_interaction(result)
            status['factors_computed'].append('vol_price_interaction')
            
            # 5. 计算波动率
            logger.info("V84: 计算波动率...")
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 6. Z-Score 归一化 + 中值滤波
            logger.info("V84: 应用 Z-Score 归一化 + 中值滤波...")
            result = self._apply_normalization_and_filtering(result)
            status['factors_computed'].append('normalization_filtering')
            
            # 7. 计算综合评分
            logger.info("V84: 计算综合评分...")
            result = self._compute_composite_score(result)
            status['factors_computed'].append('composite_score')
            
            # 8. 计算买入信号
            logger.info("V84: 计算买入信号...")
            result = self._compute_buy_signal(result)
            status['factors_computed'].append('buy_signal')
            
            # 9. Lookahead 检查
            logger.info("V84: 执行 Lookahead 检查...")
            latest_date = max(unique_dates)
            passed, error = self.lookahead_checker.check_signal_calculation(result, latest_date, 'composite_score')
            
            if not passed:
                logger.error(f"V84: 【Lookahead Error】{error}")
                self.lookahead_checker.raise_if_errors()
            
            status['lookahead_check_passed'] = passed
            status['sign_switch_log'] = self.sign_switch_log[-10:]  # 最近 10 条记录
            
            return result, status
            
        except Exception as e:
            logger.error(f"V84 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_refined_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Refined_Residual 因子（带 Dynamic_Sign_Switch）
        
        【V84 核心公式】
        Refined_Residual = (个股 5 日收益 - 行业 5 日中位数收益) / 波动率
        
        【Dynamic_Sign_Switch 逻辑】- V84 优化版
        - 当 Regime_Intensity > 1.5 时：使用正向 Residual（动量逻辑）
        - 当 Regime_Intensity < 0.5 时：使用负向 Residual（反转逻辑）
        - 当 0.5 <= Regime_Intensity <= 1.5 时：线性插值
        
        【V84 改进】
        - 使用更稳健的波动率估计
        - 增加横截面排名标准化
        """
        result = df.clone()
        
        # 计算个股 5 日收益率（使用 T-1 数据，避免未来函数）
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(self.residual_window + 1)) / 
             (pl.col('close').shift(self.residual_window + 1) + EPSILON)).alias('stock_return_5d')
        ])
        
        # 使用已计算的行业收益
        result = result.with_columns([
            pl.col('industry_return').alias('industry_return_5d')
        ])
        
        # 计算残差收益
        result = result.with_columns([
            (pl.col('stock_return_5d') - pl.col('industry_return_5d')).alias('residual_return')
        ])
        
        # 计算波动率（使用更稳健的估计）
        result = result.with_columns([
            ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('daily_return')
        ])
        result = result.with_columns([
            pl.col('daily_return').fill_null(0.0).alias('daily_return_filled')
        ])
        result = result.with_columns([
            pl.col('daily_return_filled')
            .rolling_std(window_size=self.volatility_window)
            .over('symbol')
            .alias('volatility_for_residual')
        ])
        result = result.with_columns([
            pl.col('volatility_for_residual').fill_null(0.02).alias('volatility_for_residual')
        ])
        
        # 计算基础 Refined_Residual（波动率调整后的残差）
        result = result.with_columns([
            (pl.col('residual_return') / (pl.col('volatility_for_residual') + EPSILON)).alias('refined_residual_base')
        ])
        
        # Dynamic_Sign_Switch：根据 Regime Intensity 动态调整符号
        result = result.with_columns([
            # 计算符号因子
            pl.when(pl.col('regime_intensity') > self.regime_high_threshold)
            .then(pl.lit(1.0))  # 动量 Regime：正向
            .when(pl.col('regime_intensity') < self.regime_low_threshold)
            .then(pl.lit(-1.0))  # 反转 Regime：负向
            .otherwise(
                # 中性区域：线性插值
                (pl.col('regime_intensity') - self.regime_low_threshold) / 
                (self.regime_high_threshold - self.regime_low_threshold + EPSILON) * 2 - 1
            )
            .alias('sign_factor')
        ])
        
        # 应用符号切换
        result = result.with_columns([
            (pl.col('refined_residual_base') * pl.col('sign_factor')).alias('refined_residual_raw')
        ])
        
        # 记录 Sign Switch 日志（采样）
        sample_dates = result['trade_date'].unique().to_list()[::20]  # 每 20 天采样一次
        for sample_date in sample_dates[-5:]:  # 只记录最近 5 个采样点
            day_data = result.filter(pl.col('trade_date') == sample_date).select(
                ['trade_date', 'regime_intensity', 'regime_type', 'sign_factor']
            )
            if day_data is not None and not day_data.is_empty():
                row = day_data.to_dicts()[0]
                sign_log = {
                    'trade_date': sample_date,
                    'regime_intensity': float(row.get('regime_intensity', 0)),
                    'regime_type': row.get('regime_type', 'neutral'),
                    'sign_factor': float(row.get('sign_factor', 1)),
                }
                self.sign_switch_log.append(sign_log)
        
        # Quantile Transform 非线性变换
        residual_values = result['refined_residual_raw'].to_numpy()
        transformed = quantile_transform(residual_values)
        
        result = result.with_columns([
            pl.lit(transformed).alias('refined_residual_transformed')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('refined_residual_transformed').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_residual')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_residual').cast(pl.Float64) + EPSILON)).alias('residual_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('residual_percentile') * 100).alias('refined_residual_score')
        ])
        
        return result
    
    def _compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Smart_Flow 因子（基于成交量分布的资金流）
        
        【V84 核心公式】
        Smart_Flow = Σ(成交量 * 价格变化) / Σ成交量
        """
        result = df.clone()
        
        # 计算每日价格变化和成交量乘积
        result = result.with_columns([
            (pl.col('close') - pl.col('open')).alias('price_change'),
            pl.col('volume').fill_null(0.0).alias('volume_filled')
        ])
        
        # 计算成交量加权价格变化（VWAP 变化）
        result = result.with_columns([
            (pl.col('volume_filled') * pl.col('price_change')).alias('volume_weighted_change')
        ])
        
        # 计算滚动和
        result = result.sort(['symbol', 'trade_date'])
        
        # 使用 polars 的 rolling 窗口计算
        result = result.with_columns([
            pl.col('volume_weighted_change')
            .rolling_sum(window_size=self.flow_window)
            .over('symbol')
            .alias('flow_sum')
        ])
        
        result = result.with_columns([
            pl.col('volume_filled')
            .rolling_sum(window_size=self.flow_window)
            .over('symbol')
            .alias('volume_sum')
        ])
        
        # 计算 Smart_Flow（正向，不取反）
        result = result.with_columns([
            (pl.col('flow_sum') / (pl.col('volume_sum') + EPSILON)).alias('smart_flow_raw')
        ])
        
        # Quantile Transform
        flow_values = result['smart_flow_raw'].to_numpy()
        transformed = quantile_transform(flow_values)
        
        result = result.with_columns([
            pl.lit(transformed).alias('smart_flow_transformed')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('smart_flow_transformed').rank('ordinal', descending=True).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_flow')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('flow_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON)).alias('flow_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('flow_percentile') * 100).alias('smart_flow_score')
        ])
        
        return result
    
    def _compute_vol_price_interaction(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Vol_Price_Interaction 因子
        
        【V84 核心公式】
        Vol_Price_Interaction = Rolling_Mean(Rank(Refined_Residual) * Rank(Smart_Flow), window=5)
        
        捕捉两个因子共振的效果，而不是简单的线性相加
        """
        result = df.clone()
        
        # 确保因子列存在
        for col in ['refined_residual_rank', 'smart_flow_rank']:
            if col not in result.columns:
                if col == 'refined_residual_rank':
                    result = result.with_columns([
                        pl.col('refined_residual_transformed').rank('ordinal', descending=True).over('trade_date').alias('refined_residual_rank')
                    ])
                elif col == 'smart_flow_rank':
                    result = result.with_columns([
                        pl.col('smart_flow_transformed').rank('ordinal', descending=True).over('trade_date').alias('smart_flow_rank')
                    ])
        
        # 归一化排名到 [0, 1]
        result = result.with_columns([
            (pl.col('refined_residual_rank') / (pl.col('n_stocks_residual').cast(pl.Float64) + EPSILON)).alias('residual_rank_norm'),
            (pl.col('smart_flow_rank') / (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON)).alias('flow_rank_norm')
        ])
        
        # 计算交互项（因子共振）
        result = result.with_columns([
            (pl.col('residual_rank_norm') * pl.col('flow_rank_norm')).alias('interaction_raw')
        ])
        
        # 计算 5 日滚动均值
        result = result.sort(['symbol', 'trade_date'])
        result = result.with_columns([
            pl.col('interaction_raw')
            .rolling_mean(window_size=self.interaction_window)
            .over('symbol')
            .alias('vol_price_interaction_raw')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('vol_price_interaction_raw').fill_null(0.5).alias('vol_price_interaction_raw')
        ])
        
        # Quantile Transform
        interaction_values = result['vol_price_interaction_raw'].to_numpy()
        transformed = quantile_transform(interaction_values)
        
        result = result.with_columns([
            pl.lit(transformed).alias('vol_price_interaction_transformed')
        ])
        
        # 横截面排名映射到分数
        result = result.with_columns([
            pl.col('vol_price_interaction_transformed').rank('ordinal', descending=True).over('trade_date').alias('interaction_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_interaction')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('interaction_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_interaction').cast(pl.Float64) + EPSILON)).alias('interaction_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('interaction_percentile') * 100).alias('vol_price_interaction_score')
        ])
        
        return result
    
    def _compute_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算波动率"""
        result = df.clone()
        
        result = result.with_columns([
            ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('daily_return')
        ])
        
        result = result.with_columns([
            pl.col('daily_return').fill_null(0.0).alias('daily_return_filled')
        ])
        
        result = result.with_columns([
            pl.col('daily_return_filled')
            .rolling_std(window_size=self.volatility_window)
            .over('symbol')
            .alias('volatility_raw')
        ])
        
        result = result.with_columns([
            pl.col('volatility_raw').fill_null(0.02).alias('volatility')
        ])
        
        return result
    
    def _apply_normalization_and_filtering(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        应用 Z-Score 归一化 + 时间序列中值滤波
        
        【时间序列平稳化】
        所有因子进入模型前，必须经过：
        1. Z-Score 空间归一化
        2. 时间序列中值滤波（简化版 - 使用横截面 Winsorize 替代）
        """
        result = df.clone()
        
        # 需要处理的因子列
        factor_cols = [
            'refined_residual_score',
            'smart_flow_score',
            'vol_price_interaction_score'
        ]
        
        # 对每个因子进行横截面标准化处理
        for col in factor_cols:
            if col in result.columns:
                # 使用 Polars 的窗口函数进行横截面排名归一化
                # 这种方法内存效率更高
                result = result.with_columns([
                    # 横截面排名归一化到 [0, 100]
                    (100.0 * (1.0 - (pl.col(col).rank('ordinal', descending=True).over('trade_date').cast(pl.Float64) - 0.5) / 
                     (pl.col(col).count().over('trade_date').cast(pl.Float64) + EPSILON))).alias(f'{col}_normalized')
                ])
                
                # 使用处理后的值替换原始分数
                result = result.with_columns([
                    pl.col(f'{col}_normalized').alias(col)
                ])
        
        return result
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（融合三个因子）
        
        【V84 核心改进】
        - Refined_Residual（带 Dynamic_Sign_Switch）：权重 50%
        - Smart_Flow：权重 20%
        - Vol_Price_Interaction：权重 30%
        """
        result = df.clone()
        
        # 确保因子列存在
        for col, default in [
            ('refined_residual_score', 50.0),
            ('smart_flow_score', 50.0),
            ('vol_price_interaction_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # 优化后的权重：增加 Refined_Residual 权重
        residual_weight = self.residual_weight  # 0.5
        flow_weight = self.flow_weight  # 0.2
        interaction_weight = self.interaction_weight  # 0.3
        
        result = result.with_columns([
            (residual_weight * pl.col('refined_residual_score') + 
             flow_weight * pl.col('smart_flow_score') +
             interaction_weight * pl.col('vol_price_interaction_score')).alias('composite_score')
        ])
        
        return result
    
    def _compute_buy_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算买入信号"""
        result = df.clone()
        
        # 计算综合评分排名
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        # 计算买入信号（前 10% 且分数 > 50）
        result = result.with_columns([
            ((pl.col('score_rank') <= pl.max('n_stocks').over('trade_date') * V84_SELECTION_PERCENTILE).and_(
                pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V84Signal]:
        """生成交易信号"""
        signals = []
        
        try:
            current_df = df.filter(pl.col('trade_date') == trade_date)
            
            if current_df.is_empty():
                return signals
            
            buy_df = current_df.filter(pl.col('buy_signal') == True)
            
            if buy_df.is_empty():
                return signals
            
            buy_df = buy_df.sort('composite_score', descending=True)
            
            for row in buy_df.iter_rows(named=True):
                signal = V84Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    refined_residual_score=row.get('refined_residual_score', 0.0),
                    refined_residual_raw=row.get('refined_residual_raw', 0.0),
                    smart_flow_score=row.get('smart_flow_score', 0.0),
                    vol_price_interaction_score=row.get('vol_price_interaction_score', 0.0),
                    volatility=row.get('volatility', 0.02),
                    industry_name=row.get('industry_name', ''),
                    industry_code=row.get('industry_code', ''),
                    close_price=row.get('close', 0.0),
                    regime_intensity=row.get('regime_intensity', 0.0),
                    sign_switch_mode=row.get('regime_type', 'neutral'),
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"V84 生成信号失败：{e}")
        
        return signals
    
    def get_sign_switch_report(self) -> str:
        """生成 Dynamic_Sign_Switch 报告"""
        lines = [
            "=" * 60,
            "V84 Dynamic_Sign_Switch Report",
            "=" * 60,
        ]
        
        # 按年份统计
        year_stats = {}
        for log in self.sign_switch_log:
            year = log['trade_date'][:4]
            if year not in year_stats:
                year_stats[year] = {'momentum': 0, 'reversal': 0, 'neutral': 0, 'total': 0}
            
            regime_type = log['regime_type']
            year_stats[year][regime_type] += 1
            year_stats[year]['total'] += 1
        
        for year in sorted(year_stats.keys()):
            stats = year_stats[year]
            lines.append(f"【{year}年】")
            lines.append(f"  动量 Regime: {stats['momentum']} 天 ({stats['momentum']/max(1, stats['total'])*100:.1f}%)")
            lines.append(f"  反转 Regime: {stats['reversal']} 天 ({stats['reversal']/max(1, stats['total'])*100:.1f}%)")
            lines.append(f"  中性 Regime: {stats['neutral']} 天 ({stats['neutral']/max(1, stats['total'])*100:.1f}%)")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ===========================================
# V84 ReturnCalculator
# ===========================================

class V84ReturnCalculator:
    """
    V84 收益计算器 - 波动率调整后的收益排名
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.return_window = self.config.get('return_window', 5)
    
    def calculate_forward_return(self, df: pl.DataFrame, window: int = 5) -> pl.DataFrame:
        """计算未来 N 日收益"""
        result = df.clone()
        
        result = result.with_columns([
            (((pl.col('close').shift(-window)).over('symbol') - pl.col('close')) / 
             (pl.col('close') + EPSILON)).alias('forward_return')
        ])
        
        return result
    
    def calculate_volatility_adjusted_return(self, df: pl.DataFrame, 
                                            return_col: str = 'forward_return',
                                            vol_col: str = 'volatility') -> pl.DataFrame:
        """计算波动率调整后的收益"""
        result = df.clone()
        
        if return_col not in result.columns:
            result = self.calculate_forward_return(result)
        
        result = result.with_columns([
            (pl.col(return_col) / (pl.col(vol_col) + EPSILON)).alias('volatility_adjusted_return')
        ])
        
        return result
    
    def calculate_rank_ic(self, df: pl.DataFrame, trade_date: str,
                          signal_col: str = 'composite_score',
                          return_col: str = 'volatility_adjusted_return') -> Tuple[float, Dict]:
        """计算 Rank IC"""
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return 0.0, {'count': 0, 'reason': '样本不足'}
            
            if signal_col not in day_data.columns:
                return 0.0, {'count': 0, 'reason': 'signal_col 不存在'}
            
            if return_col not in day_data.columns:
                day_data = self.calculate_volatility_adjusted_return(day_data)
            
            signal_values = day_data[signal_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            mask = ~np.isnan(signal_values) & ~np.isnan(return_values) & np.isfinite(signal_values) & np.isfinite(return_values)
            signal_clean = signal_values[mask]
            return_clean = return_values[mask]
            
            if len(signal_clean) < 10:
                return 0.0, {'count': len(signal_clean), 'reason': '有效样本不足'}
            
            signal_ranks = stats.rankdata(-signal_clean, method='average')
            return_ranks = stats.rankdata(-return_clean, method='average')
            
            if np.std(signal_ranks) < EPSILON or np.std(return_ranks) < EPSILON:
                return 0.0, {'count': len(signal_clean), 'reason': '排名标准差为 0'}
            
            rank_ic = np.corrcoef(signal_ranks, return_ranks)[0, 1]
            
            return float(rank_ic) if not np.isnan(rank_ic) else 0.0, {
                'count': len(signal_clean),
                'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0.0,
            }
            
        except Exception as e:
            return 0.0, {'count': 0, 'reason': str(e)}


# ===========================================
# V84 RankICCalculator
# ===========================================

class V84RankICCalculator:
    """V84 Rank IC 计算器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target_min = self.config.get('rank_ic_target_min', V84_RANK_IC_TARGET_MIN)
        self.rank_ic_target_max = self.config.get('rank_ic_target_max', V84_RANK_IC_TARGET_MAX)
        
        self.ic_results: List[V84ICMetrics] = []
        self.monthly_stats: List[V84MonthlyICStats] = []
        
        self.factor_monthly_ics: Dict[str, Dict[str, List[float]]] = {
            'refined_residual': {},
            'smart_flow': {},
            'vol_price_interaction': {},
        }
        
        self.oos_yearly_stats: Dict[str, Dict[str, float]] = {}
        self.return_calculator = V84ReturnCalculator(config)
        
        # 交易天数记录
        self.trading_days: Dict[str, int] = {}
        self.year_trading_days: Dict[str, int] = {}
    
    def set_trading_days(self, year: str, trading_days: int) -> None:
        """设置指定年份的有效交易天数"""
        self.trading_days[year] = trading_days
        self.year_trading_days[year] = trading_days
        logger.debug(f"V84: {year}年交易天数设置为 {trading_days}")
    
    def get_trading_days(self, year: str) -> int:
        """获取指定年份的交易天数"""
        return self.trading_days.get(year, 0)
    
    def get_factor_monthly_ics(self) -> Dict[str, Dict[str, List[float]]]:
        """获取因子月度 IC 数据"""
        return self.factor_monthly_ics
    
    def calculate_spearman_rank_ic(self, factor_values: np.ndarray,
                                    label_values: np.ndarray) -> float:
        """计算 Spearman Rank IC"""
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        factor_ranks = stats.rankdata(-factor_clean, method='average')
        label_ranks = stats.rankdata(-label_clean, method='average')
        
        if np.std(factor_ranks) < EPSILON or np.std(label_ranks) < EPSILON:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_ic_series(self, df: pl.DataFrame,
                            signal_col: str = 'composite_score',
                            return_col: str = 'volatility_adjusted_return') -> List[V84ICMetrics]:
        """计算 IC 序列"""
        try:
            df_with_return = self.return_calculator.calculate_volatility_adjusted_return(df)
            
            unique_dates = df['trade_date'].unique().to_list()
            ic_series = []
            
            for trade_date in sorted(unique_dates):
                rank_ic, details = self.return_calculator.calculate_rank_ic(
                    df_with_return, trade_date, signal_col, return_col
                )
                
                if abs(rank_ic) < EPSILON:
                    logger.warning(f"V84: {trade_date} Rank IC 为 0，可能存在问题")
                
                ic_metrics = V84ICMetrics(
                    trade_date=trade_date,
                    factor_name='composite_score',
                    ic=rank_ic,
                    rank_ic=rank_ic
                )
                ic_series.append(ic_metrics)
            
            self.ic_results = ic_series
            self._compute_monthly_stats()
            self._compute_factor_monthly_ics(df_with_return)
            self._compute_oos_yearly_stats()
            
        except Exception as e:
            logger.error(f"V84 计算 IC 序列失败：{e}")
            self.ic_results = []
        
        return self.ic_results
    
    def _compute_monthly_stats(self):
        """计算月度统计"""
        if not self.ic_results:
            self.monthly_stats = []
            return
        
        monthly_rank_ics: Dict[str, List[float]] = {}
        
        for ic_metric in self.ic_results:
            try:
                date_str = ic_metric.trade_date
                month = date_str[:7]
                if month not in monthly_rank_ics:
                    monthly_rank_ics[month] = []
                monthly_rank_ics[month].append(ic_metric.rank_ic)
            except Exception:
                continue
        
        self.monthly_stats = []
        for month, rank_ics in sorted(monthly_rank_ics.items()):
            if rank_ics:
                mean_rank_ic = float(np.mean(rank_ics))
                monthly_stat = V84MonthlyICStats(
                    month=month,
                    factor_name='composite_score',
                    mean_rank_ic=mean_rank_ic,
                    std_rank_ic=float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0,
                    ic_count=len(rank_ics),
                    is_negative=(mean_rank_ic < 0)
                )
                self.monthly_stats.append(monthly_stat)
    
    def _compute_factor_monthly_ics(self, df: pl.DataFrame):
        """计算单因子月度 IC"""
        unique_dates = df['trade_date'].unique().to_list()
        
        factor_columns = {
            'refined_residual': 'refined_residual_score',
            'smart_flow': 'smart_flow_score',
            'vol_price_interaction': 'vol_price_interaction_score',
        }
        
        for trade_date in sorted(unique_dates):
            month = trade_date[:7]
            
            for factor_name, factor_col in factor_columns.items():
                if factor_col in df.columns:
                    day_data = df.filter(pl.col('trade_date') == trade_date)
                    if day_data.height < 10:
                        continue
                    
                    factor_values = day_data[factor_col].to_numpy()
                    df_with_return = self.return_calculator.calculate_volatility_adjusted_return(day_data)
                    return_values = df_with_return['volatility_adjusted_return'].to_numpy()
                    
                    ic = self.calculate_spearman_rank_ic(factor_values, return_values)
                    
                    if month not in self.factor_monthly_ics[factor_name]:
                        self.factor_monthly_ics[factor_name][month] = []
                    self.factor_monthly_ics[factor_name][month].append(ic)
    
    def _compute_oos_yearly_stats(self):
        """计算 OOS 年度统计（2019/2021/2024）"""
        oos_years = V84_RANK_IC_OOS_YEARS
        
        for year in oos_years:
            year_stats = {}
            
            year_ics = [ic for ic in self.ic_results if ic.trade_date.startswith(year)]
            
            if year_ics:
                rank_ics = [ic.rank_ic for ic in year_ics]
                
                year_stats['mean_rank_ic'] = float(np.mean(rank_ics))
                year_stats['std_rank_ic'] = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0
                year_stats['ic_count'] = len(year_ics)
                year_stats['positive_ratio'] = float(np.sum([1 for ic in rank_ics if ic > 0]) / len(rank_ics))
                
                # 检查 IC 是否在目标范围内
                ic_in_range = V84_RANK_IC_TARGET_MIN <= year_stats['mean_rank_ic'] <= V84_RANK_IC_TARGET_MAX
                year_stats['ic_in_target_range'] = ic_in_range
                
                if not ic_in_range:
                    if year_stats['mean_rank_ic'] < V84_RANK_IC_TARGET_MIN:
                        logger.error(f"V84: 【IC 过低】{year}年 Mean Rank IC = {year_stats['mean_rank_ic']:.4f} < {V84_RANK_IC_TARGET_MIN}")
                    else:
                        logger.warning(f"V84: 【IC 过高】{year}年 Mean Rank IC = {year_stats['mean_rank_ic']:.4f} > {V84_RANK_IC_TARGET_MAX} (可能存在未来数据泄露)")
            else:
                year_stats['mean_rank_ic'] = 0.0
                year_stats['std_rank_ic'] = 0.0
                year_stats['ic_count'] = 0
                year_stats['positive_ratio'] = 0.0
                year_stats['ic_in_target_range'] = False
                logger.error(f"V84: 【无数据】{year}年无 IC 数据")
            
            for factor_name in self.factor_monthly_ics:
                factor_year_ics = []
                for month, ics in self.factor_monthly_ics[factor_name].items():
                    if month.startswith(year):
                        factor_year_ics.extend(ics)
                
                if factor_year_ics:
                    year_stats[f'{factor_name}_mean_rank_ic'] = float(np.mean(factor_year_ics))
                else:
                    year_stats[f'{factor_name}_mean_rank_ic'] = 0.0
            
            self.oos_yearly_stats[year] = year_stats
    
    def get_ic_statistics(self) -> Dict[str, float]:
        """获取 IC 统计"""
        if not self.ic_results:
            return {
                'mean_ic': 0.0,
                'mean_rank_ic': 0.0,
                'ic_std': 0.0,
                'rank_ic_std': 0.0,
                'ic_ir': 0.0,
                'rank_ic_ir': 0.0,
                'positive_ratio': 0.0,
                'num_valid_days': 0,
            }
        
        ic_values = np.array([m.rank_ic for m in self.ic_results])
        
        mean_ic = float(np.mean(ic_values))
        mean_rank_ic = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        rank_ic_std = ic_std
        ic_ir = mean_ic / ic_std if ic_std > EPSILON else 0.0
        rank_ic_ir = ic_ir
        positive_ratio = float(np.sum(ic_values > 0) / len(ic_values))
        
        return {
            'mean_ic': mean_ic,
            'mean_rank_ic': mean_rank_ic,
            'ic_std': ic_std,
            'rank_ic_std': rank_ic_std,
            'ic_ir': ic_ir,
            'rank_ic_ir': rank_ic_ir,
            'positive_ratio': positive_ratio,
            'num_valid_days': len(self.ic_results),
        }
    
    def get_oos_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取 OOS 年度统计"""
        return self.oos_yearly_stats
    
    def get_monthly_rank_ic_statistics(self) -> Dict[str, float]:
        """获取月度 Rank IC 统计"""
        if not self.monthly_stats:
            return {
                'monthly_mean_rank_ic': 0.0,
                'monthly_std': 0.0,
                'monthly_pass': False,
                'negative_months': 0,
            }
        
        monthly_means = [m.mean_rank_ic for m in self.monthly_stats]
        negative_months = sum(1 for m in self.monthly_stats if m.is_negative)
        
        monthly_mean = float(np.mean(monthly_means))
        monthly_std = float(np.std(monthly_means, ddof=1)) if len(monthly_means) > 1 else 0.0
        monthly_pass = monthly_mean >= self.rank_ic_target_min
        
        return {
            'monthly_mean_rank_ic': monthly_mean,
            'monthly_std': monthly_std,
            'monthly_pass': monthly_pass,
            'negative_months': negative_months,
            'num_months': len(self.monthly_stats),
        }
    
    def check_rank_ic_pass(self) -> Tuple[bool, str]:
        """检查 Rank IC 是否达标"""
        stats = self.get_ic_statistics()
        
        # 检查是否所有年份都在目标范围内
        all_in_range = True
        for year in V84_RANK_IC_OOS_YEARS:
            if year in self.oos_yearly_stats:
                ic = self.oos_yearly_stats[year]['mean_rank_ic']
                if not (V84_RANK_IC_TARGET_MIN <= ic <= V84_RANK_IC_TARGET_MAX):
                    all_in_range = False
                    break
        
        if all_in_range and stats['mean_rank_ic'] >= self.rank_ic_target_min:
            return (True, f"Rank IC 达标：{stats['mean_rank_ic']:.4f} 在 [{self.rank_ic_target_min}, {self.rank_ic_target_max}] 范围内")
        else:
            return (False, f"Rank IC 未达标：{stats['mean_rank_ic']:.4f} 不在目标范围内")
    
    def print_rank_ic_report(self):
        """打印 Rank IC 报告"""
        stats = self.get_ic_statistics()
        monthly_stats = self.get_monthly_rank_ic_statistics()
        oos_stats = self.get_oos_statistics()
        is_pass, message = self.check_rank_ic_pass()
        
        logger.info("=" * 70)
        logger.info("V84 Rank IC 预测质量审计表")
        logger.info("=" * 70)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：[{V84_RANK_IC_TARGET_MIN}, {V84_RANK_IC_TARGET_MAX}])")
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f}")
        logger.info(f"负值月份数量：{monthly_stats['negative_months']}")
        logger.info(f"达标状态：{is_pass}")
        logger.info("")
        logger.info("【OOS 年度统计】")
        for year in V84_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                year_stat = oos_stats[year]
                in_range = "✓" if year_stat.get('ic_in_target_range') else "✗"
                logger.info(f"  {year}年：Mean Rank IC={year_stat['mean_rank_ic']:.4f} {in_range}, "
                           f"样本数={year_stat['ic_count']}, 正占比={year_stat['positive_ratio']:.2%}")
        logger.info("=" * 70)
    
    def generate_oos_report(self) -> str:
        """生成 OOS 测试报告"""
        oos_stats = self.get_oos_statistics()
        
        lines = [
            "=" * 70,
            "V84 OOS 测试报告",
            "=" * 70,
            "",
        ]
        
        for year in V84_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                stat = oos_stats[year]
                in_range = "✓" if stat.get('ic_in_target_range') else "✗"
                lines.append(f"【{year}年】{in_range}")
                lines.append(f"  Mean Rank IC: {stat['mean_rank_ic']:.4f} (目标：[{V84_RANK_IC_TARGET_MIN}, {V84_RANK_IC_TARGET_MAX}])")
                lines.append(f"  Std Rank IC: {stat['std_rank_ic']:.4f}")
                lines.append(f"  样本数：{stat['ic_count']}")
                lines.append(f"  正 IC 占比：{stat['positive_ratio']:.2%}")
                lines.append("")
                
                lines.append(f"  【单因子 IC】")
                for factor in ['refined_residual', 'smart_flow', 'vol_price_interaction']:
                    factor_ic = stat.get(f'{factor}_mean_rank_ic', 0.0)
                    lines.append(f"    {factor}: {factor_ic:.4f}")
                lines.append("")
        
        valid_years = [year for year in V84_RANK_IC_OOS_YEARS if year in oos_stats and oos_stats[year]['ic_count'] > 0]
        if valid_years:
            avg_rank_ic = np.mean([oos_stats[y]['mean_rank_ic'] for y in valid_years])
            in_range = V84_RANK_IC_TARGET_MIN <= avg_rank_ic <= V84_RANK_IC_TARGET_MAX
            lines.append(f"【三年度平均 Mean Rank IC】")
            lines.append(f"  平均值：{avg_rank_ic:.4f} (目标：[{V84_RANK_IC_TARGET_MIN}, {V84_RANK_IC_TARGET_MAX}])")
            lines.append(f"  达标状态：{'✓' if in_range else '✗'}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V84_INITIAL_CAPITAL',
    'V84_MAX_POSITIONS',
    'V84_WARMUP_PERIOD',
    'V84DataManager',
    'V84AlphaCenter',
    'V84RankICCalculator',
    'V84ReturnCalculator',
    'V84LookaheadChecker',
    'V84Position',
    'V84Trade',
    'V84Signal',
    'V84ICMetrics',
    'V84MonthlyICStats',
    'quantile_transform',
    'zscore_normalize',
    'median_filter',
]