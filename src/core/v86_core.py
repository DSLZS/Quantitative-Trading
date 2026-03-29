"""
V86 Core Module - 因子稳定性增强与时空一致性审计

【V86 核心理念】
1. IC 衰减审计 (IC Decay Audit)
   - 计算 T+1, T+2, T+3 的 Rank IC
   - 检测高频噪声：若 T+1 高但 T+2 归零，需通过 Decay_Filter 平滑

2. 动态 Regime 分类器 (Dynamic Regime Classifier)
   - 基于过去 20 日的行业离散度 (Sector Dispersion) 和波动率偏度
   - 高离散度 = 有主线行情 → 坚持动量
   - 低离散度 = 电风扇行情 → 自动切换到反转

3. 行业中性化 2.0 (Industry Neutralization 2.0)
   - 在生成 Final_Score 后进行横截面行业中性化
   - 减去所属行业平均分，确保选股是个股超额而非行业轮动

4. 指数退避重试 (Exponential Backoff Retry)
   - Database connection timeout 或 Data missing 时自动重试
   - 退避公式：delay = base_delay * (2 ^ attempt)

【硬性指标】
- 指标 A (稳定性): 三年度 Mean Rank IC 均值 >= 0.045，且每个年度的 IC IR >= 0.5
- 指标 B (可交易性): T+2 延迟 Rank IC > 0.02（确保信号不是一秒即逝的噪声）
- 指标 C (纯净度): 行业中性化后，最大单一行业持仓占比不得连续 3 天超过 30%

作者：量化系统
版本：V86.0
日期：2026-03-29
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
# V86 配置常量
# ===========================================

V86_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V86_MAX_POSITIONS = 10
V86_WARMUP_PERIOD = 250
V86_MIN_SAMPLE_SIZE = 100
V86_MIN_STOCK_DAILY_ROWS = 500000  # 每年至少 50 万条记录

# 重试配置（指数退避）
V86_RETRY_ATTEMPTS = 5
V86_RETRY_BASE_DELAY = 1.0  # 基础延迟 1 秒
V86_RETRY_MAX_DELAY = 30.0  # 最大延迟 30 秒

# IC 衰减审计配置
V86_IC_DECAY_MAX_LAG = 3  # 计算 T+1, T+2, T+3
V86_IC_DECAY_THRESHOLD = 0.02  # T+2 IC 必须 > 0.02

# Regime 分类器配置
V86_REGIME_LOOKBACK = 20  # 过去 20 日
V86_REGIME_DISPERSION_THRESHOLD = 0.6  # 行业离散度阈值（高/低）
V86_REGIME_VOL_SKEW_THRESHOLD = 0.5  # 波动率偏度阈值

# 行业中性化 2.0 配置
V86_INDUSTRY_NEUTRAL_WINDOW = 20  # 行业中性化窗口
V86_MAX_SINGLE_INDUSTRY_PCT = 0.30  # 最大单一行业占比 30%
V86_MAX_CONSECUTIVE_DAYS_OVER_LIMIT = 3  # 不得连续 3 天超过

# 费率配置（严禁修改）
V86_COMMISSION_RATE = 0.002  # 0.2%
V86_MIN_COMMISSION = 5.0
V86_STAMP_DUTY = 0.0005  # 印花税
V86_TRANSFER_FEE = 0.00001

# 绩效目标
V86_RANK_IC_TARGET_MIN = 0.045  # V86 要求更高
V86_RANK_IC_IR_TARGET = 0.5
V86_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V86ICDecayMetrics:
    """IC 衰减指标"""
    trade_date: str
    factor_name: str
    ic_t1: float  # T+1 IC
    ic_t2: float  # T+2 IC
    ic_t3: float  # T+3 IC
    decay_rate: float  # 衰减率 (T+1 -> T+2)
    is_high_frequency_noise: bool  # 是否为高频噪声


@dataclass
class V86RegimeState:
    """市场状态分类"""
    trade_date: str
    regime_type: str  # "momentum" 或 "reversal"
    sector_dispersion: float  # 行业离散度
    volatility_skew: float  # 波动率偏度
    confidence: float  # 分类置信度
    reason: str  # 分类原因


@dataclass
class V86IndustryExposure:
    """行业暴露度"""
    trade_date: str
    industry_name: str
    industry_code: str
    exposure: float  # 行业暴露度（相对行业平均的超额）
    neutralized_score: float  # 行业中性化后的分数


@dataclass
class V86PositionConcentration:
    """持仓集中度"""
    trade_date: str
    industry_name: str
    industry_code: str
    position_count: int
    position_value: float
    position_pct: float  # 占持仓比例
    is_over_limit: bool  # 是否超过限制


@dataclass
class V86RetryEvent:
    """重试事件记录"""
    operation: str
    attempt: int
    delay: float
    error_message: str
    success: bool


# ===========================================
# V86 工具函数
# ===========================================

def exponential_backoff_delay(attempt: int, base_delay: float = V86_RETRY_BASE_DELAY,
                               max_delay: float = V86_RETRY_MAX_DELAY) -> float:
    """
    计算指数退避延迟
    
    公式：delay = base_delay * (2 ^ attempt)
    
    Parameters
    ----------
    attempt : int
        当前尝试次数（从 0 开始）
    base_delay : float
        基础延迟（秒）
    max_delay : float
        最大延迟（秒）
        
    Returns
    -------
    float
        延迟时间（秒）
    """
    delay = base_delay * (2 ** attempt)
    return min(delay, max_delay)


def retry_with_backoff(max_attempts: int = V86_RETRY_ATTEMPTS,
                       base_delay: float = V86_RETRY_BASE_DELAY,
                       max_delay: float = V86_RETRY_MAX_DELAY,
                       retryable_exceptions: Tuple = None):
    """
    带指数退避的重试装饰器
    
    Parameters
    ----------
    max_attempts : int
        最大尝试次数
    base_delay : float
        基础延迟（秒）
    max_delay : float
        最大延迟（秒）
    retryable_exceptions : Tuple
        可重试的异常类型元组
        
    Returns
    -------
    decorator
        装饰器函数
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    delay = exponential_backoff_delay(attempt, base_delay, max_delay)
                    logger.warning(f"[重试] {func.__name__} 第 {attempt + 1}/{max_attempts} 次失败："
                                   f"{e}，{delay:.1f}秒后重试...")
                    time.sleep(delay)
            
            logger.error(f"[重试耗尽] {func.__name__} 在 {max_attempts} 次尝试后仍失败："
                         f"{last_exception}")
            raise last_exception
        return wrapper
    return decorator


def calculate_sector_dispersion(df: pl.DataFrame, return_col: str = 'pct_chg') -> float:
    """
    计算行业离散度 (Sector Dispersion)
    
    【公式】
    Sector Dispersion = Std(各行业中位数收益)
    
    Parameters
    ----------
    df : pl.DataFrame
        包含行业和收益数据的数据框
    return_col : str
        收益列名
        
    Returns
    -------
    float
        行业离散度
    """
    try:
        # 按行业计算中位数收益
        industry_median = df.group_by('industry_code').agg([
            pl.col(return_col).median().alias('industry_median_return')
        ])
        
        if industry_median.height < 3:
            return 0.0
        
        # 计算行业间离散度
        dispersion = industry_median['industry_median_return'].std()
        return float(dispersion) if dispersion is not None else 0.0
        
    except Exception as e:
        logger.error(f"计算行业离散度失败：{e}")
        return 0.0


def calculate_volatility_skew(df: pl.DataFrame, window: int = 20) -> float:
    """
    计算波动率偏度 (Volatility Skew)
    
    【公式】
    Volatility Skew = Skew(个股波动率)
    
    Parameters
    ----------
    df : pl.DataFrame
        包含股票数据的数据框
    window : int
        波动率计算窗口
        
    Returns
    -------
    float
        波动率偏度
    """
    try:
        # 计算个股波动率
        df_with_vol = df.with_columns([
            ((pl.col('close').shift(1) / (pl.col('close').shift(2) + EPSILON)) - 1).alias('daily_return')
        ])
        
        df_with_vol = df_with_vol.with_columns([
            pl.col('daily_return')
            .rolling_std(window_size=window)
            .over('symbol')
            .alias('volatility')
        ])
        
        vol_values = df_with_vol['volatility'].drop_nulls().to_numpy()
        
        if len(vol_values) < 10:
            return 0.0
        
        # 计算偏度
        skew = stats.skew(vol_values)
        return float(skew) if not np.isnan(skew) else 0.0
        
    except Exception as e:
        logger.error(f"计算波动率偏度失败：{e}")
        return 0.0


def quantile_transform(series: np.ndarray, n_quantiles: int = 1000) -> np.ndarray:
    """分位数映射到正态分布"""
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
    """Z-Score 空间归一化"""
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


# ===========================================
# V86 DataManager
# ===========================================

class V86DataManager:
    """
    V86 数据管理器 - 带指数退避重试
    
    【核心改进】
    - 所有数据库操作都支持指数退避重试
    - 自动检测 Database connection timeout 和 Data missing
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V86_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V86_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V86_RETRY_ATTEMPTS)
        self.retry_base_delay = self.config.get('retry_base_delay', V86_RETRY_BASE_DELAY)
        self.retry_max_delay = self.config.get('retry_max_delay', V86_RETRY_MAX_DELAY)
        self._data_cache: Dict[str, pl.DataFrame] = {}
        
        # 重试事件记录
        self.retry_events: List[V86RetryEvent] = []
    
    def _record_retry_event(self, operation: str, attempt: int, 
                            delay: float, error_message: str, success: bool):
        """记录重试事件"""
        event = V86RetryEvent(
            operation=operation,
            attempt=attempt,
            delay=delay,
            error_message=error_message,
            success=success
        )
        self.retry_events.append(event)
    
    @retry_with_backoff(
        max_attempts=V86_RETRY_ATTEMPTS,
        base_delay=V86_RETRY_BASE_DELAY,
        max_delay=V86_RETRY_MAX_DELAY,
        retryable_exceptions=(Exception,)  # 捕获所有异常进行重试
    )
    def _execute_db_query(self, query: str, operation_name: str) -> pl.DataFrame:
        """
        执行数据库查询（带重试）
        
        Parameters
        ----------
        query : str
            SQL 查询
        operation_name : str
            操作名称（用于日志）
            
        Returns
        -------
        pl.DataFrame
            查询结果
        """
        if self.db is None:
            raise ValueError("数据库连接未初始化")
        
        try:
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"[Data missing] {operation_name}: 查询结果为空")
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            
            # 检测 Database connection timeout
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                raise  # 触发重试
            
            # 检测 Data missing
            if "missing" in error_msg.lower() or "empty" in error_msg.lower():
                raise  # 触发重试
            
            raise
    
    def _calculate_warmup_start_date(self, start_date: str) -> str:
        """计算热身起始日期"""
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
            df = self._execute_db_query(query, f"data_integrity_check_{year}")
            
            daily_count = int(df['cnt'][0])
            
            if daily_count < V86_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count:,} < {V86_MIN_STOCK_DAILY_ROWS:,}"
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
            df = self._execute_db_query(query, f"trading_days_count_{year}")
            
            if df.is_empty():
                return 0
            
            return int(df['days'][0])
            
        except Exception as e:
            logger.error(f"获取交易天数失败：{e}")
            return 0
    
    def load_stock_data(self, start_date: str, end_date: str, 
                        symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载股票数据（带重试）"""
        actual_start_date = self._calculate_warmup_start_date(start_date)
        
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
        
        df = self._execute_db_query(query, f"load_stock_data_{start_date}_{end_date}")
        
        if df.is_empty():
            raise ValueError(f"未加载到任何数据")
        
        return df
    
    def load_industry_mapping(self) -> pl.DataFrame:
        """加载行业映射（带重试）"""
        if self.db is None:
            return self._empty_industry_mapping_df()
        
        query = """
            SELECT DISTINCT symbol, industry_name, industry_code
            FROM stock_industry_daily
            WHERE industry_name IS NOT NULL
        """
        
        try:
            df = self._execute_db_query(query, "load_industry_mapping")
            
            if df.is_empty():
                return self._empty_industry_mapping_df()
            
            return df
            
        except Exception:
            return self._empty_industry_mapping_df()
    
    def load_index_data(self, start_date: str, end_date: str, 
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据（带重试）"""
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
        
        try:
            df = self._execute_db_query(query, f"load_index_data_{index_code}")
            
            if df.is_empty():
                return self._empty_index_df()
            
            return df
            
        except Exception:
            return self._empty_index_df()
    
    def _empty_industry_mapping_df(self) -> pl.DataFrame:
        """返回空的行业映射 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'industry_name': pl.Utf8,
            'industry_code': pl.Utf8
        })
    
    def _empty_index_df(self) -> pl.DataFrame:
        """返回空的指数 DataFrame"""
        return pl.DataFrame(schema={
            'trade_date': pl.Utf8,
            'close': pl.Float64
        })
    
    def get_retry_report(self) -> str:
        """生成重试事件报告"""
        if not self.retry_events:
            return "无重试事件"
        
        lines = [
            "=" * 60,
            "V86 指数退避重试报告",
            "=" * 60,
        ]
        
        success_count = sum(1 for e in self.retry_events if e.success)
        fail_count = len(self.retry_events) - success_count
        
        lines.append(f"总重试次数：{len(self.retry_events)}")
        lines.append(f"成功：{success_count}, 失败：{fail_count}")
        lines.append("")
        
        # 按操作分组统计
        ops: Dict[str, List[V86RetryEvent]] = {}
        for event in self.retry_events:
            if event.operation not in ops:
                ops[event.operation] = []
            ops[event.operation].append(event)
        
        for op, events in ops.items():
            lines.append(f"操作：{op}")
            lines.append(f"  重试次数：{len(events)}")
            max_attempt = max(e.attempt for e in events)
            lines.append(f"  最大尝试：{max_attempt + 1}")
            total_delay = sum(e.delay for e in events)
            lines.append(f"  总延迟：{total_delay:.1f}秒")
            lines.append("")
        
        return "\n".join(lines)


# ===========================================
# V86 AlphaCenter - 核心因子计算
# ===========================================

class V86AlphaCenter:
    """
    V86 AlphaCenter - 因子稳定性增强与时空一致性审计
    
    【核心改进】
    1. IC 衰减审计：计算 T+1, T+2, T+3 的 Rank IC
    2. 动态 Regime 分类器：基于行业离散度和波动率偏度
    3. 行业中性化 2.0：横截面行业调整
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Regime 分类器配置
        self.regime_lookback = self.config.get('regime_lookback', V86_REGIME_LOOKBACK)
        self.regime_dispersion_threshold = self.config.get('regime_dispersion_threshold', V86_REGIME_DISPERSION_THRESHOLD)
        self.regime_vol_skew_threshold = self.config.get('regime_vol_skew_threshold', V86_REGIME_VOL_SKEW_THRESHOLD)
        
        # 行业中性化配置
        self.industry_neutral_window = self.config.get('industry_neutral_window', V86_INDUSTRY_NEUTRAL_WINDOW)
        
        # IC 衰减审计配置
        self.ic_decay_max_lag = self.config.get('ic_decay_max_lag', V86_IC_DECAY_MAX_LAG)
        self.ic_decay_threshold = self.config.get('ic_decay_threshold', V86_IC_DECAY_THRESHOLD)
        
        # 状态记录
        self.regime_history: List[V86RegimeState] = []
        self.ic_decay_history: List[V86ICDecayMetrics] = []
        self.industry_exposure_history: List[V86IndustryExposure] = []
        self.position_concentration_history: List[V86PositionConcentration] = []
        
        # Regime 分类器缓存
        self._regime_cache: Dict[str, V86RegimeState] = {}
        
        logger.info("V86 AlphaCenter 初始化完成")
        logger.info("V86: IC 衰减审计已启用 (T+1, T+2, T+3)")
        logger.info("V86: 动态 Regime 分类器已启用 (行业离散度 + 波动率偏度)")
        logger.info("V86: 行业中性化 2.0 已启用 (横截面行业调整)")
        logger.info(f"V86: Regime 阈值 - 离散度={V86_REGIME_DISPERSION_THRESHOLD}, 偏度={V86_REGIME_VOL_SKEW_THRESHOLD}")
    
    def compute_regime_classifier(self, df: pl.DataFrame, trade_date: str) -> V86RegimeState:
        """
        动态 Regime 分类器
        
        【逻辑】
        1. 计算过去 20 日的行业离散度 (Sector Dispersion)
        2. 计算过去 20 日的波动率偏度 (Volatility Skew)
        3. 高离散度 (> 0.6) = 有主线行情 → 动量策略
        4. 低离散度 (< 0.6) = 电风扇行情 → 反转策略
        
        Parameters
        ----------
        df : pl.DataFrame
            包含行业和收益数据的数据框
        trade_date : str
            交易日
            
        Returns
        -------
        V86RegimeState
            市场状态分类结果
        """
        # 检查缓存
        if trade_date in self._regime_cache:
            return self._regime_cache[trade_date]
        
        try:
            # 获取过去 N 日数据
            lookback_df = df.filter(pl.col('trade_date') <= trade_date).sort('trade_date', descending=True).head(self.regime_lookback)
            
            # 计算行业离散度
            sector_dispersion = calculate_sector_dispersion(lookback_df)
            
            # 计算波动率偏度
            volatility_skew = calculate_volatility_skew(lookback_df)
            
            # 分类逻辑
            # 高离散度 + 高偏度 = 动量行情（有主线）
            # 低离散度 + 低偏度 = 电风扇行情（快速轮动）
            
            is_high_dispersion = sector_dispersion > self.regime_dispersion_threshold
            is_high_skew = volatility_skew > self.regime_vol_skew_threshold
            
            if is_high_dispersion:
                regime_type = "momentum"  # 动量行情
                confidence = min(1.0, sector_dispersion / (self.regime_dispersion_threshold * 2))
                reason = f"行业离散度高 ({sector_dispersion:.4f})，市场有主线行情"
            else:
                regime_type = "reversal"  # 反转行情
                confidence = min(1.0, 1.0 - sector_dispersion / self.regime_dispersion_threshold)
                reason = f"行业离散度低 ({sector_dispersion:.4f})，市场为电风扇行情"
            
            # 考虑波动率偏度的调整
            if is_high_skew and regime_type == "momentum":
                confidence = min(1.0, confidence + 0.1)
                reason += "，波动率偏度确认"
            elif not is_high_skew and regime_type == "reversal":
                confidence = min(1.0, confidence + 0.1)
                reason += "，波动率偏度确认"
            
            regime_state = V86RegimeState(
                trade_date=trade_date,
                regime_type=regime_type,
                sector_dispersion=sector_dispersion,
                volatility_skew=volatility_skew,
                confidence=confidence,
                reason=reason
            )
            
            # 记录历史
            self.regime_history.append(regime_state)
            self._regime_cache[trade_date] = regime_state
            
            logger.debug(f"V86 Regime [{trade_date}]: {regime_type} (dispersion={sector_dispersion:.4f}, skew={volatility_skew:.4f})")
            
            return regime_state
            
        except Exception as e:
            logger.error(f"计算 Regime 分类失败：{e}")
            
            # 返回默认状态
            default_state = V86RegimeState(
                trade_date=trade_date,
                regime_type="momentum",  # 默认动量
                sector_dispersion=0.0,
                volatility_skew=0.0,
                confidence=0.5,
                reason="计算失败，使用默认值"
            )
            self.regime_history.append(default_state)
            return default_state
    
    def compute_industry_neutralization(self, df: pl.DataFrame, 
                                        score_col: str = 'composite_score') -> pl.DataFrame:
        """
        行业中性化 2.0 - 横截面行业调整
        
        【逻辑】
        1. 计算每个行业的平均分数
        2. 将个股分数减去所属行业平均分
        3. 确保选股是个股超额而非行业轮动
        
        Parameters
        ----------
        df : pl.DataFrame
            包含股票数据的数据框
        score_col : str
            综合评分列名
            
        Returns
        -------
        pl.DataFrame
            行业中性化后的数据框
        """
        result = df.clone()
        
        # 确保行业列存在
        if 'industry_code' not in result.columns:
            logger.warning("缺少 industry_code 列，跳过行业中性化")
            return result
        
        if score_col not in result.columns:
            logger.warning(f"缺少 {score_col} 列，跳过行业中性化")
            return result
        
        try:
            # 按行业计算平均分数
            industry_avg = result.group_by(['industry_code', 'trade_date']).agg([
                pl.col(score_col).mean().alias('industry_avg_score')
            ])
            
            # 合并行业平均分
            result = result.join(
                industry_avg.select(['industry_code', 'trade_date', 'industry_avg_score']),
                on=['industry_code', 'trade_date'],
                how='left'
            )
            
            # 计算行业中性化分数（个股超额）
            result = result.with_columns([
                (pl.col(score_col) - pl.col('industry_avg_score')).alias('industry_neutral_score')
            ])
            
            # 填充 NaN
            result = result.with_columns([
                pl.col('industry_neutral_score').fill_null(0.0).alias('industry_neutral_score')
            ])
            
            # 记录行业暴露度
            for trade_date in result['trade_date'].unique().to_list():
                day_data = result.filter(pl.col('trade_date') == trade_date)
                for row in day_data.iter_rows(named=True):
                    exposure = V86IndustryExposure(
                        trade_date=trade_date,
                        industry_name=row.get('industry_name', ''),
                        industry_code=row.get('industry_code', ''),
                        exposure=row.get('industry_avg_score', 0.0),
                        neutralized_score=row.get('industry_neutral_score', 0.0)
                    )
                    self.industry_exposure_history.append(exposure)
            
            logger.info(f"V86: 行业中性化完成，处理 {result.height} 条记录")
            
            return result
            
        except Exception as e:
            logger.error(f"行业中性化失败：{e}")
            return result
    
    def compute_ic_decay(self, df: pl.DataFrame, signal_col: str = 'composite_score',
                         return_col: str = 'forward_return') -> List[V86ICDecayMetrics]:
        """
        IC 衰减审计 - 计算 T+1, T+2, T+3 的 Rank IC
        
        【逻辑】
        1. 计算 T+1 的 Rank IC
        2. 计算 T+2 的 Rank IC
        3. 计算 T+3 的 Rank IC
        4. 计算衰减率 (T+1 -> T+2)
        5. 检测高频噪声 (T+1 高但 T+2 归零)
        
        Parameters
        ----------
        df : pl.DataFrame
            包含信号和收益的数据框
        signal_col : str
            信号列名
        return_col : str
            收益列名（应为未来收益）
            
        Returns
        -------
        List[V86ICDecayMetrics]
            IC 衰减指标列表
        """
        ic_decay_metrics = []
        
        try:
            unique_dates = sorted(df['trade_date'].unique().to_list())
            
            for i, trade_date in enumerate(unique_dates):
                # 获取当日数据
                day_data = df.filter(pl.col('trade_date') == trade_date)
                
                if day_data.height < 10:
                    continue
                
                signal_values = day_data[signal_col].to_numpy()
                
                # 计算 T+1 IC
                ic_t1 = self._calculate_rank_ic_for_day(df, trade_date, signal_values, 1, signal_col, return_col)
                
                # 计算 T+2 IC
                ic_t2 = self._calculate_rank_ic_for_day(df, trade_date, signal_values, 2, signal_col, return_col)
                
                # 计算 T+3 IC
                ic_t3 = self._calculate_rank_ic_for_day(df, trade_date, signal_values, 3, signal_col, return_col)
                
                # 计算衰减率
                if abs(ic_t1) > EPSILON:
                    decay_rate = (ic_t1 - ic_t2) / ic_t1
                else:
                    decay_rate = 0.0
                
                # 检测高频噪声
                # T+1 高 (> 0.04) 但 T+2 归零 (< 0.02)
                is_high_frequency_noise = (abs(ic_t1) > 0.04) and (abs(ic_t2) < 0.02)
                
                metric = V86ICDecayMetrics(
                    trade_date=trade_date,
                    factor_name=signal_col,
                    ic_t1=ic_t1,
                    ic_t2=ic_t2,
                    ic_t3=ic_t3,
                    decay_rate=decay_rate,
                    is_high_frequency_noise=is_high_frequency_noise
                )
                ic_decay_metrics.append(metric)
            
            self.ic_decay_history = ic_decay_metrics
            
            logger.info(f"V86: IC 衰减审计完成，处理 {len(ic_decay_metrics)} 个交易日")
            
            return ic_decay_metrics
            
        except Exception as e:
            logger.error(f"IC 衰减审计失败：{e}")
            return []
    
    def _calculate_rank_ic_for_day(self, df: pl.DataFrame, trade_date: str,
                                    signal_values: np.ndarray, lag: int,
                                    signal_col: str, return_col: str) -> float:
        """
        计算指定 lag 的 Rank IC
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        trade_date : str
            交易日
        signal_values : np.ndarray
            信号值
        lag : int
            滞后天数
        signal_col : str
            信号列名
        return_col : str
            收益列名
            
        Returns
        -------
        float
            Rank IC 值
        """
        try:
            # 获取 lag 天后的收益数据
            future_date_df = df.filter(pl.col('trade_date') == trade_date)
            
            # 计算未来收益（如果 return_col 不存在）
            if return_col not in df.columns:
                # 使用 close 价格计算未来收益
                future_close = df.with_columns([
                    ((pl.col('close').shift(-lag)).over('symbol')).alias('future_close')
                ])
                
                future_date_df = future_date_df.join(
                    future_close.select(['symbol', 'trade_date', 'future_close']),
                    on=['symbol', 'trade_date'],
                    how='left'
                )
                
                future_returns = ((future_date_df['future_close'] - future_date_df['close']) / 
                                  (future_date_df['close'] + EPSILON)).to_numpy()
            else:
                # 使用已有的收益列
                future_returns = future_date_df[return_col].to_numpy()
            
            # 过滤有效数据
            mask = (~np.isnan(signal_values) & ~np.isnan(future_returns) & 
                    np.isfinite(signal_values) & np.isfinite(future_returns))
            
            if np.sum(mask) < 10:
                return 0.0
            
            signal_clean = signal_values[mask]
            return_clean = future_returns[mask]
            
            # 计算 Spearman Rank IC
            signal_ranks = stats.rankdata(-signal_clean, method='average')
            return_ranks = stats.rankdata(-return_clean, method='average')
            
            if np.std(signal_ranks) < EPSILON or np.std(return_ranks) < EPSILON:
                return 0.0
            
            rank_ic = np.corrcoef(signal_ranks, return_ranks)[0, 1]
            
            return float(rank_ic) if not np.isnan(rank_ic) else 0.0
            
        except Exception as e:
            logger.debug(f"计算 Rank IC (lag={lag}) 失败：{e}")
            return 0.0
    
    def compute_position_concentration(self, positions: Dict[str, Any], 
                                        trade_date: str) -> List[V86PositionConcentration]:
        """
        计算持仓集中度
        
        Parameters
        ----------
        positions : Dict[str, Any]
            持仓字典
        trade_date : str
            交易日
            
        Returns
        -------
        List[V86PositionConcentration]
            持仓集中度列表
        """
        concentration_list = []
        
        if not positions:
            return concentration_list
        
        # 按行业分组
        industry_positions: Dict[str, List[Any]] = {}
        for symbol, pos in positions.items():
            industry_code = getattr(pos, 'industry_code', 'UNKNOWN')
            if industry_code not in industry_positions:
                industry_positions[industry_code] = []
            industry_positions[industry_code].append(pos)
        
        # 计算总持仓价值
        total_value = sum(getattr(pos, 'market_value', 0.0) for pos in positions.values())
        
        if total_value < EPSILON:
            return concentration_list
        
        # 计算各行业集中度
        for industry_code, pos_list in industry_positions.items():
            industry_value = sum(getattr(pos, 'market_value', 0.0) for pos in pos_list)
            industry_pct = industry_value / total_value
            
            concentration = V86PositionConcentration(
                trade_date=trade_date,
                industry_name=pos_list[0].industry_name if pos_list else '',
                industry_code=industry_code,
                position_count=len(pos_list),
                position_value=industry_value,
                position_pct=industry_pct,
                is_over_limit=industry_pct > V86_MAX_SINGLE_INDUSTRY_PCT
            )
            concentration_list.append(concentration)
        
        self.position_concentration_history.append(concentration_list)
        
        return concentration_list
    
    def get_regime_adjustment_factor(self, trade_date: str) -> float:
        """
        获取 Regime 调整因子
        
        【逻辑】
        - 动量行情：坚持动量因子（权重 1.0）
        - 反转行情：切换到反转因子（权重 -1.0）
        
        Parameters
        ----------
        trade_date : str
            交易日
            
        Returns
        -------
        float
            调整因子（1.0 或 -1.0）
        """
        regime_state = self._regime_cache.get(trade_date)
        
        if regime_state is None:
            return 1.0  # 默认动量
        
        if regime_state.regime_type == "momentum":
            return 1.0  # 坚持动量
        else:
            return -1.0  # 切换到反转
    
    def get_ic_decay_summary(self) -> Dict[str, Any]:
        """获取 IC 衰减摘要"""
        if not self.ic_decay_history:
            return {
                'mean_ic_t1': 0.0,
                'mean_ic_t2': 0.0,
                'mean_ic_t3': 0.0,
                'mean_decay_rate': 0.0,
                'high_frequency_noise_ratio': 0.0,
            }
        
        ic_t1_list = [m.ic_t1 for m in self.ic_decay_history]
        ic_t2_list = [m.ic_t2 for m in self.ic_decay_history]
        ic_t3_list = [m.ic_t3 for m in self.ic_decay_history]
        decay_rate_list = [m.decay_rate for m in self.ic_decay_history]
        noise_count = sum(1 for m in self.ic_decay_history if m.is_high_frequency_noise)
        
        return {
            'mean_ic_t1': float(np.mean(ic_t1_list)),
            'mean_ic_t2': float(np.mean(ic_t2_list)),
            'mean_ic_t3': float(np.mean(ic_t3_list)),
            'mean_decay_rate': float(np.mean(decay_rate_list)),
            'high_frequency_noise_ratio': float(noise_count / len(self.ic_decay_history)),
            't2_ic_pass': float(np.mean(ic_t2_list)) > V86_IC_DECAY_THRESHOLD,
        }
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """获取 Regime 分类摘要"""
        if not self.regime_history:
            return {
                'momentum_days': 0,
                'reversal_days': 0,
                'momentum_ratio': 0.0,
                'avg_dispersion': 0.0,
                'avg_skew': 0.0,
            }
        
        momentum_days = sum(1 for r in self.regime_history if r.regime_type == "momentum")
        reversal_days = len(self.regime_history) - momentum_days
        dispersion_list = [r.sector_dispersion for r in self.regime_history]
        skew_list = [r.volatility_skew for r in self.regime_history]
        
        return {
            'momentum_days': momentum_days,
            'reversal_days': reversal_days,
            'momentum_ratio': float(momentum_days / len(self.regime_history)),
            'avg_dispersion': float(np.mean(dispersion_list)),
            'avg_skew': float(np.mean(skew_list)),
        }
    
    def check_concentration_violation(self) -> Tuple[bool, List[str]]:
        """
        检查持仓集中度违规
        
        【规则】
        最大单一行业持仓占比不得连续 3 天超过 30%
        
        Returns
        -------
        Tuple[bool, List[str]]
            (是否违规，违规日期列表)
        """
        if not self.position_concentration_history:
            return False, []
        
        violation_dates = []
        consecutive_count = 0
        
        for concentration_list in self.position_concentration_history:
            trade_date = concentration_list[0].trade_date if concentration_list else None
            
            # 检查是否有行业超过限制
            is_over_limit = any(c.is_over_limit for c in concentration_list)
            
            if is_over_limit:
                consecutive_count += 1
                if consecutive_count >= V86_MAX_CONSECUTIVE_DAYS_OVER_LIMIT:
                    violation_dates.append(trade_date)
            else:
                consecutive_count = 0
        
        is_violated = len(violation_dates) > 0
        
        return is_violated, violation_dates


# ===========================================
# V86 RankICCalculator - IC 衰减审计
# ===========================================

class V86RankICCalculator:
    """
    V86 Rank IC 计算器 - 支持 IC 衰减审计
    
    【核心改进】
    - 计算 T+1, T+2, T+3 的 Rank IC
    - 输出 IC 衰减率
    - 检测高频噪声
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        
        self.ic_decay_max_lag = self.config.get('ic_decay_max_lag', V86_IC_DECAY_MAX_LAG)
        self.ic_decay_threshold = self.config.get('ic_decay_threshold', V86_IC_DECAY_THRESHOLD)
        
        self.ic_results: List[Dict] = []
        self.ic_decay_results: List[V86ICDecayMetrics] = []
        
        # 年度 IC 统计
        self.year_ic_stats: Dict[str, Dict[str, float]] = {}
        
        # IC IR 统计
        self.year_ic_ir: Dict[str, float] = {}
    
    def calculate_ic_decay_series(self, df: pl.DataFrame,
                                   signal_col: str = 'composite_score') -> List[V86ICDecayMetrics]:
        """
        计算 IC 衰减序列
        
        Parameters
        ----------
        df : pl.DataFrame
            包含信号的数据框
        signal_col : str
            信号列名
            
        Returns
        -------
        List[V86ICDecayMetrics]
            IC 衰减指标列表
        """
        try:
            # 添加未来收益列
            df_with_return = self._add_forward_returns(df)
            
            unique_dates = sorted(df['trade_date'].unique().to_list())
            ic_decay_results = []
            
            for trade_date in unique_dates:
                day_data = df_with_return.filter(pl.col('trade_date') == trade_date)
                
                if day_data.height < 10:
                    continue
                
                signal_values = day_data[signal_col].to_numpy()
                
                # 计算 T+1, T+2, T+3 IC
                ic_t1 = self._calculate_spearman_ic(day_data, signal_values, 'forward_return_1')
                ic_t2 = self._calculate_spearman_ic(day_data, signal_values, 'forward_return_2')
                ic_t3 = self._calculate_spearman_ic(day_data, signal_values, 'forward_return_3')
                
                # 计算衰减率
                decay_rate = (ic_t1 - ic_t2) / ic_t1 if abs(ic_t1) > EPSILON else 0.0
                
                # 检测高频噪声
                is_high_frequency_noise = (abs(ic_t1) > 0.04) and (abs(ic_t2) < 0.02)
                
                metric = V86ICDecayMetrics(
                    trade_date=trade_date,
                    factor_name=signal_col,
                    ic_t1=ic_t1,
                    ic_t2=ic_t2,
                    ic_t3=ic_t3,
                    decay_rate=decay_rate,
                    is_high_frequency_noise=is_high_frequency_noise
                )
                ic_decay_results.append(metric)
            
            self.ic_decay_results = ic_decay_results
            
            # 计算年度统计
            self._compute_year_ic_stats()
            
            logger.info(f"V86: IC 衰减审计完成，处理 {len(ic_decay_results)} 个交易日")
            
            return ic_decay_results
            
        except Exception as e:
            logger.error(f"计算 IC 衰减序列失败：{e}")
            logger.error(traceback.format_exc())
            return []
    
    def _add_forward_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """添加 T+1, T+2, T+3 未来收益列"""
        result = df.clone()
        
        # 确保 close 列存在
        if 'close' not in result.columns:
            return result
        
        result = result.sort(['symbol', 'trade_date'])
        
        # T+1 收益
        result = result.with_columns([
            (((pl.col('close').shift(-1)).over('symbol') - pl.col('close')) / 
             (pl.col('close') + EPSILON)).alias('forward_return_1')
        ])
        
        # T+2 收益
        result = result.with_columns([
            (((pl.col('close').shift(-2)).over('symbol') - pl.col('close')) / 
             (pl.col('close') + EPSILON)).alias('forward_return_2')
        ])
        
        # T+3 收益
        result = result.with_columns([
            (((pl.col('close').shift(-3)).over('symbol') - pl.col('close')) / 
             (pl.col('close') + EPSILON)).alias('forward_return_3')
        ])
        
        return result
    
    def _calculate_spearman_ic(self, day_data: pl.DataFrame, 
                                signal_values: np.ndarray,
                                return_col: str) -> float:
        """计算 Spearman Rank IC"""
        try:
            if return_col not in day_data.columns:
                return 0.0
            
            return_values = day_data[return_col].to_numpy()
            
            # 过滤有效数据
            mask = (~np.isnan(signal_values) & ~np.isnan(return_values) & 
                    np.isfinite(signal_values) & np.isfinite(return_values))
            
            if np.sum(mask) < 10:
                return 0.0
            
            signal_clean = signal_values[mask]
            return_clean = return_values[mask]
            
            # 计算排名
            signal_ranks = stats.rankdata(-signal_clean, method='average')
            return_ranks = stats.rankdata(-return_clean, method='average')
            
            if np.std(signal_ranks) < EPSILON or np.std(return_ranks) < EPSILON:
                return 0.0
            
            # 计算 IC
            rank_ic = np.corrcoef(signal_ranks, return_ranks)[0, 1]
            
            return float(rank_ic) if not np.isnan(rank_ic) else 0.0
            
        except Exception as e:
            logger.debug(f"计算 Spearman IC 失败：{e}")
            return 0.0
    
    def _compute_year_ic_stats(self):
        """计算年度 IC 统计"""
        if not self.ic_decay_results:
            return
        
        for year in V86_RANK_IC_OOS_YEARS:
            year_ic_t1 = []
            year_ic_t2 = []
            year_ic_t3 = []
            
            for metric in self.ic_decay_results:
                if metric.trade_date.startswith(year):
                    year_ic_t1.append(metric.ic_t1)
                    year_ic_t2.append(metric.ic_t2)
                    year_ic_t3.append(metric.ic_t3)
            
            if year_ic_t1:
                mean_ic_t1 = np.mean(year_ic_t1)
                std_ic_t1 = np.std(year_ic_t1, ddof=1) if len(year_ic_t1) > 1 else 0.0
                ic_ir_t1 = mean_ic_t1 / std_ic_t1 if std_ic_t1 > EPSILON else 0.0
                
                mean_ic_t2 = np.mean(year_ic_t2)
                mean_ic_t3 = np.mean(year_ic_t3)
                
                self.year_ic_stats[year] = {
                    'mean_ic_t1': float(mean_ic_t1),
                    'std_ic_t1': float(std_ic_t1),
                    'ic_ir_t1': float(ic_ir_t1),
                    'mean_ic_t2': float(mean_ic_t2),
                    'mean_ic_t3': float(mean_ic_t3),
                    'sample_count': len(year_ic_t1),
                }
                
                self.year_ic_ir[year] = float(ic_ir_t1)
    
    def get_ic_decay_summary(self) -> Dict[str, Any]:
        """获取 IC 衰减摘要"""
        if not self.ic_decay_results:
            return {
                'mean_ic_t1': 0.0,
                'mean_ic_t2': 0.0,
                'mean_ic_t3': 0.0,
                'mean_decay_rate': 0.0,
                'high_frequency_noise_ratio': 0.0,
                't2_ic_pass': False,
            }
        
        ic_t1_list = [m.ic_t1 for m in self.ic_decay_results]
        ic_t2_list = [m.ic_t2 for m in self.ic_decay_results]
        ic_t3_list = [m.ic_t3 for m in self.ic_decay_results]
        decay_rate_list = [m.decay_rate for m in self.ic_decay_results]
        noise_count = sum(1 for m in self.ic_decay_results if m.is_high_frequency_noise)
        
        mean_ic_t2 = float(np.mean(ic_t2_list))
        
        return {
            'mean_ic_t1': float(np.mean(ic_t1_list)),
            'mean_ic_t2': mean_ic_t2,
            'mean_ic_t3': float(np.mean(ic_t3_list)),
            'mean_decay_rate': float(np.mean(decay_rate_list)),
            'high_frequency_noise_ratio': float(noise_count / len(self.ic_decay_results)),
            't2_ic_pass': mean_ic_t2 > V86_IC_DECAY_THRESHOLD,
        }
    
    def get_year_ic_stats(self) -> Dict[str, Dict[str, float]]:
        """获取年度 IC 统计"""
        return self.year_ic_stats
    
    def check_ic_ir_target(self) -> Tuple[bool, Dict[str, float]]:
        """
        检查 IC IR 目标
        
        【要求】
        每个年度的 IC IR >= 0.5
        
        Returns
        -------
        Tuple[bool, Dict[str, float]]
            (是否达标，各年度 IC IR)
        """
        if not self.year_ic_ir:
            return False, {}
        
        all_pass = all(ic_ir >= V86_RANK_IC_IR_TARGET for ic_ir in self.year_ic_ir.values())
        
        return all_pass, self.year_ic_ir
    
    def generate_ic_decay_report(self) -> str:
        """生成 IC 衰减审计报告"""
        summary = self.get_ic_decay_summary()
        year_stats = self.get_year_ic_stats()
        ic_ir_pass, ic_ir_values = self.check_ic_ir_target()
        
        lines = [
            "=" * 70,
            "V86 IC 衰减审计报告",
            "=" * 70,
            "",
            "【整体 IC 衰减统计】",
            f"  T+1 Rank IC: {summary['mean_ic_t1']:.4f}",
            f"  T+2 Rank IC: {summary['mean_ic_t2']:.4f} (目标：> {V86_IC_DECAY_THRESHOLD})",
            f"  T+3 Rank IC: {summary['mean_ic_t3']:.4f}",
            f"  平均衰减率：{summary['mean_decay_rate']:.2%}",
            f"  高频噪声占比：{summary['high_frequency_noise_ratio']:.2%}",
            "",
            f"  指标 B (T+2 IC > 0.02): {'✓' if summary['t2_ic_pass'] else '✗'}",
            "",
            "【年度 IC IR 统计】",
        ]
        
        for year in V86_RANK_IC_OOS_YEARS:
            if year in year_stats:
                stat = year_stats[year]
                ic_ir = stat.get('ic_ir_t1', 0.0)
                ic_ir_pass_mark = '✓' if ic_ir >= V86_RANK_IC_IR_TARGET else '✗'
                
                lines.append(f"  {year}年:")
                lines.append(f"    T+1 IC: {stat['mean_ic_t1']:.4f}, Std: {stat['std_ic_t1']:.4f}")
                lines.append(f"    IC IR: {ic_ir:.4f} {ic_ir_pass_mark} (目标：>= {V86_RANK_IC_IR_TARGET})")
                lines.append(f"    T+2 IC: {stat['mean_ic_t2']:.4f}")
                lines.append(f"    T+3 IC: {stat['mean_ic_t3']:.4f}")
                lines.append(f"    样本数：{stat['sample_count']}")
                lines.append("")
        
        lines.append(f"  指标 A (IC IR >= 0.5): {'✓' if ic_ir_pass else '✗'}")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V86_INITIAL_CAPITAL',
    'V86_MAX_POSITIONS',
    'V86_WARMUP_PERIOD',
    'V86DataManager',
    'V86AlphaCenter',
    'V86RankICCalculator',
    'V86ICDecayMetrics',
    'V86RegimeState',
    'V86IndustryExposure',
    'V86PositionConcentration',
    'V86RetryEvent',
    'exponential_backoff_delay',
    'retry_with_backoff',
    'calculate_sector_dispersion',
    'calculate_volatility_skew',
    'quantile_transform',
    'zscore_normalize',
]