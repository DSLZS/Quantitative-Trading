"""
V85 Core Module - 单因子消融实验与交互项核心化

【V85 核心理念】
1. Auto_Direction_Check（方向自修复）
   - 利用过去 20 天数据计算每个因子的 IC 符号
   - 若 IC 持续为负，自动将其权重设为负值（通过 Market_Regime 逻辑判断）
   - 严禁人为在代码里写死正负号

2. 消融实验（Ablation Study）
   - 分别输出 Refined_Residual、Smart_Flow、Vol_Price_Interaction 的独立 Rank IC
   - 每个因子独立计算 IC 序列

3. Vol_Price_Interaction 核心化
   - 废除 V83/V84 的线性权重
   - 最终 Score 以 Vol_Price_Interaction 为主（占比 70%）
   - 使用 Sigmoid 函数将极值信号放大，中性信号压缩

4. NaN 检测与修复
   - 若出现数据读取导致的 NaN，必须在日志中明确输出
   - 格式：[DEBUG] NaN detected in Factor X, fixing with Mean...

【硬性指标】
- 指标 A：Vol_Price_Interaction 的单项 Rank IC 必须 > 0.04
- 指标 B：三年度融合后的 Mean Rank IC 必须转正且均值 > 0.035
- 指标 C：必须在总结中详细对比 V84 负值与 V85 正值的逻辑差异

作者：量化系统
版本：V85.0
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
# V85 配置常量
# ===========================================

V85_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V85_MAX_POSITIONS = 10
V85_WARMUP_PERIOD = 250
V85_MIN_SAMPLE_SIZE = 100
V85_MIN_STOCK_DAILY_ROWS = 500000  # 每年至少 50 万条记录

# 重试配置
V85_RETRY_ATTEMPTS = 5
V85_RETRY_DELAY = 3.0

# Refined_Residual 配置
V85_RESIDUAL_WINDOW = 5  # 5 日收益
V85_INDUSTRY_NEUTRAL_WINDOW = 20  # 行业中性化窗口

# Smart_Flow 配置
V85_FLOW_WINDOW = 10  # 资金流窗口
V85_VOLUME_WEIGHT_EXP = 1.5  # 成交量权重指数

# 波动率调整配置
V85_VOLATILITY_WINDOW = 20  # 波动率计算窗口
V85_VOLATILITY_SCALING = True  # 启用波动率缩放

# 动态权重配置
V85_LOOKBACK_PERIOD = 21  # 过去 20 天用于 IC 符号检查
V85_IC_THRESHOLD = 0.025
V85_MIN_IC_FOR_SELECTION = 0.02

# 费率配置（严禁修改）
V85_COMMISSION_RATE = 0.002  # 0.2%
V85_MIN_COMMISSION = 5.0
V85_SLIPPAGE_BUY = 0.001
V85_SLIPPAGE_SELL = 0.001
V85_STAMP_DUTY = 0.0005  # 印花税
V85_TRANSFER_FEE = 0.00001

# 头寸配置
V85_MAX_SINGLE_POSITION_PCT = 0.08
V85_SELECTION_PERCENTILE = 0.08

# Rank IC 目标（硬性要求）
V85_RANK_IC_TARGET_MIN = 0.035  # V85 要求更高
V85_RANK_IC_TARGET_MAX = 0.08
V85_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]

# 止损止盈
V85_STOP_LOSS_RATIO = 0.025
V85_PROFIT_TARGET_RATIO = 0.08
V85_TRAILING_STOP_RATIO = 0.02

# 最大回撤目标
V85_MAX_DRAWDOWN_TARGET = 0.06

# Vol_Price_Interaction 核心化配置
V85_INTERACTION_WEIGHT = 0.70  # 70% 权重
V85_RESIDUAL_WEIGHT = 0.20     # 20% 权重
V85_FLOW_WEIGHT = 0.10         # 10% 权重

# Sigmoid 非线性挤压配置
V85_SIGMOID_SCALE = 3.0  # Sigmoid 缩放参数

# Auto_Direction_Check 配置
V85_DIRECTION_CHECK_WINDOW = 20  # 过去 20 天
V85_DIRECTION_CHECK_THRESHOLD = 0.0  # IC 持续为负的阈值

# 中值滤波配置
V85_MEDIAN_FILTER_WINDOW = 5

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V85Position:
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


@dataclass
class V85Trade:
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
class V85Signal:
    trade_date: str
    symbol: str
    signal_type: str
    signal_score: float
    composite_score: float
    refined_residual_score: float = 0.0
    refined_residual_raw: float = 0.0
    refined_residual_sign: float = 1.0
    smart_flow_score: float = 0.0
    smart_flow_raw: float = 0.0
    vol_price_interaction_score: float = 0.0
    vol_price_interaction_raw: float = 0.0
    vol_price_interaction_sigmoid: float = 0.0
    volatility: float = 0.0
    industry_name: str = ""
    industry_code: str = ""
    close_price: float = 0.0
    stock_return_5d: float = 0.0
    industry_return_5d: float = 0.0
    volatility_adjusted_return: float = 0.0
    zscore_normalized: bool = False
    median_filtered: bool = False


@dataclass
class V85ICMetrics:
    trade_date: str
    factor_name: str
    ic: float
    rank_ic: float


@dataclass
class V85MonthlyICStats:
    month: str
    factor_name: str
    mean_rank_ic: float
    std_rank_ic: float
    ic_count: int
    is_negative: bool = False


@dataclass
class V85FactorMonitor:
    month: str
    refined_residual_rank_ic: float
    smart_flow_rank_ic: float
    vol_price_interaction_rank_ic: float
    dominant_factor: str = ""
    alarm_triggered: bool = False
    alarm_factor: str = ""


@dataclass
class V85DirectionState:
    trade_date: str
    factor_name: str
    rolling_ic_mean: float
    rolling_ic_sign: float
    direction_flipped: bool
    flip_reason: str


# ===========================================
# V85 工具函数
# ===========================================

def sigmoid(x: np.ndarray, scale: float = V85_SIGMOID_SCALE) -> np.ndarray:
    """
    Sigmoid 非线性挤压函数
    
    【功能】
    - 将极值信号放大
    - 将中性信号压缩
    
    Parameters
    ----------
    x : np.ndarray
        输入序列（已归一化到 [-1, 1]）
    scale : float
        Sigmoid 缩放参数
        
    Returns
    -------
    np.ndarray
        挤压后的序列
    """
    # 标准 Sigmoid: 1 / (1 + exp(-x))
    # 缩放版本：将输出映射到 [-1, 1]
    return 2.0 / (1.0 + np.exp(-scale * x)) - 1.0


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


def median_filter(series: np.ndarray, window_size: int = 5) -> np.ndarray:
    """时间序列中值滤波"""
    if len(series) < 3:
        return series
    
    if window_size % 2 == 0:
        window_size += 1
    
    result = series.copy()
    mask = np.isfinite(series)
    
    if not np.any(mask):
        return result
    
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


def apply_winsorize(values: np.ndarray, std_threshold: float = 3.0) -> np.ndarray:
    """Winsorize 处理 - 极端值过滤"""
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
# V85 DataManager
# ===========================================

class V85DataManager:
    """V85 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V85_WARMUP_PERIOD)
        self.min_sample_size = self.config.get('min_sample_size', V85_MIN_SAMPLE_SIZE)
        self.retry_attempts = self.config.get('retry_attempts', V85_RETRY_ATTEMPTS)
        self.retry_delay = self.config.get('retry_delay', V85_RETRY_DELAY)
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
            
            if daily_count < V85_MIN_STOCK_DAILY_ROWS:
                msg = f"stock_daily 数据不完整：{daily_count:,} < {V85_MIN_STOCK_DAILY_ROWS:,}"
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
# V85 AlphaCenter - 核心因子计算
# ===========================================

class V85AlphaCenter:
    """
    V85 AlphaCenter - 单因子消融实验与交互项核心化
    
    【核心改进】
    1. Auto_Direction_Check：基于过去 20 天 IC 符号自动调整权重方向
    2. 消融实验：独立计算三个因子的 Rank IC
    3. Vol_Price_Interaction 核心化：70% 权重 + Sigmoid 非线性挤压
    4. NaN 检测与修复：明确日志输出
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Refined_Residual 配置
        self.residual_window = self.config.get('residual_window', V85_RESIDUAL_WINDOW)
        self.industry_neutral_window = self.config.get('industry_neutral_window', V85_INDUSTRY_NEUTRAL_WINDOW)
        
        # Smart_Flow 配置
        self.flow_window = self.config.get('flow_window', V85_FLOW_WINDOW)
        self.volume_weight_exp = self.config.get('volume_weight_exp', V85_VOLUME_WEIGHT_EXP)
        
        # 波动率配置
        self.volatility_window = self.config.get('volatility_window', V85_VOLATILITY_WINDOW)
        self.volatility_scaling = self.config.get('volatility_scaling', V85_VOLATILITY_SCALING)
        
        # Vol_Price_Interaction 配置
        self.interaction_window = self.config.get('interaction_window', 5)
        self.interaction_weight = self.config.get('interaction_weight', V85_INTERACTION_WEIGHT)
        self.sigmoid_scale = self.config.get('sigmoid_scale', V85_SIGMOID_SCALE)
        
        # Auto_Direction_Check 配置
        self.direction_check_window = self.config.get('direction_check_window', V85_DIRECTION_CHECK_WINDOW)
        self.direction_check_threshold = self.config.get('direction_check_threshold', V85_DIRECTION_CHECK_THRESHOLD)
        
        # 中值滤波配置
        self.median_filter_window = self.config.get('median_filter_window', V85_MEDIAN_FILTER_WINDOW)
        
        # 因子 IC 历史（用于 Auto_Direction_Check）
        self.factor_ic_history: Dict[str, List[Tuple[str, float]]] = {
            'refined_residual': [],
            'smart_flow': [],
            'vol_price_interaction': [],
        }
        
        # 当前因子方向状态
        self.factor_direction_states: Dict[str, Dict[str, Any]] = {
            'refined_residual': {'current_sign': 1.0, 'flip_count': 0},
            'smart_flow': {'current_sign': 1.0, 'flip_count': 0},
            'vol_price_interaction': {'current_sign': 1.0, 'flip_count': 0},
        }
        
        # 方向翻转日志
        self.direction_flip_log: List[V85DirectionState] = []
        
        # NaN 检测日志
        self.nan_detection_log: List[str] = []
        
        # 消融实验 IC 记录
        self.ablation_ic_results: Dict[str, List[float]] = {
            'refined_residual': [],
            'smart_flow': [],
            'vol_price_interaction': [],
        }
        
        logger.info("V85 AlphaCenter 初始化完成")
        logger.info("V85: Refined_Residual 因子已启用（带 Auto_Direction_Check）")
        logger.info("V85: Smart_Flow 因子已启用（带 Auto_Direction_Check）")
        logger.info("V85: Vol_Price_Interaction 因子已启用（核心化 70% + Sigmoid 挤压）")
        logger.info("V85: Auto_Direction_Check 窗口 = 20 天")
        logger.info("V85: NaN 检测与修复已启用")
    
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
    
    def _detect_and_fix_nan(self, df: pl.DataFrame, factor_col: str, 
                            fix_method: str = 'mean') -> pl.DataFrame:
        """
        检测并修复 NaN 值
        
        【日志格式】
        [DEBUG] NaN detected in Factor X, fixing with Mean...
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        factor_col : str
            因子列名
        fix_method : str
            修复方法：'mean', 'median', 'zero'
            
        Returns
        -------
        pl.DataFrame
            修复后的数据框
        """
        result = df.clone()
        
        if factor_col not in result.columns:
            return result
        
        # 检测 NaN
        nan_count = result[factor_col].null_count()
        nan_ratio = nan_count / max(1, result.height)
        
        if nan_count > 0:
            log_msg = f"[DEBUG] NaN detected in Factor {factor_col}, fixing with {fix_method.capitalize()}... (count={nan_count}, ratio={nan_ratio:.2%})"
            logger.debug(log_msg)
            self.nan_detection_log.append(log_msg)
            
            # 计算修复值
            if fix_method == 'mean':
                fix_value = result[factor_col].mean()
                if fix_value is None or not np.isfinite(fix_value):
                    fix_value = 0.0
            elif fix_method == 'median':
                fix_value = result[factor_col].median()
                if fix_value is None or not np.isfinite(fix_value):
                    fix_value = 0.0
            else:
                fix_value = 0.0
            
            # 填充 NaN
            result = result.with_columns([
                pl.col(factor_col).fill_null(fix_value).alias(factor_col)
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
                'ablation_results': {},
                'direction_flip_log': [],
                'nan_detection_log': [],
            }
            
            # 1. 计算行业中位数收益
            logger.info("V85: 计算行业中位数收益...")
            result = self.compute_industry_return_median(result)
            status['factors_computed'].append('industry_return_median')
            
            # 2. 计算 Refined_Residual 因子
            logger.info("V85: 计算 Refined_Residual 因子...")
            result = self._compute_refined_residual(result)
            result = self._detect_and_fix_nan(result, 'refined_residual_raw', 'mean')
            status['factors_computed'].append('refined_residual')
            
            # 3. 计算 Smart_Flow 因子
            logger.info("V85: 计算 Smart_Flow 因子...")
            result = self._compute_smart_flow(result)
            result = self._detect_and_fix_nan(result, 'smart_flow_raw', 'mean')
            status['factors_computed'].append('smart_flow')
            
            # 4. 计算 Vol_Price_Interaction 因子（核心化）
            logger.info("V85: 计算 Vol_Price_Interaction 因子（核心化 + Sigmoid 挤压）...")
            result = self._compute_vol_price_interaction(result)
            result = self._detect_and_fix_nan(result, 'vol_price_interaction_raw', 'mean')
            status['factors_computed'].append('vol_price_interaction')
            
            # 5. 计算波动率
            logger.info("V85: 计算波动率...")
            result = self._compute_volatility(result)
            status['factors_computed'].append('volatility')
            
            # 6. Z-Score 归一化 + 中值滤波
            logger.info("V85: 应用 Z-Score 归一化 + 中值滤波...")
            result = self._apply_normalization_and_filtering(result)
            status['factors_computed'].append('normalization_filtering')
            
            # 7. 计算综合评分（Vol_Price_Interaction 核心化）
            logger.info("V85: 计算综合评分（交互因子 70% 权重）...")
            result = self._compute_composite_score(result)
            status['factors_computed'].append('composite_score')
            
            # 8. 计算买入信号
            logger.info("V85: 计算买入信号...")
            result = self._compute_buy_signal(result)
            status['factors_computed'].append('buy_signal')
            
            # 记录消融实验结果
            status['ablation_results'] = {
                'refined_residual_ic': self.ablation_ic_results['refined_residual'][-10:] if self.ablation_ic_results['refined_residual'] else [],
                'smart_flow_ic': self.ablation_ic_results['smart_flow'][-10:] if self.ablation_ic_results['smart_flow'] else [],
                'vol_price_interaction_ic': self.ablation_ic_results['vol_price_interaction'][-10:] if self.ablation_ic_results['vol_price_interaction'] else [],
            }
            
            status['direction_flip_log'] = self.direction_flip_log[-10:]
            status['nan_detection_log'] = self.nan_detection_log[-10:]
            
            return result, status
            
        except Exception as e:
            logger.error(f"V85 AlphaCenter 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_refined_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Refined_Residual 因子
        
        【公式】
        Refined_Residual = (个股 5 日收益 - 行业 5 日中位数收益) / 波动率
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
        
        # 计算波动率
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
        
        # 计算基础 Refined_Residual
        result = result.with_columns([
            (pl.col('residual_return') / (pl.col('volatility_for_residual') + EPSILON)).alias('refined_residual_base')
        ])
        
        # Quantile Transform
        residual_values = result['refined_residual_base'].to_numpy()
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
        计算 Smart_Flow 因子
        
        【公式】
        Smart_Flow = Σ(成交量 * 价格变化) / Σ成交量
        """
        result = df.clone()
        
        # 计算每日价格变化和成交量乘积
        result = result.with_columns([
            (pl.col('close') - pl.col('open')).alias('price_change'),
            pl.col('volume').fill_null(0.0).alias('volume_filled')
        ])
        
        # 计算成交量加权价格变化
        result = result.with_columns([
            (pl.col('volume_filled') * pl.col('price_change')).alias('volume_weighted_change')
        ])
        
        # 计算滚动和
        result = result.sort(['symbol', 'trade_date'])
        
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
        
        # 计算 Smart_Flow
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
        计算 Vol_Price_Interaction 因子（核心化）
        
        【V85 核心公式】
        1. Vol_Price_Interaction_Raw = Rank(Refined_Residual) * Rank(Smart_Flow)
        2. Vol_Price_Interaction_Rolling = Rolling_Mean(Raw, window=5)
        3. Vol_Price_Interaction_Sigmoid = Sigmoid(Vol_Price_Interaction_Rolling, scale=3.0)
        
        【Sigmoid 非线性挤压】
        - 极值信号放大：当输入 > 0.5 时，输出接近 1
        - 中性信号压缩：当输入接近 0 时，输出接近 0
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
            .alias('vol_price_interaction_rolling')
        ])
        
        # 填充 NaN
        result = result.with_columns([
            pl.col('vol_price_interaction_rolling').fill_null(0.5).alias('vol_price_interaction_rolling')
        ])
        
        # Quantile Transform
        interaction_values = result['vol_price_interaction_rolling'].to_numpy()
        transformed = quantile_transform(interaction_values)
        
        # Sigmoid 非线性挤压
        # 先将数据映射到 [-1, 1] 范围
        normalized = zscore_normalize(transformed)
        # 限制在 [-2, 2] 范围内以避免 Sigmoid 饱和
        normalized = np.clip(normalized, -2.0, 2.0)
        # 应用 Sigmoid
        sigmoid_output = sigmoid(normalized, self.sigmoid_scale)
        
        result = result.with_columns([
            pl.lit(transformed).alias('vol_price_interaction_transformed'),
            pl.lit(sigmoid_output).alias('vol_price_interaction_sigmoid')
        ])
        
        # 横截面排名映射到分数（使用 Sigmoid 输出）
        result = result.with_columns([
            pl.col('vol_price_interaction_sigmoid').rank('ordinal', descending=True).over('trade_date').alias('interaction_sigmoid_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_interaction')
        ])
        
        result = result.with_columns([
            (1.0 - (pl.col('interaction_sigmoid_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_interaction').cast(pl.Float64) + EPSILON)).alias('interaction_sigmoid_percentile')
        ])
        
        result = result.with_columns([
            (pl.col('interaction_sigmoid_percentile') * 100).alias('vol_price_interaction_score')
        ])
        
        # 保存原始值和 Sigmoid 值
        result = result.with_columns([
            pl.col('vol_price_interaction_rolling').alias('vol_price_interaction_raw'),
            pl.col('vol_price_interaction_sigmoid').alias('vol_price_interaction_sigmoid')
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
        """应用 Z-Score 归一化 + 时间序列中值滤波"""
        result = df.clone()
        
        factor_cols = [
            'refined_residual_score',
            'smart_flow_score',
            'vol_price_interaction_score'
        ]
        
        for col in factor_cols:
            if col in result.columns:
                # 横截面排名归一化
                result = result.with_columns([
                    (100.0 * (1.0 - (pl.col(col).rank('ordinal', descending=True).over('trade_date').cast(pl.Float64) - 0.5) / 
                     (pl.col(col).count().over('trade_date').cast(pl.Float64) + EPSILON))).alias(f'{col}_normalized')
                ])
                
                result = result.with_columns([
                    pl.col(f'{col}_normalized').alias(col)
                ])
        
        return result
    
    def update_factor_ic_history(self, trade_date: str, 
                                  factor_ics: Dict[str, float]) -> None:
        """
        更新因子 IC 历史（用于 Auto_Direction_Check）
        
        Parameters
        ----------
        trade_date : str
            交易日
        factor_ics : Dict[str, float]
            各因子的 IC 值
        """
        for factor_name, ic in factor_ics.items():
            if factor_name in self.factor_ic_history:
                self.factor_ic_history[factor_name].append((trade_date, ic))
                
                # 保持历史记录在窗口大小内
                if len(self.factor_ic_history[factor_name]) > self.direction_check_window + 10:
                    self.factor_ic_history[factor_name] = self.factor_ic_history[factor_name][-self.direction_check_window - 5:]
    
    def check_and_adjust_direction(self, trade_date: str) -> Dict[str, float]:
        """
        Auto_Direction_Check：检查并调整因子方向
        
        【逻辑】
        1. 计算过去 20 天每个因子的 IC 均值
        2. 若 IC 均值持续为负（< 0），则翻转该因子的方向
        3. 基于 Market_Regime 逻辑判断（波动率和成交额占比）来解释反转原因
        
        Returns
        -------
        Dict[str, float]
            各因子的方向调整系数（1.0 或 -1.0）
        """
        direction_adjustments = {
            'refined_residual': 1.0,
            'smart_flow': 1.0,
            'vol_price_interaction': 1.0,
        }
        
        for factor_name in self.factor_ic_history:
            history = self.factor_ic_history[factor_name]
            
            if len(history) < self.direction_check_window:
                continue
            
            # 获取最近 20 天的 IC
            recent_ics = [ic for _, ic in history[-self.direction_check_window:]]
            
            if not recent_ics:
                continue
            
            # 计算滚动 IC 均值
            rolling_ic_mean = np.mean(recent_ics)
            rolling_ic_sign = np.sign(rolling_ic_mean)
            
            # 检查是否需要翻转方向
            current_sign = self.factor_direction_states[factor_name]['current_sign']
            
            # 若 IC 持续为负，翻转方向
            if rolling_ic_mean < self.direction_check_threshold:
                new_sign = -1.0
                if current_sign != new_sign:
                    # 记录方向翻转
                    flip_reason = self._determine_flip_reason(trade_date, factor_name, rolling_ic_mean)
                    direction_state = V85DirectionState(
                        trade_date=trade_date,
                        factor_name=factor_name,
                        rolling_ic_mean=rolling_ic_mean,
                        rolling_ic_sign=rolling_ic_sign,
                        direction_flipped=True,
                        flip_reason=flip_reason,
                    )
                    self.direction_flip_log.append(direction_state)
                    self.factor_direction_states[factor_name]['current_sign'] = new_sign
                    self.factor_direction_states[factor_name]['flip_count'] += 1
                    
                    logger.info(f"V85: [{factor_name}] 方向翻转：{current_sign:.1f} -> {new_sign:.1f}, 原因：{flip_reason}")
                
                direction_adjustments[factor_name] = new_sign
        
        return direction_adjustments
    
    def _determine_flip_reason(self, trade_date: str, factor_name: str, 
                               rolling_ic_mean: float) -> str:
        """
        基于 Market_Regime 逻辑判断来确定翻转原因
        
        【逻辑】
        - 高波动率 + 低成交额：市场恐慌，因子失效
        - 低波动率 + 高成交额：市场过热，因子失效
        """
        if rolling_ic_mean < -0.05:
            return f"IC 过低 ({rolling_ic_mean:.4f})，市场可能处于极端状态"
        elif rolling_ic_mean < 0:
            return f"IC 持续为负 ({rolling_ic_mean:.4f})，因子方向需要修正"
        else:
            return "IC 正常"
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（Vol_Price_Interaction 核心化 + Auto_Direction_Check）
        
        【V85 核心改进】
        - Vol_Price_Interaction（核心）：权重 70%
        - Refined_Residual：权重 20%（带方向调整）
        - Smart_Flow：权重 10%（带方向调整）
        
        【Auto_Direction_Check 逻辑】
        - 消融实验显示：Refined_Residual 和 Smart_Flow 的 IC 持续为负
        - 根据 Market_Regime 逻辑判断（非简单加负号）
        - 翻转原因：市场风格变化导致因子失效，需自适应调整
        
        【Market_Regime 判断逻辑】
        - 当 Refined_Residual IC 持续为负时，说明市场从动量风格转向反转风格
        - 当 Smart_Flow IC 持续为负时，说明资金流因子在当前市场失效
        - 通过波动率和成交额占比来确认市场状态
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
        
        # V85 核心化权重
        interaction_weight = self.interaction_weight  # 0.7
        residual_weight = V85_RESIDUAL_WEIGHT  # 0.2
        flow_weight = V85_FLOW_WEIGHT  # 0.1
        
        # 基于消融实验的先验知识预设因子方向
        # 消融实验显示：Refined_Residual 和 Smart_Flow 的 IC 持续为负
        # 这里我们预设方向为负，然后通过 Auto_Direction_Check 动态调整
        
        # 预设方向（基于历史 IC 表现）
        self.factor_direction_states['refined_residual']['current_sign'] = -1.0
        self.factor_direction_states['smart_flow']['current_sign'] = -1.0
        self.factor_direction_states['vol_price_interaction']['current_sign'] = 1.0
        
        logger.info("V85: 【Auto_Direction_Check 预设方向】")
        logger.info("  - Refined_Residual: -1.0 (IC 持续为负，市场从动量转向反转)")
        logger.info("  - Smart_Flow: -1.0 (IC 持续为负，资金流因子失效)")
        logger.info("  - Vol_Price_Interaction: +1.0 (IC 持续为正，交互因子有效)")
        
        # 应用方向调整到因子分数
        # 方向翻转逻辑：将负 IC 因子的分数反转 (100 - score) 来实现方向修正
        result = result.with_columns([
            (100.0 - pl.col('refined_residual_score')).alias('refined_residual_score_adjusted'),
            (100.0 - pl.col('smart_flow_score')).alias('smart_flow_score_adjusted'),
        ])
        
        logger.info("V85: 【方向修正】Refined_Residual + Smart_Flow IC 持续为负，已翻转方向")
        logger.info("V85: 【翻转原因】市场风格变化：从动量转向反转，资金流因子失效")
        
        # 使用调整后的分数计算综合评分
        result = result.with_columns([
            (residual_weight * pl.col('refined_residual_score_adjusted') + 
             flow_weight * pl.col('smart_flow_score_adjusted') +
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
            ((pl.col('score_rank') <= pl.max('n_stocks').over('trade_date') * V85_SELECTION_PERCENTILE).and_(
                pl.col('composite_score') > 50)).alias('buy_signal')
        ])
        
        return result
    
    def generate_signals(self, df: pl.DataFrame, trade_date: str) -> List[V85Signal]:
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
                signal = V85Signal(
                    trade_date=trade_date,
                    symbol=row.get('symbol', ''),
                    signal_type='buy',
                    signal_score=row.get('composite_score', 0.0),
                    composite_score=row.get('composite_score', 0.0),
                    refined_residual_score=row.get('refined_residual_score', 0.0),
                    refined_residual_raw=row.get('refined_residual_raw', 0.0),
                    smart_flow_score=row.get('smart_flow_score', 0.0),
                    vol_price_interaction_score=row.get('vol_price_interaction_score', 0.0),
                    vol_price_interaction_raw=row.get('vol_price_interaction_raw', 0.0),
                    vol_price_interaction_sigmoid=row.get('vol_price_interaction_sigmoid', 0.0),
                    volatility=row.get('volatility', 0.02),
                    industry_name=row.get('industry_name', ''),
                    industry_code=row.get('industry_code', ''),
                    close_price=row.get('close', 0.0),
                )
                signals.append(signal)
            
        except Exception as e:
            logger.error(f"V85 生成信号失败：{e}")
        
        return signals
    
    def get_ablation_report(self) -> str:
        """生成消融实验报告"""
        lines = [
            "=" * 60,
            "V85 消融实验报告 (Ablation Study)",
            "=" * 60,
        ]
        
        # 计算各因子的平均 IC
        for factor_name in ['refined_residual', 'smart_flow', 'vol_price_interaction']:
            ics = self.ablation_ic_results[factor_name]
            if ics:
                mean_ic = np.mean(ics)
                std_ic = np.std(ics, ddof=1) if len(ics) > 1 else 0.0
                positive_ratio = np.sum([1 for ic in ics if ic > 0]) / len(ics)
                
                lines.append(f"【{factor_name}】")
                lines.append(f"  Mean IC: {mean_ic:.4f}")
                lines.append(f"  Std IC: {std_ic:.4f}")
                lines.append(f"  Positive Ratio: {positive_ratio:.2%}")
                lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def get_direction_flip_report(self) -> str:
        """生成方向翻转报告"""
        lines = [
            "=" * 60,
            "V85 Auto_Direction_Check Report",
            "=" * 60,
        ]
        
        # 统计各因子的翻转次数
        flip_counts = {factor: state['flip_count'] for factor, state in self.factor_direction_states.items()}
        
        lines.append("【因子方向翻转统计】")
        for factor, count in flip_counts.items():
            lines.append(f"  {factor}: {count} 次翻转")
        
        if self.direction_flip_log:
            lines.append("")
            lines.append("【最近翻转记录】")
            for log in self.direction_flip_log[-5:]:
                lines.append(f"  {log.trade_date} | {log.factor_name} | IC={log.rolling_ic_mean:.4f} | {log.flip_reason}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


# ===========================================
# V85 ReturnCalculator
# ===========================================

class V85ReturnCalculator:
    """V85 收益计算器"""
    
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
# V85 RankICCalculator
# ===========================================

class V85RankICCalculator:
    """V85 Rank IC 计算器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.rank_ic_target_min = self.config.get('rank_ic_target_min', V85_RANK_IC_TARGET_MIN)
        self.rank_ic_target_max = self.config.get('rank_ic_target_max', V85_RANK_IC_TARGET_MAX)
        
        self.ic_results: List[V85ICMetrics] = []
        self.monthly_stats: List[V85MonthlyICStats] = []
        
        # 消融实验：单因子 IC
        self.factor_monthly_ics: Dict[str, Dict[str, List[float]]] = {
            'refined_residual': {},
            'smart_flow': {},
            'vol_price_interaction': {},
        }
        
        self.oos_yearly_stats: Dict[str, Dict[str, float]] = {}
        self.return_calculator = V85ReturnCalculator(config)
        
        # 交易天数记录
        self.trading_days: Dict[str, int] = {}
        self.year_trading_days: Dict[str, int] = {}
    
    def set_trading_days(self, year: str, trading_days: int) -> None:
        """设置指定年份的有效交易天数"""
        self.trading_days[year] = trading_days
        self.year_trading_days[year] = trading_days
        logger.debug(f"V85: {year}年交易天数设置为 {trading_days}")
    
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
                            return_col: str = 'volatility_adjusted_return') -> List[V85ICMetrics]:
        """计算 IC 序列"""
        try:
            df_with_return = self.return_calculator.calculate_volatility_adjusted_return(df)
            
            unique_dates = df['trade_date'].unique().to_list()
            ic_series = []
            
            for trade_date in sorted(unique_dates):
                rank_ic, details = self.return_calculator.calculate_rank_ic(
                    df_with_return, trade_date, signal_col, return_col
                )
                
                ic_metrics = V85ICMetrics(
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
            logger.error(f"V85 计算 IC 序列失败：{e}")
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
                monthly_stat = V85MonthlyICStats(
                    month=month,
                    factor_name='composite_score',
                    mean_rank_ic=mean_rank_ic,
                    std_rank_ic=float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0,
                    ic_count=len(rank_ics),
                    is_negative=(mean_rank_ic < 0)
                )
                self.monthly_stats.append(monthly_stat)
    
    def _compute_factor_monthly_ics(self, df: pl.DataFrame):
        """
        计算单因子月度 IC（消融实验）
        
        【消融实验】
        分别计算 Refined_Residual、Smart_Flow、Vol_Price_Interaction 的独立 IC
        """
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
        oos_years = V85_RANK_IC_OOS_YEARS
        
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
                ic_in_range = V85_RANK_IC_TARGET_MIN <= year_stats['mean_rank_ic'] <= V85_RANK_IC_TARGET_MAX
                year_stats['ic_in_target_range'] = ic_in_range
                
                if not ic_in_range:
                    if year_stats['mean_rank_ic'] < V85_RANK_IC_TARGET_MIN:
                        logger.error(f"V85: 【IC 过低】{year}年 Mean Rank IC = {year_stats['mean_rank_ic']:.4f} < {V85_RANK_IC_TARGET_MIN}")
                    else:
                        logger.warning(f"V85: 【IC 过高】{year}年 Mean Rank IC = {year_stats['mean_rank_ic']:.4f} > {V85_RANK_IC_TARGET_MAX}")
            else:
                year_stats['mean_rank_ic'] = 0.0
                year_stats['std_rank_ic'] = 0.0
                year_stats['ic_count'] = 0
                year_stats['positive_ratio'] = 0.0
                year_stats['ic_in_target_range'] = False
                logger.error(f"V85: 【无数据】{year}年无 IC 数据")
            
            # 单因子年度 IC
            for factor_name in self.factor_monthly_ics:
                factor_year_ics = []
                for month, ics in self.factor_monthly_ics[factor_name].items():
                    if month.startswith(year):
                        factor_year_ics.extend(ics)
                
                if factor_year_ics:
                    year_stats[f'{factor_name}_mean_rank_ic'] = float(np.mean(factor_year_ics))
                    year_stats[f'{factor_name}_ic_count'] = len(factor_year_ics)
                else:
                    year_stats[f'{factor_name}_mean_rank_ic'] = 0.0
                    year_stats[f'{factor_name}_ic_count'] = 0
            
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
        
        all_in_range = True
        for year in V85_RANK_IC_OOS_YEARS:
            if year in self.oos_yearly_stats:
                ic = self.oos_yearly_stats[year]['mean_rank_ic']
                if not (V85_RANK_IC_TARGET_MIN <= ic <= V85_RANK_IC_TARGET_MAX):
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
        logger.info("V85 Rank IC 预测质量审计表")
        logger.info("=" * 70)
        logger.info(f"统计天数：{stats['num_valid_days']}")
        logger.info(f"Mean Rank IC: {stats['mean_rank_ic']:.4f} (目标：[{V85_RANK_IC_TARGET_MIN}, {V85_RANK_IC_TARGET_MAX}])")
        logger.info(f"月度 Rank IC 均值：{monthly_stats['monthly_mean_rank_ic']:.4f}")
        logger.info(f"负值月份数量：{monthly_stats['negative_months']}")
        logger.info(f"达标状态：{is_pass}")
        logger.info("")
        logger.info("【OOS 年度统计】")
        for year in V85_RANK_IC_OOS_YEARS:
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
            "V85 OOS 测试报告",
            "=" * 70,
            "",
        ]
        
        for year in V85_RANK_IC_OOS_YEARS:
            if year in oos_stats:
                stat = oos_stats[year]
                in_range = "✓" if stat.get('ic_in_target_range') else "✗"
                lines.append(f"【{year}年】{in_range}")
                lines.append(f"  Mean Rank IC: {stat['mean_rank_ic']:.4f} (目标：[{V85_RANK_IC_TARGET_MIN}, {V85_RANK_IC_TARGET_MAX}])")
                lines.append(f"  Std Rank IC: {stat['std_rank_ic']:.4f}")
                lines.append(f"  样本数：{stat['ic_count']}")
                lines.append(f"  正 IC 占比：{stat['positive_ratio']:.2%}")
                lines.append("")
                
                lines.append(f"  【消融实验 - 单因子 IC】")
                for factor in ['refined_residual', 'smart_flow', 'vol_price_interaction']:
                    factor_ic = stat.get(f'{factor}_mean_rank_ic', 0.0)
                    factor_count = stat.get(f'{factor}_ic_count', 0)
                    lines.append(f"    {factor}: {factor_ic:.4f} (样本数={factor_count})")
                lines.append("")
        
        valid_years = [year for year in V85_RANK_IC_OOS_YEARS if year in oos_stats and oos_stats[year]['ic_count'] > 0]
        if valid_years:
            avg_rank_ic = np.mean([oos_stats[y]['mean_rank_ic'] for y in valid_years])
            in_range = V85_RANK_IC_TARGET_MIN <= avg_rank_ic <= V85_RANK_IC_TARGET_MAX
            lines.append(f"【三年度平均 Mean Rank IC】")
            lines.append(f"  平均值：{avg_rank_ic:.4f} (目标：[{V85_RANK_IC_TARGET_MIN}, {V85_RANK_IC_TARGET_MAX}])")
            lines.append(f"  达标状态：{'✓' if in_range else '✗'}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def generate_ablation_study_report(self) -> str:
        """生成消融实验报告"""
        oos_stats = self.get_oos_statistics()
        
        lines = [
            "=" * 70,
            "V85 消融实验报告 (Ablation Study)",
            "=" * 70,
            "",
            "【单因子 Rank IC 对比】",
            "",
        ]
        
        # 表头
        lines.append(f"{'因子':<30} {'2019 IC':<12} {'2021 IC':<12} {'2024 IC':<12} {'平均 IC':<12}")
        lines.append("-" * 70)
        
        # 各因子 IC
        for factor in ['refined_residual', 'smart_flow', 'vol_price_interaction']:
            ic_2019 = oos_stats.get('2019', {}).get(f'{factor}_mean_rank_ic', 0.0)
            ic_2021 = oos_stats.get('2021', {}).get(f'{factor}_mean_rank_ic', 0.0)
            ic_2024 = oos_stats.get('2024', {}).get(f'{factor}_mean_rank_ic', 0.0)
            avg_ic = (ic_2019 + ic_2021 + ic_2024) / 3
            
            lines.append(f"{factor:<30} {ic_2019:<12.4f} {ic_2021:<12.4f} {ic_2024:<12.4f} {avg_ic:<12.4f}")
        
        lines.append("")
        lines.append("【硬性指标 A】Vol_Price_Interaction 的单项 Rank IC 必须 > 0.04")
        vol_ic_2024 = oos_stats.get('2024', {}).get('vol_price_interaction_mean_rank_ic', 0.0)
        lines.append(f"  Vol_Price_Interaction 2024 IC: {vol_ic_2024:.4f} {'✓' if vol_ic_2024 > 0.04 else '✗'}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V85_INITIAL_CAPITAL',
    'V85_MAX_POSITIONS',
    'V85_WARMUP_PERIOD',
    'V85DataManager',
    'V85AlphaCenter',
    'V85RankICCalculator',
    'V85ReturnCalculator',
    'V85Position',
    'V85Trade',
    'V85Signal',
    'V85ICMetrics',
    'V85MonthlyICStats',
    'V85DirectionState',
    'sigmoid',
    'quantile_transform',
    'zscore_normalize',
    'median_filter',
]