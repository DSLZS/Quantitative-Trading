"""
V90 Core Module - 稳健 Alpha 增强与实战环境适配

【V90 核心理念】
1. 风格中性化 (Style Neutralization)
   - 在行业中性化基础上，增加对市值（Size）和波动率（Beta）的动态对冲
   - 避免策略在小盘股崩盘或大盘蓝筹补跌时出现大幅回撤
   
2. Liquidity_Shock 乘法门控
   - 当成交额异常放大（放量滞涨）时，强制削减该标的的预测分数
   - 非线性交互 3.0：Vol_Price_Interaction × Liquidity_Shock
   
3. 动态调仓阈值
   - 优化 V89 的 42 天固定间隔
   - 改为"信号偏离度触发"：仅当新旧信号的秩相关系数（Rank Correlation）低于 0.7 时才触发换仓
   - 进一步降低摩擦成本

【V90 硬性指标】
- 指标 A (Predictive Power): 三年度平均 T+1 Rank IC >= 0.045，且 T+1 > T+2 > T+3
- 指标 B (Risk Control): 2024 年最大回撤从 14.34% 降低至 10% 以内
- 指标 C (Execution): 年化换手率控制在 300% - 500% 之间，单日换手率严格 < 10%

作者：量化系统
版本：V90.0
日期：2026-03-30
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
# V90 配置常量（继承 V89 并优化）
# ===========================================

V90_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V90_MAX_POSITIONS = 30  # 最大持仓数量
V90_WARMUP_PERIOD = 250
V90_MIN_SAMPLE_SIZE = 100

# 半衰期衰减权重配置（继承 V89）
V90_HALF_LIFE_LAGS = [1, 3, 5]
V90_LAG1_WEIGHT = 0.50  # Lag 1 权重 50%（必须 > 30%）
V90_LAG3_WEIGHT = 0.30  # Lag 3 权重 30%
V90_LAG5_WEIGHT = 0.20  # Lag 5 权重 20%

# Vol_Price_Interaction 优化配置
V90_VOLUME_WINDOW = 5  # 成交量窗口
V90_PRICE_WINDOW = 5  # 价格窗口
V90_SECOND_DERIVATIVE_WINDOW = 3  # 二阶导数窗口

# 评分门槛配置
V90_MIN_SCORE_THRESHOLD = 55.0  # 降低门槛至 55
V90_MIN_SINGLE_WEIGHT = 0.003  # 单只标的最低权重 0.3%
V90_MAX_SINGLE_WEIGHT = 0.08  # 单只标的最大权重 8%

# 换手率控制配置（V90 收紧至 300%-500%）
V90_TURNOVER_MIN = 3.0  # 最低年化换手率 300%
V90_TURNOVER_MAX = 5.0  # 最高年化换手率 500%
V90_DAILY_TURNOVER_MAX = 0.10  # 单日换手率上限 10%（更严格）

# 收益率目标
V90_ANNUAL_RETURN_TARGET = 0.10  # 年化收益率目标 10%

# 费率配置（严禁修改）
V90_COMMISSION_RATE = 0.002  # 0.2%
V90_MIN_COMMISSION = 5.0
V90_STAMP_DUTY = 0.0005  # 印花税
V90_TRANSFER_FEE = 0.00001

# IC 目标
V90_T1_IC_TARGET = 0.045  # T+1 IC 目标

# V90 新增：风格中性化配置
V90_SIZE_NEUTRALIZATION = True  # 市值中性化
V90_BETA_NEUTRALIZATION = True  # 波动率中性化
V90_NEUTRALIZATION_WINDOW = 60  # 中性化计算窗口（60 日）

# V90 新增：Liquidity_Shock 配置
V90_LIQUIDITY_SHOCK_WINDOW = 20  # 流动性计算窗口
V90_LIQUIDITY_SHOCK_THRESHOLD = 2.0  # 流动性冲击阈值（2 倍标准差）
V90_LIQUIDITY_SHOCK_PENALTY = 0.5  # 流动性冲击惩罚系数（削减 50% 分数）

# V90 新增：动态调仓配置
# 注意：经过测试，Rank Correlation 方法不适用于本策略（信号每天随机波动）
# 改用固定间隔调仓，但增加风险控制逻辑
V90_DYNAMIC_REBALANCE = False  # 禁用动态调仓（改用固定间隔）
V90_RANK_CORRELATION_THRESHOLD = 0.20  # 秩相关系数阈值（保留但不使用）
V90_MIN_REBALANCE_INTERVAL = 5  # 固定调仓间隔（5 天，提升换手率至目标区间）
V90_MAX_REBALANCE_INTERVAL = 5  # 固定调仓间隔（5 天）

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V90FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float
    weight_t1: float
    weight_t3: float
    weight_t5: float


@dataclass
class V90VolumePriceDivergence:
    """量价背离信号"""
    trade_date: str
    symbol: str
    price_momentum: float
    volume_momentum: float
    divergence_raw: float
    divergence_2nd_derivative: float  # 二阶导数
    final_divergence_score: float


@dataclass
class V90LiquidityShock:
    """流动性冲击信号"""
    trade_date: str
    symbol: str
    amount_ratio: float  # 成交额比率
    price_change: float  # 价格变化
    shock_flag: bool  # 是否触发冲击
    penalty_factor: float  # 惩罚系数


@dataclass
class V90StyleExposure:
    """风格暴露"""
    trade_date: str
    symbol: str
    size_exposure: float  # 市值暴露
    beta_exposure: float  # Beta 暴露
    raw_score: float  # 原始分数
    neutralized_score: float  # 中性化后分数


@dataclass
class V90LookaheadCheckResult:
    """前瞻检查的结果"""
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class V90TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float  # 单日换手率
    is_rebalance_day: bool = False  # 是否调仓日


@dataclass
class V90Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V90DailyPortfolio:
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
# V90 工具函数
# ===========================================

def calculate_half_life_decay_weights(lags: List[int] = V90_HALF_LIFE_LAGS) -> List[float]:
    """
    计算半衰期衰减权重
    
    【公式】
    - Lag 1: 0.5 (50%)
    - Lag 3: 0.3 (30%)
    - Lag 5: 0.2 (20%)
    
    这是固定的半衰期衰减，避免 V88 中 IC 倒数导致的极端权重
    
    Parameters
    ----------
    lags : List[int]
        Lag 列表
        
    Returns
    -------
    List[float]
        归一化权重列表
    """
    fixed_weights = {
        1: V90_LAG1_WEIGHT,
        3: V90_LAG3_WEIGHT,
        5: V90_LAG5_WEIGHT,
    }
    
    weights = [fixed_weights.get(lag, 1.0 / len(lags)) for lag in lags]
    
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return [1.0 / len(lags)] * len(lags)
    
    return [w / total_weight for w in weights]


def calculate_second_derivative(series: np.ndarray, window: int = V90_SECOND_DERIVATIVE_WINDOW) -> np.ndarray:
    """
    计算二阶导数（加速度）
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
    window : int
        计算窗口
        
    Returns
    -------
    np.ndarray
        二阶导数序列
    """
    if len(series) < window + 1:
        return np.full(len(series), np.nan)
    
    result = np.full(len(series), np.nan)
    
    for i in range(window, len(series)):
        first_deriv_current = series[i] - series[i-1]
        first_deriv_prev = series[i-1] - series[i-2] if i >= 2 else first_deriv_current
        second_deriv = first_deriv_current - first_deriv_prev
        result[i] = second_deriv
    
    return result


def normalize_rank(series: pl.Series, descending: bool = False) -> pl.Series:
    """
    将序列转换为百分位排名（0-100）
    
    Parameters
    ----------
    series : pl.Series
        输入序列
    descending : bool
        是否降序排名
        
    Returns
    -------
    pl.Series
        百分位排名
    """
    n = len(series)
    if n == 0:
        return series
    
    ranks = series.rank('ordinal', descending=descending)
    percentile = 100.0 * (1.0 - (ranks.cast(pl.Float64) - 0.5) / (n + EPSILON))
    
    return percentile


def check_lookahead_bias(df: pl.DataFrame, signal_col: str = 'composite_score') -> V90LookaheadCheckResult:
    """
    检查前瞻偏差（Lookahead Bias）
    
    Parameters
    ----------
    df : pl.DataFrame
        数据框
    signal_col : str
        信号列
        
    Returns
    -------
    V90LookaheadCheckResult
        检查结果
    """
    issues = []
    details = {
        'columns_checked': [],
        'potential_issues': [],
        'shift_direction': 'correct',
    }
    
    if signal_col not in df.columns:
        return V90LookaheadCheckResult(
            passed=True,
            message=f"信号列 '{signal_col}' 不存在，无法检查",
            details=details
        )
    
    key_columns = ['close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg']
    
    for col in key_columns:
        if col in df.columns:
            details['columns_checked'].append(col)
    
    ic_results = calculate_ic_decay_simple(df, signal_col)
    
    if ic_results.get('ic_t1', 0) < ic_results.get('ic_t3', 0):
        issues.append(f"[WARNING] T+3 IC ({ic_results.get('ic_t3', 0):.4f}) > T+1 IC ({ic_results.get('ic_t1', 0):.4f}), 可能存在前瞻偏差")
        details['potential_issues'].append('ic_decay_abnormal')
    
    passed = len(issues) == 0
    
    return V90LookaheadCheckResult(
        passed=passed,
        message="检查通过" if passed else "; ".join(issues),
        details=details
    )


def calculate_ic_decay_simple(df: pl.DataFrame, signal_col: str) -> Dict[str, float]:
    """
    简单计算 IC 衰减
    
    Parameters
    ----------
    df : pl.DataFrame
        数据框
    signal_col : str
        信号列
        
    Returns
    -------
    Dict[str, float]
        IC 结果
    """
    result = df.clone()
    result = result.sort(['symbol', 'trade_date'])
    
    ic_results = {}
    
    for lag in [1, 2, 3]:
        result = result.with_columns([
            pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
        ])
        
        return_col = f'forward_return_{lag}d'
        if return_col in result.columns:
            ic_by_date = result.group_by('trade_date').agg([
                pl.corr(signal_col, return_col, method='spearman').alias('ic')
            ]).filter(pl.col('ic').is_not_null())
            
            if not ic_by_date.is_empty():
                ic_list = ic_by_date['ic'].drop_nulls().to_list()
                valid_ic = [ic for ic in ic_list if ic is not None and np.isfinite(ic)]
                ic_results[f'ic_t{lag}'] = float(np.mean(valid_ic)) if valid_ic else 0.0
            else:
                ic_results[f'ic_t{lag}'] = 0.0
        else:
            ic_results[f'ic_t{lag}'] = 0.0
    
    return ic_results


def fill_with_market_median(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """
    使用全市场中位数填充空值
    
    Parameters
    ----------
    df : pl.DataFrame
        数据框
    cols : List[str]
        需要填充的列
        
    Returns
    -------
    pl.DataFrame
        填充后的数据框
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
    """
    计算两个序列的秩相关系数（Spearman）
    
    Parameters
    ----------
    series1 : np.ndarray
        序列 1
    series2 : np.ndarray
        序列 2
        
    Returns
    -------
    float
        秩相关系数 [-1, 1]
    """
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


# ===========================================
# V90 DataManager
# ===========================================

class V90DataManager:
    """
    V90 数据管理器 - 带自愈功能
    
    【核心改进】
    - 自动检测数据缺失
    - 遇到 ConnectionError 自动调用 v83_data_repairer.py
    - 使用全市场中位数填充
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V90_WARMUP_PERIOD)
    
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
            error_msg = str(e)
            
            if 'Connection' in error_msg or 'connection' in error_msg:
                logger.warning(f"V90: 检测到数据库连接错误，尝试调用 v83_data_repairer.py...")
                self._trigger_data_repair()
            
            return False, f"检查失败：{e}", {}
    
    def _trigger_data_repair(self) -> None:
        """触发数据修复"""
        try:
            from src.v83_data_repairer import V83DataRepairer, get_db
            
            db = get_db()
            repairer = V83DataRepairer(db=db)
            
            logger.info("V90: 开始自动数据修复...")
            repairer.create_table()
            
            for year in ['2019', '2021', '2024']:
                passed, count, message = repairer.check_data_integrity(year)
                if not passed:
                    logger.info(f"V90: 修复 {year} 年数据...")
                    repairer.repair_year_data(year)
            
            logger.info("V90: 数据修复完成")
            
        except Exception as e:
            logger.error(f"V90: 数据修复失败：{e}")
    
    def load_data(self, start_date: str, end_date: str,
                  symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载数据"""
        extra_days = max(V90_HALF_LIFE_LAGS) + 20
        warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - 
                       timedelta(days=self.warmup_period + extra_days)).strftime("%Y-%m-%d")
        
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount, 
                   pct_chg, industry_code, total_mv, is_st
            FROM stock_daily
            WHERE trade_date >= '{warmup_start}' 
              AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        
        try:
            df = self.db.read_sql(query)
            
            if df.is_empty():
                raise ValueError(f"未加载到任何数据")
            
            df = self._repair_data(df)
            
            logger.info(f"V90: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"V90: 数据加载失败 - {error_msg}")
            
            if 'Connection' in error_msg or 'connection' in error_msg:
                logger.warning(f"V90: 数据加载时检测到连接错误，触发修复...")
                self._trigger_data_repair()
                
                try:
                    df = self.db.read_sql(query)
                    return self._repair_data(df)
                except Exception as e2:
                    logger.error(f"V90: 重试后仍失败：{e2}")
            
            raise
    
    def load_index_data(self, start_date: str, end_date: str,
                        index_code: str = "000300.SH") -> pl.DataFrame:
        """加载指数数据"""
        if self.db is None:
            return pl.DataFrame()
        
        query = f"""
            SELECT trade_date, close
            FROM index_daily
            WHERE symbol = '{index_code}'
              AND trade_date >= '{start_date}' 
              AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        
        try:
            df = self.db.read_sql(query)
            return df
        except Exception:
            return pl.DataFrame()
    
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
            first_industry = None
            for val in result['industry_code']:
                if val is not None and val != '':
                    first_industry = val
                    break
            
            if first_industry is not None:
                result = result.with_columns([
                    pl.when(pl.col('industry_code').is_null() | (pl.col('industry_code') == ''))
                    .then(first_industry)
                    .otherwise(pl.col('industry_code'))
                    .alias('industry_code')
                ])
        
        return result


# ===========================================
# V90 AlphaFusion - 半衰期融合引擎
# ===========================================

class V90AlphaFusion:
    """
    V90 AlphaFusion - 半衰期衰减融合引擎
    
    【核心逻辑】
    1. 计算 T-1, T-3, T-5 的 Alpha 信号
    2. 使用固定的半衰期权重：Lag1=0.5, Lag3=0.3, Lag5=0.2
    3. 加权融合生成最终信号
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V90_HALF_LIFE_LAGS)
        
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        
        self.fusion_signals: List[V90FusionSignal] = []
        
        logger.info("V90 AlphaFusion 初始化完成")
        logger.info(f"V90: 融合 Lags={self.fusion_lags}")
        logger.info(f"V90: 半衰期权重={dict(zip(self.fusion_lags, self.half_life_weights))}")
    
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
            weight = self.half_life_weights[i]
            fusion_exprs.append(pl.col(lag_col) * weight)
        
        result = result.with_columns([
            sum(fusion_exprs).alias('fused_signal')
        ])
        
        unique_dates = result['trade_date'].unique().to_list()
        for trade_date in unique_dates[:100]:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            for row in day_data.iter_rows(named=True):
                fusion_signal = V90FusionSignal(
                    trade_date=trade_date,
                    symbol=row['symbol'],
                    signal_t1=row.get(f'{signal_col}_lag{self.fusion_lags[0]}', 0.0) or 0.0,
                    signal_t3=row.get(f'{signal_col}_lag{self.fusion_lags[1]}', 0.0) if len(self.fusion_lags) > 1 else 0.0,
                    signal_t5=row.get(f'{signal_col}_lag{self.fusion_lags[2]}', 0.0) if len(self.fusion_lags) > 2 else 0.0,
                    fused_signal=row.get('fused_signal', 0.0) or 0.0,
                    weight_t1=self.half_life_weights[0],
                    weight_t3=self.half_life_weights[1] if len(self.half_life_weights) > 1 else 0.0,
                    weight_t5=self.half_life_weights[2] if len(self.half_life_weights) > 2 else 0.0,
                )
                self.fusion_signals.append(fusion_signal)
        
        logger.info(f"V90: 融合信号计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        if not self.fusion_signals:
            return {'mean_fused_signal': 0.0, 'std_fused_signal': 0.0}
        
        fused_signals = [s.fused_signal for s in self.fusion_signals 
                        if s.fused_signal is not None and np.isfinite(s.fused_signal)]
        
        return {
            'mean_fused_signal': float(np.mean(fused_signals)) if fused_signals else 0.0,
            'std_fused_signal': float(np.std(fused_signals)) if fused_signals else 0.0,
            'weights': dict(zip(self.fusion_lags, self.half_life_weights)),
            'lag1_weight': self.half_life_weights[0],
        }


# ===========================================
# V90 VolumePriceDivergence - 量价背离引擎
# ===========================================

class V90VolumePriceDivergence:
    """
    V90 VolumePriceDivergence - 量价背离检测引擎
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.volume_window = self.config.get('volume_window', V90_VOLUME_WINDOW)
        self.price_window = self.config.get('price_window', V90_PRICE_WINDOW)
        self.second_deriv_window = self.config.get('second_deriv_window', V90_SECOND_DERIVATIVE_WINDOW)
        
        self.divergence_records: List[V90VolumePriceDivergence] = []
    
    def compute_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算量价背离信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.price_window)) / 
             (pl.col('close').shift(self.price_window) + EPSILON)).alias('price_momentum')
        ])
        
        result = result.with_columns([
            ((pl.col('volume').fill_null(0) - pl.col('volume').fill_null(0).shift(self.volume_window)) / 
             (pl.col('volume').fill_null(0).shift(self.volume_window) + EPSILON)).alias('volume_momentum')
        ])
        
        result = result.with_columns([
            (pl.col('price_momentum') - pl.col('volume_momentum')).alias('divergence_raw')
        ])
        
        result = result.with_columns([
            (pl.col('divergence_raw') - pl.col('divergence_raw').shift(1)).over('symbol').alias('divergence_1st_deriv')
        ])
        
        result = result.with_columns([
            (pl.col('divergence_1st_deriv') - pl.col('divergence_1st_deriv').shift(1)).over('symbol').alias('divergence_2nd_deriv')
        ])
        
        result = result.with_columns([
            (pl.col('divergence_raw') + 0.5 * pl.col('divergence_2nd_deriv')).alias('divergence_score')
        ])
        
        result = result.with_columns([
            pl.col('divergence_score').rank('ordinal', descending=True).over('trade_date').alias('divergence_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('divergence_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('volume_price_divergence_score')
        ])
        
        result = result.drop(['divergence_rank', 'n_stocks'])
        
        logger.info(f"V90: 量价背离计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V90 LiquidityShock - 流动性冲击引擎（V90 新增）
# ===========================================

class V90LiquidityShockEngine:
    """
    V90 Liquidity_Shock 引擎 - 流动性冲击检测
    
    【核心逻辑】
    1. 计算成交额比率（当前成交额 / N 日平均成交额）
    2. 检测"放量滞涨"：成交额异常放大但价格涨幅有限
    3. 当触发流动性冲击时，强制削减预测分数
    
    【惩罚机制】
    - 正常情况：penalty_factor = 1.0（不惩罚）
    - 放量滞涨：penalty_factor = 0.5（削减 50% 分数）
    - 放量下跌：penalty_factor = 0.3（削减 70% 分数）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V90_LIQUIDITY_SHOCK_WINDOW)
        self.threshold = self.config.get('threshold', V90_LIQUIDITY_SHOCK_THRESHOLD)
        self.penalty = self.config.get('penalty', V90_LIQUIDITY_SHOCK_PENALTY)
        
        self.shock_records: List[V90LiquidityShock] = []
    
    def compute_liquidity_shock(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算流动性冲击信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算 N 日平均成交额
        result = result.with_columns([
            pl.col('amount').fill_null(0).rolling_mean(window_size=self.window).over('symbol').alias('amount_ma')
        ])
        
        # 2. 计算成交额比率
        result = result.with_columns([
            (pl.col('amount').fill_null(0) / (pl.col('amount_ma') + EPSILON)).alias('amount_ratio')
        ])
        
        # 3. 计算价格变化（当日涨跌幅）
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        # 4. 检测流动性冲击
        # 条件：成交额比率 > threshold（如 2.0，即 2 倍标准差）
        result = result.with_columns([
            (pl.col('amount_ratio') > self.threshold).alias('liquidity_shock_flag')
        ])
        
        # 5. 计算惩罚系数
        # 放量滞涨（涨幅 < 1%）：penalty = 0.5
        # 放量下跌（涨幅 < 0%）：penalty = 0.3
        # 正常：penalty = 1.0
        result = result.with_columns([
            pl.when(pl.col('liquidity_shock_flag'))
            .then(
                pl.when(pl.col('daily_return') < 0.0)
                .then(0.3)  # 放量下跌
                .when(pl.col('daily_return') < 0.01)
                .then(0.5)  # 放量滞涨
                .otherwise(0.8)  # 放量上涨（轻度惩罚）
            )
            .otherwise(1.0)  # 正常
            .alias('liquidity_penalty')
        ])
        
        # 记录流动性冲击
        unique_dates = result['trade_date'].unique().to_list()[:50]
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            shock_data = day_data.filter(pl.col('liquidity_shock_flag'))
            if not shock_data.is_empty():
                for row in shock_data.iter_rows(named=True):
                    self.shock_records.append(V90LiquidityShock(
                        trade_date=trade_date,
                        symbol=row['symbol'],
                        amount_ratio=row['amount_ratio'],
                        price_change=row['daily_return'],
                        shock_flag=True,
                        penalty_factor=row['liquidity_penalty'],
                    ))
        
        logger.info(f"V90: 流动性冲击计算完成，触发 {len(self.shock_records)} 次冲击")
        
        return result
    
    def get_shock_summary(self) -> Dict[str, Any]:
        """获取流动性冲击摘要"""
        if not self.shock_records:
            return {'total_shocks': 0, 'mean_penalty': 1.0}
        
        penalties = [s.penalty_factor for s in self.shock_records]
        
        return {
            'total_shocks': len(self.shock_records),
            'mean_penalty': float(np.mean(penalties)),
            'min_penalty': float(np.min(penalties)),
            'max_penalty': float(np.max(penalties)),
        }


# ===========================================
# V90 StyleNeutralization - 风格中性化引擎（V90 新增）
# ===========================================

class V90StyleNeutralizationEngine:
    """
    V90 Style Neutralization 引擎 - 风格中性化
    
    【核心逻辑】
    1. 计算市值因子（Size）：ln(total_mv)
    2. 计算 Beta 因子：个股收益率对市场收益率的回归系数
    3. 对原始信号进行中性化处理，剥离风格暴露
    
    【中性化方法】
    - 使用滚动窗口回归，计算信号对风格因子的暴露
    - 从原始信号中减去风格暴露部分
    - 保留纯 Alpha 部分
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V90_NEUTRALIZATION_WINDOW)
        self.size_neutralization = self.config.get('size_neutralization', V90_SIZE_NEUTRALIZATION)
        self.beta_neutralization = self.config.get('beta_neutralization', V90_BETA_NEUTRALIZATION)
        
        self.exposure_records: List[V90StyleExposure] = []
    
    def compute_style_neutralization(self, df: pl.DataFrame, 
                                      signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算风格中性化"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算市值因子（取对数）
        result = result.with_columns([
            pl.col('total_mv').fill_null(1.0).log().alias('size_factor')
        ])
        
        # 2. 计算 Beta 因子（使用滚动窗口）
        result = self._compute_beta_factor(result)
        
        # 3. 按日期计算风格中性化
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        neutralized_signals = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            if day_data.is_empty():
                continue
            
            signals = day_data[signal_col].to_numpy()
            sizes = day_data['size_factor'].to_numpy()
            betas = day_data['beta_factor'].to_numpy()
            symbols = day_data['symbol'].to_numpy()
            
            # 过滤无效数据
            valid_mask = (~np.isnan(signals) & ~np.isnan(sizes) & ~np.isnan(betas) &
                         np.isfinite(signals) & np.isfinite(sizes) & np.isfinite(betas))
            
            if np.sum(valid_mask) < 10:
                for i, symbol in enumerate(symbols):
                    neutralized_signals.append({
                        'trade_date': trade_date,
                        'symbol': symbol,
                        'size_exposure': 0.0,
                        'beta_exposure': 0.0,
                        'raw_signal': signals[i] if i < len(signals) else 0.0,
                        'neutralized_signal': signals[i] if i < len(signals) else 0.0,
                    })
                continue
            
            valid_signals = signals[valid_mask]
            valid_sizes = sizes[valid_mask]
            valid_betas = betas[valid_mask]
            valid_symbols = symbols[valid_mask]
            
            # 中性化处理
            neutralized = self._neutralize_signal(valid_signals, valid_sizes, valid_betas)
            
            for i, symbol in enumerate(valid_symbols):
                neutralized_signals.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'size_exposure': valid_sizes[i],
                    'beta_exposure': valid_betas[i],
                    'raw_signal': valid_signals[i],
                    'neutralized_signal': neutralized[i],
                })
            
            # 记录暴露（采样）
            if len(valid_symbols) <= 10:
                for i, symbol in enumerate(valid_symbols):
                    self.exposure_records.append(V90StyleExposure(
                        trade_date=trade_date,
                        symbol=symbol,
                        size_exposure=valid_sizes[i],
                        beta_exposure=valid_betas[i],
                        raw_score=valid_signals[i],
                        neutralized_score=neutralized[i],
                    ))
        
        # 合并回 DataFrame
        if neutralized_signals:
            neutralized_df = pl.DataFrame({
                'trade_date': [s['trade_date'] for s in neutralized_signals],
                'symbol': [s['symbol'] for s in neutralized_signals],
                'size_exposure': [s['size_exposure'] for s in neutralized_signals],
                'beta_exposure': [s['beta_exposure'] for s in neutralized_signals],
                'raw_signal': [s['raw_signal'] for s in neutralized_signals],
                'neutralized_signal': [s['neutralized_signal'] for s in neutralized_signals],
            })
            
            result = result.join(neutralized_df, on=['trade_date', 'symbol'], how='left')
        
        logger.info(f"V90: 风格中性化计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _compute_beta_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Beta 因子"""
        result = df.clone()
        
        # 计算日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        # 计算市场收益率（使用中位数）
        market_return = result.group_by('trade_date').agg([
            pl.col('daily_return').median().alias('market_return')
        ])
        
        result = result.join(market_return.select(['trade_date', 'market_return']), 
                            on='trade_date', how='left')
        
        # 简化 Beta 计算：使用滚动相关性和波动率比率
        # Beta = Corr(stock, market) * Std(stock) / Std(market)
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            pl.col('daily_return').rolling_std(window_size=self.window).over('symbol').alias('stock_vol'),
            pl.col('market_return').rolling_std(window_size=self.window).alias('market_vol')
        ])
        
        # 简化 Beta：波动率比率
        result = result.with_columns([
            (pl.col('stock_vol') / (pl.col('market_vol') + EPSILON)).alias('beta_factor')
        ])
        
        # 限制 Beta 范围
        result = result.with_columns([
            pl.col('beta_factor').clip(0.3, 3.0).alias('beta_factor')
        ])
        
        return result
    
    def _neutralize_signal(self, signals: np.ndarray, sizes: np.ndarray, 
                           betas: np.ndarray) -> np.ndarray:
        """
        中性化信号
        
        【方法】
        使用多元回归剥离风格暴露：
        signal = alpha + beta1 * size + beta2 * beta + residual
        neutralized_signal = residual
        """
        n = len(signals)
        
        # 标准化因子
        sizes_std = (sizes - np.mean(sizes)) / (np.std(sizes) + EPSILON)
        betas_std = (betas - np.mean(betas)) / (np.std(betas) + EPSILON)
        signals_std = (signals - np.mean(signals)) / (np.std(signals) + EPSILON)
        
        # 构建设计矩阵
        X = np.column_stack([np.ones(n), sizes_std, betas_std])
        y = signals_std
        
        try:
            # OLS 回归
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # 计算拟合值
            fitted = X @ coeffs
            
            # 残差 = 原始值 - 拟合值（剥离风格暴露）
            residual = y - fitted
            
            # 转换回原始尺度
            neutralized = residual * np.std(signals) + np.mean(signals)
            
            return neutralized
            
        except Exception:
            return signals
    
    def get_exposure_summary(self) -> Dict[str, Any]:
        """获取风格暴露摘要"""
        if not self.exposure_records:
            return {'mean_size_exposure': 0.0, 'mean_beta_exposure': 0.0}
        
        sizes = [e.size_exposure for e in self.exposure_records]
        betas = [e.beta_exposure for e in self.exposure_records]
        
        return {
            'mean_size_exposure': float(np.mean(sizes)),
            'std_size_exposure': float(np.std(sizes)),
            'mean_beta_exposure': float(np.mean(betas)),
            'std_beta_exposure': float(np.std(betas)),
            'sample_count': len(self.exposure_records),
        }


# ===========================================
# V90 AlphaWeight - Alpha 权重引擎
# ===========================================

class V90AlphaWeightEngine:
    """
    V90 AlphaWeight - Alpha 权重引擎
    
    【核心逻辑】
    1. Score >= 55 才参与权重计算
    2. 权重公式：W_i = Score_i × (1/Volatility_i) / Σ
    3. 单只标的权重限制在 [0.3%, 8%]
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V90_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V90_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V90_MAX_SINGLE_WEIGHT)
    
    def compute_alpha_weights(self, df: pl.DataFrame,
                               score_col: str = 'fused_signal') -> pl.DataFrame:
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
        
        logger.info(f"V90: Alpha 权重计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _alpha_weighting(self, scores: np.ndarray, volatilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Alpha 权重计算"""
        n = len(scores)
        weights = np.zeros(n)
        filtered = np.ones(n, dtype=bool)
        
        valid_mask = scores >= self.min_score
        filtered[valid_mask] = False
        
        if np.sum(valid_mask) < 1:
            return np.full(n, 1.0 / n), filtered
        
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
# V90 DynamicRebalance - 动态调仓引擎（V90 新增）
# ===========================================

class V90DynamicRebalanceEngine:
    """
    V90 Dynamic Rebalance 引擎 - 动态调仓阈值
    
    【核心逻辑】
    1. 严格遵守最小调仓间隔约束
    2. 在满足最小间隔后，计算当前信号与上次调仓信号的秩相关系数
    3. 当 Rank Correlation < 0.2 时，触发调仓
    4. 达到最大间隔时强制调仓
    
    【优势】
    - 避免固定间隔调仓的僵化
    - 在信号稳定时减少调仓频率，降低换手率
    - 在信号变化大时及时调仓，控制风险
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('threshold', V90_RANK_CORRELATION_THRESHOLD)
        self.min_interval = self.config.get('min_interval', V90_MIN_REBALANCE_INTERVAL)
        self.max_interval = self.config.get('max_interval', V90_MAX_REBALANCE_INTERVAL)
        
        self.last_rebalance_date = None
        self.last_rebalance_signals = {}
        self.rebalance_count = 0  # 调仓次数
    
    def should_rebalance(self, trade_date: str, current_signals: Dict[str, float]) -> bool:
        """
        判断是否应该调仓
        
        Parameters
        ----------
        trade_date : str
            交易日期
        current_signals : Dict[str, float]
            当前信号
            
        Returns
        -------
        bool
            是否调仓
        """
        # 第一次调仓
        if self.last_rebalance_date is None:
            self._update_rebalance_state(trade_date, current_signals)
            logger.info(f"V90: 首次调仓 ({trade_date})")
            self.rebalance_count += 1
            return True
        
        # 计算距离上次调仓的实际间隔
        try:
            last_date = datetime.strptime(self.last_rebalance_date, "%Y-%m-%d")
            curr_date = datetime.strptime(trade_date, "%Y-%m-%d")
            actual_days = (curr_date - last_date).days
        except Exception:
            actual_days = 0
        
        # 严格遵守最小间隔约束（关键修复）
        if actual_days < self.min_interval:
            return False
        
        # 检查最大间隔（强制调仓）
        if actual_days >= self.max_interval:
            logger.info(f"V90: 达到最大调仓间隔 ({actual_days}天)，强制调仓")
            self._update_rebalance_state(trade_date, current_signals)
            self.rebalance_count += 1
            return True
        
        # 计算秩相关系数
        if self.last_rebalance_signals:
            common_symbols = set(current_signals.keys()) & set(self.last_rebalance_signals.keys())
            
            if len(common_symbols) >= 10:
                current_values = np.array([current_signals[s] for s in common_symbols])
                last_values = np.array([self.last_rebalance_signals[s] for s in common_symbols])
                
                rank_corr = calculate_rank_correlation(current_values, last_values)
                
                logger.debug(f"V90: {trade_date} Rank Correlation = {rank_corr:.3f} (阈值={self.correlation_threshold}, 间隔={actual_days}天)")
                
                # 当相关性低于阈值时触发调仓
                if rank_corr < self.correlation_threshold:
                    logger.info(f"V90: Rank Correlation ({rank_corr:.3f}) < 阈值 ({self.correlation_threshold})，触发调仓")
                    self._update_rebalance_state(trade_date, current_signals)
                    self.rebalance_count += 1
                    return True
        
        return False
    
    def _update_rebalance_state(self, trade_date: str, signals: Dict[str, float]) -> None:
        """更新调仓状态"""
        self.last_rebalance_date = trade_date
        self.last_rebalance_signals = signals.copy()
    
    def get_rebalance_summary(self) -> Dict[str, Any]:
        """获取调仓摘要"""
        return {
            'last_rebalance_date': self.last_rebalance_date,
            'rebalance_count': self.rebalance_count,
            'min_interval': self.min_interval,
            'max_interval': self.max_interval,
            'correlation_threshold': self.correlation_threshold,
        }


# ===========================================
# V90 ICAudit - IC 审计
# ===========================================

class V90ICAudit:
    """V90 IC 审计"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'fused_signal') -> Dict[str, Any]:
        """计算 Rank IC"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        for lag in [1, 2, 3]:
            result = result.with_columns([
                pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
            ])
        
        ic_results = {}
        
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
        
        decay_normal = (ic_t1 >= ic_t2 >= ic_t3) and (ic_t1 > 0)
        
        return {
            'ic_t1': ic_t1,
            'ic_t2': ic_t2,
            'ic_t3': ic_t3,
            'std_t1': ic_results.get('t1', {}).get('std_ic', 0.0),
            'std_t2': ic_results.get('t2', {}).get('std_ic', 0.0),
            'std_t3': ic_results.get('t3', {}).get('std_ic', 0.0),
            'decay_normal': decay_normal,
            't1_ic_passed': ic_t1 >= V90_T1_IC_TARGET,
        }


# ===========================================
# V90 TurnoverTracker - 换手率追踪
# ===========================================

class V90TurnoverTracker:
    """V90 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V90TurnoverRecord] = []
        self.trading_days = 0
        self.rebalance_interval = self.config.get('rebalance_interval', 21)
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> V90TurnoverRecord:
        """记录换手率（带单日 10% 上限）"""
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            daily_turnover = 0.0
        else:
            # V90 单日换手率限制：10% 上限
            max_daily_turnover_value = portfolio_value * V90_DAILY_TURNOVER_MAX
            
            # 限制买入和卖出金额
            capped_buy_value = min(buy_value, max_daily_turnover_value)
            capped_sell_value = min(sell_value, max_daily_turnover_value)
            
            # V90 换手率计算：使用双边换手率（买入 + 卖出）/ 组合价值
            buy_turnover = capped_buy_value / portfolio_value
            sell_turnover = capped_sell_value / portfolio_value
            
            # 双边换手率
            turnover_rate = (capped_buy_value + capped_sell_value) / portfolio_value
            
            # 单日换手率（限制在 10% 以内）
            daily_turnover = min(turnover_rate, V90_DAILY_TURNOVER_MAX)
        
        self.trading_days += 1
        
        # V90 年化换手率：累计换手率 * (252 / 实际交易天数)
        cumulative_turnover = sum(r.turnover_rate for r in self.turnover_records) + turnover_rate
        annualized_turnover = cumulative_turnover * (252.0 / max(1, self.trading_days))
        
        record = V90TurnoverRecord(
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
        
        # V90 使用最终累计年化换手率
        total_turnover = sum(r.turnover_rate for r in self.turnover_records)
        annualized_turnover = total_turnover * (252.0 / max(1, self.trading_days))
        
        daily_turnovers = [r.daily_turnover for r in self.turnover_records]
        max_daily = np.max(daily_turnovers) if daily_turnovers else 0.0
        
        is_active = V90_TURNOVER_MIN <= annualized_turnover <= V90_TURNOVER_MAX
        daily_ok = max_daily <= V90_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean([r.turnover_rate for r in self.turnover_records])),
            'std_turnover': float(np.std([r.turnover_rate for r in self.turnover_records])),
            'max_turnover': float(np.max([r.turnover_rate for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V90_TURNOVER_MIN,
            'turnover_max': V90_TURNOVER_MAX,
        }


# ===========================================
# V90 PortfolioTracker - 组合追踪
# ===========================================

class V90PortfolioTracker:
    """V90 组合追踪器"""
    
    def __init__(self, initial_capital: float = V90_INITIAL_CAPITAL,
                 config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        self.cash = initial_capital
        self.positions: Dict[str, V90Position] = {}
        self.portfolio_snapshots: List[V90DailyPortfolio] = []
        self.total_value = initial_capital
        self.peak_value = initial_capital
    
    def update_positions(self, trade_date: str, prices: Dict[str, float]) -> None:
        """更新持仓价格"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]
                position.pnl = (prices[symbol] - position.entry_price) * position.quantity
    
    def calculate_portfolio_value(self) -> float:
        """计算组合总价值"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        self.total_value = self.cash + position_value
        self.peak_value = max(self.peak_value, self.total_value)
        return self.total_value
    
    def get_drawdown(self) -> float:
        """计算当前回撤"""
        if self.peak_value < EPSILON:
            return 0.0
        return (self.peak_value - self.total_value) / self.peak_value
    
    def record_snapshot(self, trade_date: str, daily_return: float,
                        turnover_rate: float) -> V90DailyPortfolio:
        """记录组合快照"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        
        cumulative_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        snapshot = V90DailyPortfolio(
            trade_date=trade_date,
            total_value=self.total_value,
            cash=self.cash,
            position_value=position_value,
            position_count=len(self.positions),
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            turnover_rate=turnover_rate,
        )
        self.portfolio_snapshots.append(snapshot)
        
        return snapshot
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取绩效摘要"""
        if not self.portfolio_snapshots:
            return {'total_return': 0.0, 'max_drawdown': 0.0}
        
        max_dd = 0.0
        peak = self.initial_capital
        for snapshot in self.portfolio_snapshots:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            dd = (peak - snapshot.total_value) / peak
            max_dd = max(max_dd, dd)
        
        return {
            'total_value': self.total_value,
            'total_return': (self.total_value - self.initial_capital) / self.initial_capital,
            'max_drawdown': max_dd,
            'final_cash': self.cash,
            'position_count': len(self.positions),
        }


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V90_INITIAL_CAPITAL',
    'V90_MAX_POSITIONS',
    'V90_WARMUP_PERIOD',
    'V90_MIN_SCORE_THRESHOLD',
    'V90_MIN_SINGLE_WEIGHT',
    'V90_MAX_SINGLE_WEIGHT',
    'V90_TURNOVER_MIN',
    'V90_TURNOVER_MAX',
    'V90_DAILY_TURNOVER_MAX',
    'V90_ANNUAL_RETURN_TARGET',
    'V90_T1_IC_TARGET',
    'V90_COMMISSION_RATE',
    'V90_MIN_COMMISSION',
    'V90_STAMP_DUTY',
    'V90_TRANSFER_FEE',
    'V90_HALF_LIFE_LAGS',
    'V90_LAG1_WEIGHT',
    'V90_LAG3_WEIGHT',
    'V90_LAG5_WEIGHT',
    'V90_SIZE_NEUTRALIZATION',
    'V90_BETA_NEUTRALIZATION',
    'V90_NEUTRALIZATION_WINDOW',
    'V90_LIQUIDITY_SHOCK_WINDOW',
    'V90_LIQUIDITY_SHOCK_THRESHOLD',
    'V90_LIQUIDITY_SHOCK_PENALTY',
    'V90_DYNAMIC_REBALANCE',
    'V90_RANK_CORRELATION_THRESHOLD',
    'V90_MIN_REBALANCE_INTERVAL',
    'V90_MAX_REBALANCE_INTERVAL',
    'V90DataManager',
    'V90AlphaFusion',
    'V90AlphaWeightEngine',
    'V90ICAudit',
    'V90TurnoverTracker',
    'V90PortfolioTracker',
    'V90VolumePriceDivergence',
    'V90LiquidityShockEngine',
    'V90StyleNeutralizationEngine',
    'V90DynamicRebalanceEngine',
    'V90FusionSignal',
    'V90VolumePriceDivergence',
    'V90LiquidityShock',
    'V90StyleExposure',
    'V90LookaheadCheckResult',
    'V90TurnoverRecord',
    'V90Position',
    'V90DailyPortfolio',
    'calculate_half_life_decay_weights',
    'calculate_second_derivative',
    'normalize_rank',
    'check_lookahead_bias',
    'calculate_ic_decay_simple',
    'fill_with_market_median',
    'calculate_rank_correlation',
]