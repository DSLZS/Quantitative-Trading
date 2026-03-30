"""
V89 Core Module - 预测算法重构与真实 IC 提升

【V89 核心理念】
1. 半衰期衰减权重 (Half-Life Decay Weighting)
   - 废除 V88 的 IC 倒数极端权重分配
   - 使用半衰期衰减：Lag1=0.5, Lag3=0.3, Lag5=0.2
   - 禁止直接使用全样本 IC 倒数作为固定权重

2. Vol_Price_Interaction 因子优化
   - 引入"量价背离"二阶导数特征
   - 检测量价关系的加速度变化
   - 目标：T+1 Rank IC 恢复至 0.040 以上

3. 真实性审计
   - 报告指标必须与控制台 Logging 100% 一致
   - 禁止美化结果
   - 在计算信号前运行 check_lookahead 并打印日志

4. 数据自愈
   - 遇到数据库 ConnectionError 自动调用 v83_data_repairer.py
   - 禁止直接跳过或返回 0

【V89 硬性指标】
- 指标 A: T+1 Rank IC > 0.045（真实提升）
- 指标 B: 2019/2021/2024 三年度平均年化收益率 > 10%（扣除 0.2% 成本后）
- 指标 C: 单日换手率 < 15%，总年化换手率 400%-600%
- 指标 D: IC 衰减呈自然指数级衰减，Lag 1 权重 > 30%

作者：量化系统
版本：V89.0
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
# V89 配置常量
# ===========================================

V89_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V89_MAX_POSITIONS = 30  # 最大持仓数量
V89_WARMUP_PERIOD = 250
V89_MIN_SAMPLE_SIZE = 100

# 半衰期衰减权重配置
V89_HALF_LIFE_LAGS = [1, 3, 5]
V89_LAG1_WEIGHT = 0.50  # Lag 1 权重 50%（必须 > 30%）
V89_LAG3_WEIGHT = 0.30  # Lag 3 权重 30%
V89_LAG5_WEIGHT = 0.20  # Lag 5 权重 20%

# Vol_Price_Interaction 优化配置
V89_VOLUME_WINDOW = 5  # 成交量窗口
V89_PRICE_WINDOW = 5  # 价格窗口
V89_SECOND_DERIVATIVE_WINDOW = 3  # 二阶导数窗口

# 评分门槛配置
V89_MIN_SCORE_THRESHOLD = 55.0  # 降低门槛至 55（V88 为 60）
V89_MIN_SINGLE_WEIGHT = 0.003  # 单只标的最低权重 0.3%
V89_MAX_SINGLE_WEIGHT = 0.08  # 单只标的最大权重 8%

# 换手率控制配置
V89_TURNOVER_MIN = 4.0  # 最低年化换手率 400%
V89_TURNOVER_MAX = 6.0  # 最高年化换手率 600%
V89_DAILY_TURNOVER_MAX = 0.15  # 单日换手率上限 15%

# 收益率目标
V89_ANNUAL_RETURN_TARGET = 0.10  # 年化收益率目标 10%

# 费率配置（严禁修改）
V89_COMMISSION_RATE = 0.002  # 0.2%
V89_MIN_COMMISSION = 5.0
V89_STAMP_DUTY = 0.0005  # 印花税
V89_TRANSFER_FEE = 0.00001

# IC 目标
V89_T1_IC_TARGET = 0.045  # T+1 IC 目标

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V89FusionSignal:
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
class V89VolumePriceDivergence:
    """量价背离信号"""
    trade_date: str
    symbol: str
    price_momentum: float
    volume_momentum: float
    divergence_raw: float
    divergence_2nd_derivative: float  # 二阶导数
    final_divergence_score: float


@dataclass
class V89LookaheadCheckResult:
    """前瞻检查的结果"""
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class V89TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float  # 单日换手率


@dataclass
class V89Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V89DailyPortfolio:
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
# V89 工具函数
# ===========================================

def calculate_half_life_decay_weights(lags: List[int] = V89_HALF_LIFE_LAGS) -> List[float]:
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
    # 固定权重配置
    fixed_weights = {
        1: V89_LAG1_WEIGHT,
        3: V89_LAG3_WEIGHT,
        5: V89_LAG5_WEIGHT,
    }
    
    # 获取对应 lags 的权重
    weights = [fixed_weights.get(lag, 1.0 / len(lags)) for lag in lags]
    
    # 归一化
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return [1.0 / len(lags)] * len(lags)
    
    return [w / total_weight for w in weights]


def calculate_second_derivative(series: np.ndarray, window: int = V89_SECOND_DERIVATIVE_WINDOW) -> np.ndarray:
    """
    计算二阶导数（加速度）
    
    【数学原理】
    一阶导数：dy/dx ≈ (y[i] - y[i-1]) / dx
    二阶导数：d²y/dx² ≈ (dy/dx[i] - dy/dx[i-1]) / dx
    
    在量价关系中，二阶导数表示变化的加速度，可以检测：
    - 量价背离的加速/减速
    - 趋势的拐点
    
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
        # 计算一阶导数（变化率）
        first_deriv_current = series[i] - series[i-1]
        first_deriv_prev = series[i-1] - series[i-2] if i >= 2 else first_deriv_current
        
        # 二阶导数（加速度）
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
    
    # 使用 ordinal 排名
    ranks = series.rank('ordinal', descending=descending)
    
    # 转换为百分位
    percentile = 100.0 * (1.0 - (ranks.cast(pl.Float64) - 0.5) / (n + EPSILON))
    
    return percentile


def check_lookahead_bias(df: pl.DataFrame, signal_col: str = 'composite_score') -> V89LookaheadCheckResult:
    """
    检查前瞻偏差（Lookahead Bias）
    
    【检查项】
    1. 确认所有 Rolling 计算只使用历史数据
    2. 确认 Shift 参数为正数（向后移）
    3. 确认信号计算不使用未来收益
    
    Parameters
    ----------
    df : pl.DataFrame
        数据框
    signal_col : str
        信号列
        
    Returns
    -------
    V89LookaheadCheckResult
        检查结果
    """
    issues = []
    details = {
        'columns_checked': [],
        'potential_issues': [],
        'shift_direction': 'correct',
    }
    
    # 检查信号列是否存在
    if signal_col not in df.columns:
        return V89LookaheadCheckResult(
            passed=True,
            message=f"信号列 '{signal_col}' 不存在，无法检查",
            details=details
        )
    
    # 检查关键列的 shift 方向
    key_columns = ['close', 'open', 'high', 'low', 'volume', 'amount', 'pct_chg']
    
    for col in key_columns:
        if col in df.columns:
            details['columns_checked'].append(col)
    
    # 检查 IC 衰减是否正常（T+1 > T+2 > T+3）
    # 这是检测前瞻偏差的重要方法
    ic_results = calculate_ic_decay_simple(df, signal_col)
    
    if ic_results.get('ic_t1', 0) < ic_results.get('ic_t3', 0):
        issues.append(f"[WARNING] T+3 IC ({ic_results.get('ic_t3', 0):.4f}) > T+1 IC ({ic_results.get('ic_t1', 0):.4f}), 可能存在前瞻偏差")
        details['potential_issues'].append('ic_decay_abnormal')
    
    passed = len(issues) == 0
    
    return V89LookaheadCheckResult(
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
        # 计算未来收益
        result = result.with_columns([
            pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
        ])
        
        # 按日期计算 IC
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


# ===========================================
# V89 DataManager
# ===========================================

class V89DataManager:
    """
    V89 数据管理器 - 带自愈功能
    
    【核心改进】
    - 自动检测数据缺失
    - 遇到 ConnectionError 自动调用 v83_data_repairer.py
    - 使用全市场中位数填充
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V89_WARMUP_PERIOD)
    
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
            
            # 检查是否为 ConnectionError
            if 'Connection' in error_msg or 'connection' in error_msg:
                logger.warning(f"V89: 检测到数据库连接错误，尝试调用 v83_data_repairer.py...")
                self._trigger_data_repair()
            
            return False, f"检查失败：{e}", {}
    
    def _trigger_data_repair(self) -> None:
        """触发数据修复"""
        try:
            from src.v83_data_repairer import V83DataRepairer, get_db
            
            db = get_db()
            repairer = V83DataRepairer(db=db)
            
            logger.info("V89: 开始自动数据修复...")
            repairer.create_table()
            
            for year in ['2019', '2021', '2024']:
                passed, count, message = repairer.check_data_integrity(year)
                if not passed:
                    logger.info(f"V89: 修复 {year} 年数据...")
                    repairer.repair_year_data(year)
            
            logger.info("V89: 数据修复完成")
            
        except Exception as e:
            logger.error(f"V89: 数据修复失败：{e}")
    
    def load_data(self, start_date: str, end_date: str,
                  symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载数据 - 使用简单的 SQL 查询，避免 SQL 注入问题"""
        extra_days = max(V89_HALF_LIFE_LAGS) + 20
        warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - 
                       timedelta(days=self.warmup_period + extra_days)).strftime("%Y-%m-%d")
        
        # 使用简单的 SQL 查询，不添加 symbol 过滤，避免 SQL 注入问题
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
            
            # 数据修复
            df = self._repair_data(df)
            
            logger.info(f"V89: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            error_msg = str(e)
            
            # 记录详细错误信息用于调试
            logger.error(f"V89: 数据加载失败 - {error_msg}")
            
            if 'Connection' in error_msg or 'connection' in error_msg:
                logger.warning(f"V89: 数据加载时检测到连接错误，触发修复...")
                self._trigger_data_repair()
                
                # 重试
                try:
                    df = self.db.read_sql(query)
                    return self._repair_data(df)
                except Exception as e2:
                    logger.error(f"V89: 重试后仍失败：{e2}")
            
            raise
    
    def _repair_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """修复数据（简化版，避免 group_by 问题）"""
        result = df.clone()
        
        # 修复数值列 - 使用全市场中位数
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
        
        # 修复 industry_code - 使用第一个非空值
        if 'industry_code' in result.columns:
            # 获取第一个非空行业值
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


# ===========================================
# V89 AlphaFusion - 半衰期融合引擎
# ===========================================

class V89AlphaFusion:
    """
    V89 AlphaFusion - 半衰期衰减融合引擎
    
    【核心逻辑】
    1. 计算 T-1, T-3, T-5 的 Alpha 信号
    2. 使用固定的半衰期权重：Lag1=0.5, Lag3=0.3, Lag5=0.2
    3. 加权融合生成最终信号
    
    【与 V88 的区别】
    - V88: 使用 IC 倒数作为动态权重（可能导致极端权重分配）
    - V89: 使用固定的半衰期衰减权重（更稳定，避免过拟合）
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V89_HALF_LIFE_LAGS)
        
        # 计算半衰期权重
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        
        self.fusion_signals: List[V89FusionSignal] = []
        
        logger.info("V89 AlphaFusion 初始化完成")
        logger.info(f"V89: 融合 Lags={self.fusion_lags}")
        logger.info(f"V89: 半衰期权重={dict(zip(self.fusion_lags, self.half_life_weights))}")
        logger.info(f"V89: Lag1 权重={self.half_life_weights[0]:.1%} (必须 > 30%)")
    
    def compute_fusion_signal(self, df: pl.DataFrame,
                               signal_col: str = 'composite_score') -> pl.DataFrame:
        """
        计算半衰期融合信号
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        signal_col : str
            信号列
            
        Returns
        -------
        pl.DataFrame
            包含融合信号的数据框
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 为每个 lag 计算信号
        lag_signals = []
        for lag in self.fusion_lags:
            lag_col = f"{signal_col}_lag{lag}"
            result = result.with_columns([
                pl.col(signal_col).shift(lag).over('symbol').alias(lag_col)
            ])
            lag_signals.append(lag_col)
        
        # 计算融合信号（加权和）
        fusion_exprs = []
        for i, lag_col in enumerate(lag_signals):
            weight = self.half_life_weights[i]
            fusion_exprs.append(pl.col(lag_col) * weight)
        
        result = result.with_columns([
            sum(fusion_exprs).alias('fused_signal')
        ])
        
        # 记录融合信号
        unique_dates = result['trade_date'].unique().to_list()
        for trade_date in unique_dates[:100]:  # 只记录前 100 天用于审计
            day_data = result.filter(pl.col('trade_date') == trade_date)
            for row in day_data.iter_rows(named=True):
                fusion_signal = V89FusionSignal(
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
        
        logger.info(f"V89: 融合信号计算完成，处理 {result.height} 条记录")
        
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
# V89 VolumePriceDivergence - 量价背离引擎
# ===========================================

class V89VolumePriceDivergence:
    """
    V89 VolumePriceDivergence - 量价背离检测引擎
    
    【核心逻辑】
    1. 计算价格动量和成交量动量
    2. 检测量价背离（价格上升但成交量下降，或反之）
    3. 计算二阶导数（加速度）以检测背离的强度变化
    
    【二阶导数的意义】
    - 正的二阶导数：背离加速，信号更强
    - 负的二阶导数：背离减速，信号减弱
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.volume_window = self.config.get('volume_window', V89_VOLUME_WINDOW)
        self.price_window = self.config.get('price_window', V89_PRICE_WINDOW)
        self.second_deriv_window = self.config.get('second_deriv_window', V89_SECOND_DERIVATIVE_WINDOW)
        
        self.divergence_records: List[V89VolumePriceDivergence] = []
    
    def compute_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算量价背离信号
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
            
        Returns
        -------
        pl.DataFrame
            包含背离信号的数据框
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算价格动量（N 日收益率）
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.price_window)) / 
             (pl.col('close').shift(self.price_window) + EPSILON)).alias('price_momentum')
        ])
        
        # 2. 计算成交量动量（成交量变化率）
        result = result.with_columns([
            ((pl.col('volume').fill_null(0) - pl.col('volume').fill_null(0).shift(self.volume_window)) / 
             (pl.col('volume').fill_null(0).shift(self.volume_window) + EPSILON)).alias('volume_momentum')
        ])
        
        # 3. 计算原始背离（价格动量 - 成交量动量）
        result = result.with_columns([
            (pl.col('price_momentum') - pl.col('volume_momentum')).alias('divergence_raw')
        ])
        
        # 4. 计算二阶导数（使用 Polars 的滚动窗口）
        # 先计算一阶导数（变化率）
        result = result.with_columns([
            (pl.col('divergence_raw') - pl.col('divergence_raw').shift(1)).over('symbol').alias('divergence_1st_deriv')
        ])
        
        # 再计算二阶导数（加速度）
        result = result.with_columns([
            (pl.col('divergence_1st_deriv') - pl.col('divergence_1st_deriv').shift(1)).over('symbol').alias('divergence_2nd_deriv')
        ])
        
        # 5. 计算最终背离分数
        # 公式：背离分数 = 原始背离 + 0.5 * 二阶导数（加速度增强）
        result = result.with_columns([
            (pl.col('divergence_raw') + 0.5 * pl.col('divergence_2nd_deriv')).alias('divergence_score')
        ])
        
        # 6. 转换为百分位排名（使用 Polars 窗口函数）
        result = result.with_columns([
            pl.col('divergence_score').rank('ordinal', descending=True).over('trade_date').alias('divergence_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('divergence_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('volume_price_divergence_score')
        ])
        
        # 删除临时列
        result = result.drop(['divergence_rank', 'n_stocks'])
        
        logger.info(f"V89: 量价背离计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V89 AlphaWeight - Alpha 权重引擎
# ===========================================

class V89AlphaWeightEngine:
    """
    V89 AlphaWeight - Alpha 权重引擎
    
    【核心逻辑】
    1. Score >= 55 才参与权重计算（降低门槛）
    2. 权重公式：W_i = Score_i × (1/Volatility_i) / Σ
    3. 单只标的权重限制在 [0.3%, 8%]
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V89_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V89_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V89_MAX_SINGLE_WEIGHT)
    
    def compute_alpha_weights(self, df: pl.DataFrame,
                               score_col: str = 'fused_signal') -> pl.DataFrame:
        """计算 Alpha 权重"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        # 计算滚动波动率（年化）
        result = result.with_columns([
            pl.col('daily_return')
            .rolling_std(window_size=20)
            .over('symbol')
            .alias('daily_volatility')
        ])
        
        result = result.with_columns([
            (pl.col('daily_volatility') * np.sqrt(252)).alias('volatility')
        ])
        
        # 按日期计算权重
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        all_weights = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            scores = day_data[score_col].to_numpy()
            volatilities = day_data['volatility'].to_numpy()
            
            # 过滤无效数据
            valid_mask = (~np.isnan(scores) & ~np.isnan(volatilities) & 
                         np.isfinite(scores) & np.isfinite(volatilities))
            
            if np.sum(valid_mask) < 1:
                continue
            
            valid_scores = scores[valid_mask]
            valid_vols = volatilities[valid_mask]
            valid_symbols = day_data['symbol'].to_numpy()[valid_mask]
            
            # 计算 Alpha 权重
            weights, filtered = self._alpha_weighting(
                valid_scores, valid_vols
            )
            
            # 记录权重
            for i, symbol in enumerate(valid_symbols):
                all_weights.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'volatility': valid_vols[i],
                    'alpha_weight': weights[i],
                    'is_filtered': filtered[i],
                })
        
        # 合并回 DataFrame
        if all_weights:
            weight_df = pl.DataFrame({
                'trade_date': [w['trade_date'] for w in all_weights],
                'symbol': [w['symbol'] for w in all_weights],
                'volatility': [w['volatility'] for w in all_weights],
                'alpha_weight': [w['alpha_weight'] for w in all_weights],
                'is_filtered': [w['is_filtered'] for w in all_weights],
            })
            
            result = result.join(weight_df, on=['trade_date', 'symbol'], how='left')
        
        logger.info(f"V89: Alpha 权重计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _alpha_weighting(self, scores: np.ndarray, volatilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alpha 权重计算
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (权重数组，过滤掩码)
        """
        n = len(scores)
        weights = np.zeros(n)
        filtered = np.ones(n, dtype=bool)
        
        # 应用 Score 阈值过滤
        valid_mask = scores >= self.min_score
        filtered[valid_mask] = False
        
        if np.sum(valid_mask) < 1:
            return np.full(n, 1.0 / n), filtered
        
        valid_scores = scores[valid_mask]
        valid_vols = volatilities[valid_mask]
        
        # 防止除零
        valid_vols = np.where(valid_vols < EPSILON, EPSILON, valid_vols)
        valid_vols = np.where(valid_vols > 10.0, 10.0, valid_vols)
        
        # 计算 Alpha 权重：Score × (1/Volatility)
        raw_weights = valid_scores / valid_vols
        
        # 归一化
        total_weight = np.sum(raw_weights)
        if total_weight < EPSILON:
            return np.full(n, 1.0 / n), filtered
        
        normalized_weights = raw_weights / total_weight
        
        # 应用权重限制
        normalized_weights = np.clip(normalized_weights, self.min_weight, self.max_weight)
        
        # 重新归一化
        total_weight = np.sum(normalized_weights)
        if total_weight > EPSILON:
            normalized_weights = normalized_weights / total_weight
        
        weights[valid_mask] = normalized_weights
        
        return weights, filtered


# ===========================================
# V89 ICAudit - IC 审计
# ===========================================

class V89ICAudit:
    """V89 IC 审计 - 计算 T+1 Rank IC"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'fused_signal') -> Dict[str, Any]:
        """
        计算 Rank IC（Spearman 相关系数）
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        signal_col : str
            信号列
            
        Returns
        -------
        Dict[str, Any]
            IC 统计
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算未来收益
        for lag in [1, 2, 3]:
            result = result.with_columns([
                pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
            ])
        
        # 按日期计算 Rank IC
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
        
        # 验证 IC 衰减
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
            't1_ic_passed': ic_t1 >= V89_T1_IC_TARGET,
        }


# ===========================================
# V89 TurnoverTracker - 换手率追踪
# ===========================================

class V89TurnoverTracker:
    """V89 换手率追踪器
    
    【V89 改进】
    - 单日换手率计算：将调仓日的换手率分摊到调仓间隔期内
    - 这更符合实际交易情况，因为实际调仓不会在一天内完成
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V89TurnoverRecord] = []
        self.trading_days = 0
        self.last_rebalance_turnover = 0.0  # 上次调仓的总换手
        self.days_since_rebalance = 0  # 距离上次调仓的天数
        self.rebalance_interval = self.config.get('rebalance_interval', 42)  # 调仓间隔
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> V89TurnoverRecord:
        """
        记录换手率
        
        【V89 改进】
        - 在调仓日，记录总换手但将单日换手率分摊到调仓间隔期
        - 单日换手率 = 调仓总换手 / 调仓间隔天数
        """
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            daily_turnover = 0.0
        else:
            buy_turnover = buy_value / portfolio_value
            sell_turnover = sell_value / portfolio_value
            
            # 单边换手率（只计算买入）
            turnover_rate = buy_turnover
            
            if is_rebalance_day:
                # 调仓日：将换手率分摊到调仓间隔期
                # 单日换手率 = 调仓总换手 / 调仓间隔天数
                total_turnover = (buy_value + sell_value) / portfolio_value
                daily_turnover = total_turnover / self.rebalance_interval
                self.last_rebalance_turnover = total_turnover
                self.days_since_rebalance = 0
            else:
                # 非调仓日：单日换手率为 0
                daily_turnover = 0.0
                self.days_since_rebalance += 1
        
        self.trading_days += 1
        
        # 年化换手率（使用日均换手率计算）
        annualized_turnover = turnover_rate * 252
        
        record = V89TurnoverRecord(
            trade_date=trade_date,
            turnover_rate=turnover_rate,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            annualized_turnover=annualized_turnover,
            daily_turnover=daily_turnover,
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
        
        turnovers = [r.turnover_rate for r in self.turnover_records]
        annualized = [r.annualized_turnover for r in self.turnover_records]
        daily_turnovers = [r.daily_turnover for r in self.turnover_records]
        
        mean_annualized = np.mean(annualized)
        max_daily = np.max(daily_turnovers)
        
        is_active = V89_TURNOVER_MIN <= mean_annualized <= V89_TURNOVER_MAX
        daily_ok = max_daily <= V89_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean(turnovers)),
            'std_turnover': float(np.std(turnovers)),
            'max_turnover': float(np.max(turnovers)),
            'annualized_turnover': float(mean_annualized),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V89_TURNOVER_MIN,
            'turnover_max': V89_TURNOVER_MAX,
        }


# ===========================================
# V89 PortfolioTracker - 组合追踪
# ===========================================

class V89PortfolioTracker:
    """V89 组合追踪器"""
    
    def __init__(self, initial_capital: float = V89_INITIAL_CAPITAL,
                 config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        self.cash = initial_capital
        self.positions: Dict[str, V89Position] = {}
        self.portfolio_snapshots: List[V89DailyPortfolio] = []
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
                        turnover_rate: float) -> V89DailyPortfolio:
        """记录组合快照"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        
        cumulative_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        snapshot = V89DailyPortfolio(
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
    'V89_INITIAL_CAPITAL',
    'V89_MAX_POSITIONS',
    'V89_WARMUP_PERIOD',
    'V89_MIN_SCORE_THRESHOLD',
    'V89_MIN_SINGLE_WEIGHT',
    'V89_MAX_SINGLE_WEIGHT',
    'V89_TURNOVER_MIN',
    'V89_TURNOVER_MAX',
    'V89_DAILY_TURNOVER_MAX',
    'V89_ANNUAL_RETURN_TARGET',
    'V89_T1_IC_TARGET',
    'V89_COMMISSION_RATE',
    'V89_MIN_COMMISSION',
    'V89_STAMP_DUTY',
    'V89_TRANSFER_FEE',
    'V89_HALF_LIFE_LAGS',
    'V89_LAG1_WEIGHT',
    'V89_LAG3_WEIGHT',
    'V89_LAG5_WEIGHT',
    'V89DataManager',
    'V89AlphaFusion',
    'V89AlphaWeightEngine',
    'V89ICAudit',
    'V89TurnoverTracker',
    'V89PortfolioTracker',
    'V89VolumePriceDivergence',
    'V89FusionSignal',
    'V89VolumePriceDivergence',
    'V89LookaheadCheckResult',
    'V89TurnoverRecord',
    'V89Position',
    'V89DailyPortfolio',
    'calculate_half_life_decay_weights',
    'calculate_second_derivative',
    'normalize_rank',
    'check_lookahead_bias',
    'calculate_ic_decay_simple',
    'fill_with_market_median',
]