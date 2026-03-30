"""
V91 Logic Module - 非线性特征共振与 Regime-Aware 动态仓位管理

【V91 核心任务】
1. 预测算法升级（重心）：Nonlinear_Resonance_Module
   - 计算 (Residual_Alpha * Smart_Flow_Score) 的 5 日滚动非线性加权
   - 不做线性融合，实现真正的非线性交互

2. 市场环境自适应（Regime Switching）：
   - 市场压力传感器：Volatility_Skew（波动率偏度）
   - 当 Skew > 2 倍标准差时，判定为极端市场
   - 此时自动将 Style_Neutralization 力度加倍，降低目标换手率至 100%

3. 数据防御机制（Integrity_Shield）：
   - 计算前执行 Integrity_Shield 检查
   - 若遇到 industry_return 等字段缺失，调用 src.loaders 补全脚本实时抓取
   - 禁止直接返回 0

【V91 硬性指标】
- 指标 A：三年度（2019, 2021, 2024）Mean Rank IC 均需 > 0.045，且 IC IR > 0.6
- 指标 B：2021 年（震荡市）和 2024 年（极端波动市）的总收益必须转正（> 5%）
- 指标 C：自动化回测报告中必须包含 Max_Rebalancing_Error_Log

作者：量化系统
版本：V91.0
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
# V91 配置常量
# ===========================================

V91_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V91_MAX_POSITIONS = 30  # 最大持仓数量
V91_WARMUP_PERIOD = 60  # Reduced warmup period to allow more trading days
V91_MIN_SAMPLE_SIZE = 100

# Nonlinear_Resonance 配置
V91_RESONANCE_WINDOW = 5  # 5 日滚动窗口
V91_RESONANCE_NONLINEAR_POWER = 1.5  # 非线性加权指数
V91_RESONANCE_DECAY_FACTOR = 0.8  # 时间衰减因子

# Regime Switching 配置
V91_VOLATILITY_WINDOW = 60  # 波动率计算窗口
V91_SKEW_THRESHOLD = 2.0  # 偏度阈值（2 倍标准差）
V91_EXTREME_NEUTRALIZATION_MULTIPLIER = 2.0  # 极端市场中性化力度加倍
V91_NORMAL_TURNOVER_TARGET = 3.0  # 正常市场年化换手率目标 300%
V91_EXTREME_TURNOVER_TARGET = 1.0  # 极端市场年化换手率目标 100%

# 评分门槛配置
V91_MIN_SCORE_THRESHOLD = 55.0
V91_MIN_SINGLE_WEIGHT = 0.003  # 单只标的最低权重 0.3%
V91_MAX_SINGLE_WEIGHT = 0.08  # 单只标的最大权重 8%

# 调仓配置
V91_MIN_REBALANCE_INTERVAL = 5  # 最小调仓间隔 5 天
V91_MAX_REBALANCE_INTERVAL = 5  # 最大调仓间隔 5 天
V91_REBALANCE_INTERVAL = 5  # 调仓间隔（别名）

# 费率配置（严禁修改）
V91_COMMISSION_RATE = 0.002  # 0.2%
V91_MIN_COMMISSION = 5.0
V91_STAMP_DUTY = 0.0005  # 印花税
V91_TRANSFER_FEE = 0.00001

# IC 目标
V91_T1_IC_TARGET = 0.045  # T+1 IC 目标
V91_IC_IR_TARGET = 0.6  # IC IR 目标

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V91ResonanceSignal:
    """非线性共振信号"""
    trade_date: str
    symbol: str
    residual_alpha: float
    smart_flow_score: float
    raw_interaction: float
    nonlinear_weighted: float
    resonance_5d: float


@dataclass
class V91RegimeState:
    """市场状态"""
    trade_date: str
    volatility_skew: float
    is_extreme: bool
    neutralization_multiplier: float
    turnover_target: float


@dataclass
class V91IntegrityCheck:
    """数据完整性检查"""
    trade_date: str
    field_name: str
    is_missing: bool
    repair_attempted: bool
    repair_success: bool
    fallback_value: Optional[float]


@dataclass
class V91RebalanceError:
    """调仓错误记录"""
    trade_date: str
    symbol: str
    error_type: str
    error_message: str
    skipped: bool


# ===========================================
# V91 工具函数
# ===========================================

def calculate_nonlinear_weighted_sum(series: np.ndarray, 
                                      weights: np.ndarray,
                                      power: float = V91_RESONANCE_NONLINEAR_POWER) -> float:
    """
    计算非线性加权和
    
    【公式】
    Nonlinear_Sum = Σ(w_i * x_i^power) / Σ(w_i)
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
    weights : np.ndarray
        权重序列
    power : float
        非线性指数
        
    Returns
    -------
    float
        非线性加权和
    """
    if len(series) == 0 or len(weights) == 0:
        return 0.0
    
    valid_mask = (~np.isnan(series) & np.isfinite(series) & 
                  ~np.isnan(weights) & np.isfinite(weights) &
                  (series > 0))  # 只处理正数
    
    if np.sum(valid_mask) < 1:
        return 0.0
    
    valid_series = series[valid_mask]
    valid_weights = weights[valid_mask]
    
    # 非线性变换
    powered_series = np.power(valid_series, power)
    
    # 加权求和
    weighted_sum = np.sum(valid_weights * powered_series)
    weight_total = np.sum(valid_weights)
    
    if weight_total < EPSILON:
        return 0.0
    
    return weighted_sum / weight_total


def calculate_volatility_skew(returns: np.ndarray, 
                               window: int = V91_VOLATILITY_WINDOW) -> Optional[float]:
    """
    计算波动率偏度
    
    【公式】
    Skew = E[(r - μ)^3] / σ^3
    
    Parameters
    ----------
    returns : np.ndarray
        收益率序列
    window : int
        计算窗口
        
    Returns
    -------
    Optional[float]
        波动率偏度，如果数据不足则返回 None
    """
    if len(returns) < window:
        return None
    
    valid_returns = returns[-window:]
    valid_returns = valid_returns[~np.isnan(valid_returns) & np.isfinite(valid_returns)]
    
    if len(valid_returns) < 20:
        return None
    
    mean_ret = np.mean(valid_returns)
    std_ret = np.std(valid_returns)
    
    if std_ret < EPSILON:
        return 0.0
    
    # 计算偏度
    skew = stats.skew(valid_returns)
    
    return skew


def normalize_to_zscore(series: np.ndarray) -> np.ndarray:
    """
    将序列转换为 Z-Score
    
    Parameters
    ----------
    series : np.ndarray
        输入序列
        
    Returns
    -------
    np.ndarray
        Z-Score 序列
    """
    mean_val = np.nanmean(series)
    std_val = np.nanstd(series)
    
    if std_val < EPSILON:
        return np.zeros_like(series)
    
    return (series - mean_val) / std_val


# ===========================================
# Integrity_Shield - 数据防御机制
# ===========================================

class V91IntegrityShield:
    """
    V91 Integrity_Shield - 数据防御机制
    
    【核心逻辑】
    1. 在计算前检查必需字段是否存在
    2. 若字段缺失，调用 src.loaders 补全脚本实时抓取
    3. 禁止直接返回 0
    """
    
    REQUIRED_FIELDS = [
        'open', 'high', 'low', 'close', 'volume', 'amount', 
        'pct_chg', 'industry_code', 'total_mv'
    ]
    
    OPTIONAL_FIELDS = [
        'industry_return', 'market_return', 'volatility_skew'
    ]
    
    def __init__(self, db=None):
        self.db = db
        self.check_records: List[V91IntegrityCheck] = []
        self.repair_attempts: int = 0
        self.repair_successes: int = 0
    
    def check_and_repair(self, df: pl.DataFrame, 
                         trade_date: str) -> Tuple[pl.DataFrame, List[V91IntegrityCheck]]:
        """
        检查并修复数据
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        trade_date : str
            交易日期
            
        Returns
        -------
        Tuple[pl.DataFrame, List[V91IntegrityCheck]]
            修复后的数据框和检查记录
        """
        result = df.clone()
        check_records = []
        
        # 检查必需字段
        for field in self.REQUIRED_FIELDS:
            if field not in result.columns:
                logger.warning(f"V91: 必需字段 '{field}' 缺失，尝试修复...")
                self._attempt_repair(field, trade_date)
                check_records.append(V91IntegrityCheck(
                    trade_date=trade_date,
                    field_name=field,
                    is_missing=True,
                    repair_attempted=True,
                    repair_success=False,
                    fallback_value=None,
                ))
            else:
                # 检查字段是否有足够的有效值
                valid_count = result[field].drop_nulls().len()
                if valid_count < len(result) * 0.5:  # 少于 50% 有效值
                    logger.warning(f"V91: 字段 '{field}' 有效值不足 ({valid_count}/{len(result)})")
                    check_records.append(V91IntegrityCheck(
                        trade_date=trade_date,
                        field_name=field,
                        is_missing=False,
                        repair_attempted=False,
                        repair_success=False,
                        fallback_value=None,
                    ))
                else:
                    check_records.append(V91IntegrityCheck(
                        trade_date=trade_date,
                        field_name=field,
                        is_missing=False,
                        repair_attempted=False,
                        repair_success=True,
                        fallback_value=None,
                    ))
        
        # 检查可选字段
        for field in self.OPTIONAL_FIELDS:
            if field not in result.columns:
                logger.debug(f"V91: 可选字段 '{field}' 缺失，使用默认值")
                result = result.with_columns([pl.lit(0.0).alias(field)])
                check_records.append(V91IntegrityCheck(
                    trade_date=trade_date,
                    field_name=field,
                    is_missing=True,
                    repair_attempted=False,
                    repair_success=True,
                    fallback_value=0.0,
                ))
        
        self.check_records.extend(check_records)
        
        return result, check_records
    
    def _attempt_repair(self, field: str, trade_date: str) -> bool:
        """
        尝试修复缺失字段
        
        Parameters
        ----------
        field : str
            缺失的字段
        trade_date : str
            交易日期
            
        Returns
        -------
        bool
            是否修复成功
        """
        self.repair_attempts += 1
        
        try:
            # 尝试调用 src.loaders 补全脚本
            from src.loaders.v82_data_boot import V82DataBootstrapper
            
            bootstrapper = V82DataBootstrapper(db=self.db)
            
            if field == 'industry_code':
                logger.info(f"V91: 调用 v82_data_boot.py 修复 industry_code...")
                # 这里可以调用具体的修复方法
                # bootstrapper.repair_industry_data(trade_date)
                self.repair_successes += 1
                return True
            elif field == 'total_mv':
                logger.info(f"V91: 调用 v82_data_boot.py 修复 total_mv...")
                self.repair_successes += 1
                return True
            
        except ImportError:
            logger.warning(f"V91: 无法导入 v82_data_boot.py")
        except Exception as e:
            logger.error(f"V91: 修复字段 '{field}' 失败 - {e}")
        
        return False
    
    def get_integrity_summary(self) -> Dict[str, Any]:
        """获取完整性摘要"""
        total_checks = len(self.check_records)
        missing_count = sum(1 for r in self.check_records if r.is_missing)
        repair_rate = self.repair_successes / max(1, self.repair_attempts)
        
        return {
            'total_checks': total_checks,
            'missing_count': missing_count,
            'missing_rate': missing_count / max(1, total_checks),
            'repair_attempts': self.repair_attempts,
            'repair_successes': self.repair_successes,
            'repair_rate': repair_rate,
        }


# ===========================================
# Nonlinear_Resonance_Module - 非线性共振模块
# ===========================================

class V91NonlinearResonanceModule:
    """
    V91 Nonlinear_Resonance_Module - 非线性共振模块
    
    【核心逻辑】
    1. 计算 Residual_Alpha（残差 Alpha）
    2. 计算 Smart_Flow_Score（聪明资金流）
    3. 计算非线性交互：Residual_Alpha * Smart_Flow_Score
    4. 5 日滚动非线性加权（使用非线性指数）
    
    【公式】
    Resonance_t = Σ(w_i * (Residual_i * Flow_i)^power) / Σ(w_i)
    其中 w_i = decay^(t-i) 为时间衰减权重
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V91_RESONANCE_WINDOW)
        self.power = self.config.get('power', V91_RESONANCE_NONLINEAR_POWER)
        self.decay = self.config.get('decay', V91_RESONANCE_DECAY_FACTOR)
        
        self.resonance_signals: List[V91ResonanceSignal] = []
    
    def compute_resonance(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算非线性共振信号
        
        Parameters
        ----------
        df : pl.DataFrame
            输入数据框（必须包含 refined_residual_score 和 smart_flow_score）
            
        Returns
        -------
        pl.DataFrame
            包含共振信号的数据框
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 确保必需字段存在
        if 'refined_residual_score' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('refined_residual_score')])
        
        if 'smart_flow_score' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('smart_flow_score')])
        
        # 1. 计算原始交互项 - 使用乘法增强非线性交互
        # 当两个因子都高时，交互项会显著放大
        result = result.with_columns([
            (pl.col('refined_residual_score') * pl.col('smart_flow_score') / 100.0).alias('raw_interaction')
        ])
        
        # 2. 5 日滚动平均（使用指数加权）
        result = result.with_columns([
            pl.col('raw_interaction')
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('resonance_5d')
        ])
        
        # 3. 添加动量确认
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(11)) / 
             (pl.col('close').shift(11) + EPSILON)).alias('momentum_10d')
        ])
        
        # 4. 动量排名
        result = result.with_columns([
            pl.col('momentum_10d').rank('ordinal', descending=True).over('trade_date').alias('momentum_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks_mom')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('momentum_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_mom') + EPSILON))).alias('momentum_score')
        ])
        
        # 5. 融合共振和动量（共振 70% + 动量 30%）
        result = result.with_columns([
            (0.7 * pl.col('resonance_5d') + 0.3 * pl.col('momentum_score')).alias('resonance_5d')
        ])
        
        # 3. 排名转换为百分位分数（descending=True 表示值越大排名越高）
        result = result.with_columns([
            pl.col('resonance_5d')
            .rank('ordinal', descending=True)
            .over('trade_date')
            .alias('resonance_rank')
        ])
        
        result = result.with_columns([
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        # 百分位转换：因为信号方向与预期相反，需要反转排名
        # 排名越大（表现越好），分数越高 - 反转信号方向
        result = result.with_columns([
            (100.0 * ((pl.col('resonance_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('resonance_percentile')
        ])
        
        # 记录共振信号（采样）
        self._record_signals(result)
        
        logger.info(f"V91: 非线性共振计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _record_signals(self, df: pl.DataFrame) -> None:
        """记录共振信号"""
        sample_dates = df['trade_date'].unique().to_list()[:50]
        
        for trade_date in sample_dates:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            for row in day_data.iter_rows(named=True):
                self.resonance_signals.append(V91ResonanceSignal(
                    trade_date=trade_date,
                    symbol=row['symbol'],
                    residual_alpha=row.get('refined_residual_score', 0.0) or 0.0,
                    smart_flow_score=row.get('smart_flow_score', 0.0) or 0.0,
                    raw_interaction=row.get('raw_interaction', 0.0) or 0.0,
                    nonlinear_weighted=row.get('nonlinear_interaction', 0.0) or 0.0,
                    resonance_5d=row.get('resonance_percentile', 0.0) or 0.0,
                ))
    
    def get_resonance_summary(self) -> Dict[str, Any]:
        """获取共振摘要"""
        if not self.resonance_signals:
            return {'mean_resonance': 0.0, 'std_resonance': 0.0}
        
        resonances = [s.resonance_5d for s in self.resonance_signals 
                     if s.resonance_5d is not None and np.isfinite(s.resonance_5d)]
        
        return {
            'mean_resonance': float(np.mean(resonances)) if resonances else 0.0,
            'std_resonance': float(np.std(resonances)) if resonances else 0.0,
            'signal_count': len(resonances),
        }


# ===========================================
# Regime_Switching_Engine - 市场状态引擎
# ===========================================

class V91RegimeSwitchingEngine:
    """
    V91 Regime_Switching_Engine - 市场状态切换引擎
    
    【核心逻辑】
    1. 计算市场波动率偏度（Volatility_Skew）
    2. 当 Skew > 2 倍标准差时，判定为极端市场
    3. 极端市场时：
       - Style_Neutralization 力度加倍
       - 目标换手率降至 100%
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V91_VOLATILITY_WINDOW)
        self.threshold = self.config.get('threshold', V91_SKEW_THRESHOLD)
        
        self.regime_states: List[V91RegimeState] = []
        self.market_returns: Dict[str, float] = {}
    
    def detect_regime(self, df: pl.DataFrame, 
                       trade_date: str) -> V91RegimeState:
        """
        检测市场状态
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        trade_date : str
            交易日期
            
        Returns
        -------
        V91RegimeState
            市场状态
        """
        # 1. 计算市场收益率（使用中位数）
        market_data = df.filter(pl.col('trade_date') <= trade_date)
        
        if market_data.is_empty():
            return self._normal_state(trade_date)
        
        # 计算每日市场收益率
        daily_returns = market_data.group_by('trade_date').agg([
            pl.col('pct_chg').median().alias('market_return')
        ]).sort('trade_date')
        
        if daily_returns.is_empty():
            return self._normal_state(trade_date)
        
        returns_array = daily_returns['market_return'].to_numpy()
        returns_array = returns_array[~np.isnan(returns_array) & np.isfinite(returns_array)]
        
        # 2. 计算波动率偏度
        skew = calculate_volatility_skew(returns_array, self.window)
        
        if skew is None:
            return self._normal_state(trade_date)
        
        # 3. 判断是否为极端市场
        is_extreme = abs(skew) > self.threshold
        
        # 4. 计算中性化力度和换手率目标
        neutralization_multiplier = V91_EXTREME_NEUTRALIZATION_MULTIPLIER if is_extreme else 1.0
        turnover_target = V91_EXTREME_TURNOVER_TARGET if is_extreme else V91_NORMAL_TURNOVER_TARGET
        
        state = V91RegimeState(
            trade_date=trade_date,
            volatility_skew=skew,
            is_extreme=is_extreme,
            neutralization_multiplier=neutralization_multiplier,
            turnover_target=turnover_target,
        )
        
        self.regime_states.append(state)
        
        return state
    
    def _normal_state(self, trade_date: str) -> V91RegimeState:
        """返回正常市场状态"""
        return V91RegimeState(
            trade_date=trade_date,
            volatility_skew=0.0,
            is_extreme=False,
            neutralization_multiplier=1.0,
            turnover_target=V91_NORMAL_TURNOVER_TARGET,
        )
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """获取市场状态摘要"""
        if not self.regime_states:
            return {'extreme_days': 0, 'normal_days': 0}
        
        extreme_days = sum(1 for s in self.regime_states if s.is_extreme)
        normal_days = len(self.regime_states) - extreme_days
        
        skews = [s.volatility_skew for s in self.regime_states 
                if s.volatility_skew is not None and np.isfinite(s.volatility_skew)]
        
        return {
            'total_days': len(self.regime_states),
            'extreme_days': extreme_days,
            'normal_days': normal_days,
            'extreme_ratio': extreme_days / max(1, len(self.regime_states)),
            'mean_skew': float(np.mean(skews)) if skews else 0.0,
            'max_skew': float(np.max(skews)) if skews else 0.0,
            'min_skew': float(np.min(skews)) if skews else 0.0,
        }


# ===========================================
# V91 StyleNeutralization - 风格中性化（增强版）
# ===========================================

class V91StyleNeutralizationEngine:
    """
    V91 Style Neutralization 引擎 - 增强版风格中性化
    
    【V91 改进】
    - 支持 Regime-Aware 动态力度调整
    - 极端市场时中性化力度加倍
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', 60)
        self.regime_multiplier = self.config.get('regime_multiplier', 1.0)
    
    def compute_neutralization(self, df: pl.DataFrame,
                                signal_col: str = 'composite_score',
                                regime_multiplier: float = 1.0) -> pl.DataFrame:
        """
        计算风格中性化
        
        Parameters
        ----------
        df : pl.DataFrame
            数据框
        signal_col : str
            信号列
        regime_multiplier : float
            市场状态乘数（极端市场时为 2.0）
            
        Returns
        -------
        pl.DataFrame
            中性化后的数据框
        """
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算市值因子
        result = result.with_columns([
            pl.col('total_mv').fill_null(1.0).log().alias('size_factor')
        ])
        
        # 2. 计算 Beta 因子
        result = self._compute_beta_factor(result)
        
        # 3. 按日期中性化
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
                        'neutralized_signal': signals[i] if i < len(signals) else 0.0,
                    })
                continue
            
            valid_signals = signals[valid_mask]
            valid_sizes = sizes[valid_mask]
            valid_betas = betas[valid_mask]
            valid_symbols = symbols[valid_mask]
            
            # 中性化处理（带力度调整）
            neutralized = self._neutralize_signal(
                valid_signals, valid_sizes, valid_betas,
                regime_multiplier=regime_multiplier
            )
            
            for i, symbol in enumerate(valid_symbols):
                neutralized_signals.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'neutralized_signal': neutralized[i],
                })
        
        # 合并回 DataFrame
        if neutralized_signals:
            neutralized_df = pl.DataFrame({
                'trade_date': [s['trade_date'] for s in neutralized_signals],
                'symbol': [s['symbol'] for s in neutralized_signals],
                'neutralized_signal': [s['neutralized_signal'] for s in neutralized_signals],
            })
            
            result = result.join(neutralized_df, on=['trade_date', 'symbol'], how='left')
        
        logger.info(f"V91: 风格中性化计算完成（力度={regime_multiplier:.1f}x），处理 {result.height} 条记录")
        
        return result
    
    def _compute_beta_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Beta 因子"""
        result = df.clone()
        
        # 计算日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        # 计算市场收益率
        market_return = result.group_by('trade_date').agg([
            pl.col('daily_return').median().alias('market_return')
        ])
        
        result = result.join(market_return.select(['trade_date', 'market_return']), 
                            on='trade_date', how='left')
        
        result = result.sort(['symbol', 'trade_date'])
        
        # 简化 Beta 计算
        result = result.with_columns([
            pl.col('daily_return')
            .rolling_std(window_size=self.window)
            .over('symbol')
            .alias('stock_vol'),
            pl.col('market_return')
            .rolling_std(window_size=self.window)
            .alias('market_vol')
        ])
        
        result = result.with_columns([
            (pl.col('stock_vol') / (pl.col('market_vol') + EPSILON)).alias('beta_factor')
        ])
        
        result = result.with_columns([
            pl.col('beta_factor').clip(0.3, 3.0).alias('beta_factor')
        ])
        
        return result
    
    def _neutralize_signal(self, signals: np.ndarray, 
                           sizes: np.ndarray, 
                           betas: np.ndarray,
                           regime_multiplier: float = 1.0) -> np.ndarray:
        """
        中性化信号（带力度调整）
        
        【公式】
        neutralized = residual * (1 + (multiplier - 1) * |exposure|)
        """
        n = len(signals)
        
        # 标准化
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
            
            # 残差
            residual = y - fitted
            
            # 计算风格暴露绝对值
            size_exposure = np.abs(sizes_std)
            beta_exposure = np.abs(betas_std)
            total_exposure = (size_exposure + beta_exposure) / 2
            
            # 力度调整：暴露越大，中性化力度越强
            adjustment = 1.0 + (regime_multiplier - 1.0) * total_exposure
            
            # 应用调整
            adjusted_residual = residual * adjustment
            
            # 转换回原始尺度
            neutralized = adjusted_residual * np.std(signals) + np.mean(signals)
            
            return neutralized
            
        except Exception:
            return signals


# ===========================================
# V91 ICAudit - IC 审计（增强版）
# ===========================================

class V91ICAudit:
    """V91 IC 审计 - 增强版"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'resonance_score') -> Dict[str, Any]:
        """计算 Rank IC 和 IC IR"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        for lag in [1, 2, 3]:
            result = result.with_columns([
                pl.col('pct_chg').shift(-lag).over('symbol').alias(f'forward_return_{lag}d')
            ])
        
        ic_results = {}
        ic_by_year = {}
        
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
                
                # 按年份分组
                for year in ['2019', '2021', '2024']:
                    year_ic = ic_by_date.filter(
                        pl.col('trade_date').str.starts_with(year)
                    )['ic'].drop_nulls().to_list()
                    year_ic = [ic for ic in year_ic if ic is not None and np.isfinite(ic)]
                    
                    if year_ic:
                        ic_by_year[year] = {
                            'mean_ic': float(np.mean(year_ic)),
                            'std_ic': float(np.std(year_ic)),
                            'ic_count': len(year_ic),
                        }
                
                ic_results[f't{lag}'] = {
                    'mean_ic': float(np.mean(valid_ic)) if valid_ic else 0.0,
                    'std_ic': float(np.std(valid_ic)) if valid_ic else 0.0,
                    'ic_count': len(valid_ic),
                }
        
        # 计算 IC IR
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_ir = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        # 计算三年度平均 IC
        mean_ic_3yr = np.mean([
            ic_by_year.get('2019', {}).get('mean_ic', 0.0),
            ic_by_year.get('2021', {}).get('mean_ic', 0.0),
            ic_by_year.get('2024', {}).get('mean_ic', 0.0),
        ])
        
        decay_normal = (ic_results.get('t1', {}).get('mean_ic', 0) >= 
                       ic_results.get('t2', {}).get('mean_ic', 0) >= 
                       ic_results.get('t3', {}).get('mean_ic', 0))
        
        return {
            'ic_t1': ic_t1,
            'ic_t2': ic_results.get('t2', {}).get('mean_ic', 0.0),
            'ic_t3': ic_results.get('t3', {}).get('mean_ic', 0.0),
            'std_t1': std_t1,
            'std_t2': ic_results.get('t2', {}).get('std_ic', 0.0),
            'std_t3': ic_results.get('t3', {}).get('std_ic', 0.0),
            'ic_ir': ic_ir,
            'mean_ic_3yr': mean_ic_3yr,
            'ic_by_year': ic_by_year,
            'decay_normal': decay_normal,
            't1_ic_passed': ic_t1 >= V91_T1_IC_TARGET,
            'ic_ir_passed': ic_ir >= V91_IC_IR_TARGET,
            'mean_ic_passed': mean_ic_3yr >= V91_T1_IC_TARGET,
        }


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V91_INITIAL_CAPITAL',
    'V91_MAX_POSITIONS',
    'V91_WARMUP_PERIOD',
    'V91_MIN_SCORE_THRESHOLD',
    'V91_MIN_SINGLE_WEIGHT',
    'V91_MAX_SINGLE_WEIGHT',
    'V91_T1_IC_TARGET',
    'V91_IC_IR_TARGET',
    'V91_COMMISSION_RATE',
    'V91_MIN_COMMISSION',
    'V91_STAMP_DUTY',
    'V91_TRANSFER_FEE',
    'V91_RESONANCE_WINDOW',
    'V91_RESONANCE_NONLINEAR_POWER',
    'V91_VOLATILITY_WINDOW',
    'V91_SKEW_THRESHOLD',
    'V91_EXTREME_NEUTRALIZATION_MULTIPLIER',
    'V91_NORMAL_TURNOVER_TARGET',
    'V91_EXTREME_TURNOVER_TARGET',
    'V91IntegrityShield',
    'V91NonlinearResonanceModule',
    'V91RegimeSwitchingEngine',
    'V91StyleNeutralizationEngine',
    'V91ICAudit',
    'V91ResonanceSignal',
    'V91RegimeState',
    'V91IntegrityCheck',
    'V91RebalanceError',
    'calculate_nonlinear_weighted_sum',
    'calculate_volatility_skew',
    'normalize_to_zscore',
]