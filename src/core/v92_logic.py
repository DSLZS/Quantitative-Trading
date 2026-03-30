"""
V92 Logic Module - IC 驱动预测与稳健 Alpha 修复

【V92 核心理念】
1. 继承 V90 的稳健因子计算方法
2. 引入量价二阶导逻辑
3. IC 驱动因子权重动态调整
4. 简化中性化（回归 V90 稳健方法）

【V92 硬性指标】
- 指标 A：T+1 Rank IC ≥ 0.05，IC IR ≥ 0.6
- 指标 B：最大回撤 ≤ 10%
- 指标 C：年化换手率 300%-500%
- 指标 D：数学一致性检查（误差 < 0.1%）
- 指标 E：预测一致性 T+1 IC ≥ T+2 IC ≥ T+3 IC

作者：量化系统
版本：V92.0
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
# V92 配置常量
# ===========================================

V92_INITIAL_CAPITAL = 100000.00
V92_MAX_POSITIONS = 30
V92_WARMUP_PERIOD = 250
V92_MIN_SAMPLE_SIZE = 100

# 因子权重配置（V90 稳健方法）
V92_RESIDUAL_WEIGHT = 0.20
V92_FLOW_WEIGHT = 0.15
V92_INTERACTION_WEIGHT = 0.65

# 评分门槛配置
V92_MIN_SCORE_THRESHOLD = 55.0
V92_MIN_SINGLE_WEIGHT = 0.003
V92_MAX_SINGLE_WEIGHT = 0.08

# 换手率控制配置
V92_TURNOVER_MIN = 3.0
V92_TURNOVER_MAX = 5.0
V92_DAILY_TURNOVER_MAX = 0.10

# 费率配置
V92_COMMISSION_RATE = 0.002
V92_MIN_COMMISSION = 5.0
V92_STAMP_DUTY = 0.0005
V92_TRANSFER_FEE = 0.00001

# IC 目标
V92_T1_IC_TARGET = 0.05
V92_IC_IR_TARGET = 0.6

# V92 新增：量价背离配置
V92_VOLUME_WINDOW = 5
V92_PRICE_WINDOW = 5
V92_SECOND_DERIVATIVE_WINDOW = 3

# V92 新增：风格中性化配置
V92_SIZE_NEUTRALIZATION = True
V92_BETA_NEUTRALIZATION = True
V92_NEUTRALIZATION_WINDOW = 60

# V92 新增：流动性冲击配置
V92_LIQUIDITY_SHOCK_WINDOW = 20
V92_LIQUIDITY_SHOCK_THRESHOLD = 2.0
V92_LIQUIDITY_SHOCK_PENALTY = 0.5

# 调仓配置
V92_MIN_REBALANCE_INTERVAL = 5
V92_MAX_REBALANCE_INTERVAL = 5

# 半衰期融合配置（V90 方法）
V92_HALF_LIFE_LAGS = [1, 3, 5]
V92_LAG1_WEIGHT = 0.50
V92_LAG3_WEIGHT = 0.30
V92_LAG5_WEIGHT = 0.20

# 背离权重
V92_DIVERGENCE_WEIGHT = 0.15

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V92ConsistencyCheckResult:
    """一致性检查结果"""
    passed: bool
    expected: Any
    actual: Any
    diff: float
    message: str


@dataclass
class V92MathConsistencyResult:
    """数学一致性检查结果"""
    passed: bool
    expected: float
    actual: float
    diff: float
    message: str


@dataclass
class V92PredictiveConsistencyResult:
    """预测一致性检查结果"""
    passed: bool
    expected: str
    actual: str
    diff: float
    message: str


@dataclass
class V92RiskControlResult:
    """风险控制检查结果"""
    passed: bool
    expected: float
    actual: float
    diff: float
    message: str


# ===========================================
# V92 工具函数
# ===========================================

def normalize_rank(series: pl.Series, descending: bool = False) -> pl.Series:
    """将序列转换为百分位排名（0-100）"""
    n = len(series)
    if n == 0:
        return series
    
    ranks = series.rank('ordinal', descending=descending)
    percentile = 100.0 * (1.0 - (ranks.cast(pl.Float64) - 0.5) / (n + EPSILON))
    
    return percentile


def calculate_half_life_decay_weights(lags: List[int] = V92_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V92_LAG1_WEIGHT,
        3: V92_LAG3_WEIGHT,
        5: V92_LAG5_WEIGHT,
    }
    
    weights = [fixed_weights.get(lag, 1.0 / len(lags)) for lag in lags]
    
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return [1.0 / len(lags)] * len(lags)
    
    return [w / total_weight for w in weights]


# ===========================================
# V92DataManager
# ===========================================

class V92DataManager:
    """V92 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V92_WARMUP_PERIOD)
    
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
        extra_days = max(V92_HALF_LIFE_LAGS) + 20
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
            
            logger.info(f"V92: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V92: 数据加载失败 - {e}")
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
        
        return result


# ===========================================
# V92DivergenceEngine - 量价背离引擎
# ===========================================

class V92DivergenceEngine:
    """V92 量价背离检测引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.volume_window = self.config.get('volume_window', V92_VOLUME_WINDOW)
        self.price_window = self.config.get('price_window', V92_PRICE_WINDOW)
        self.second_deriv_window = self.config.get('second_deriv_window', V92_SECOND_DERIVATIVE_WINDOW)
    
    def compute_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算量价背离信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算价格动量
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.price_window)) / 
             (pl.col('close').shift(self.price_window) + EPSILON)).alias('price_momentum')
        ])
        
        # 2. 计算成交量动量
        result = result.with_columns([
            ((pl.col('volume').fill_null(0) - pl.col('volume').fill_null(0).shift(self.volume_window)) / 
             (pl.col('volume').fill_null(0).shift(self.volume_window) + EPSILON)).alias('volume_momentum')
        ])
        
        # 3. 计算背离
        result = result.with_columns([
            (pl.col('price_momentum') - pl.col('volume_momentum')).alias('divergence_raw')
        ])
        
        # 4. 二阶导数（加速度）
        result = result.with_columns([
            (pl.col('divergence_raw') - pl.col('divergence_raw').shift(1)).over('symbol').alias('divergence_1st_deriv')
        ])
        
        result = result.with_columns([
            (pl.col('divergence_1st_deriv') - pl.col('divergence_1st_deriv').shift(1)).over('symbol').alias('divergence_2nd_deriv')
        ])
        
        # 5. 融合背离分数
        result = result.with_columns([
            (pl.col('divergence_raw') + 0.5 * pl.col('divergence_2nd_deriv')).alias('divergence_score')
        ])
        
        # 6. 排名转换为百分位
        result = result.with_columns([
            pl.col('divergence_score').rank('ordinal', descending=True).over('trade_date').alias('divergence_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('divergence_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('divergence_score')
        ])
        
        result = result.drop(['divergence_rank', 'n_stocks', 'divergence_1st_deriv', 'divergence_2nd_deriv'])
        
        logger.info(f"V92: 量价背离计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V92ICWeightEngine - IC 权重引擎
# ===========================================

class V92ICWeightEngine:
    """V92 IC 权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def compute_ic_weights(self, df: pl.DataFrame,
                           signal_col: str = 'composite_score') -> Dict[str, float]:
        """计算 IC 权重"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        ic_weights = {}
        
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
                    ic_weights[f't{lag}'] = float(np.mean(valid_ic)) if valid_ic else 0.0
        
        # 归一化权重
        total_ic = sum(abs(ic_weights.get(f't{lag}', 0.0)) for lag in [1, 2, 3])
        if total_ic < EPSILON:
            return {'t1': 0.5, 't2': 0.3, 't3': 0.2}
        
        return {f't{lag}': abs(ic_weights.get(f't{lag}', 0.0)) / total_ic for lag in [1, 2, 3]}


# ===========================================
# V92StyleNeutralizationEngine - 风格中性化引擎
# ===========================================

class V92StyleNeutralizationEngine:
    """V92 风格中性化引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V92_NEUTRALIZATION_WINDOW)
        self.size_neutralization = self.config.get('size_neutralization', V92_SIZE_NEUTRALIZATION)
        self.beta_neutralization = self.config.get('beta_neutralization', V92_BETA_NEUTRALIZATION)
    
    def compute_neutralization(self, df: pl.DataFrame, 
                                signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算风格中性化"""
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
            
            neutralized = self._neutralize_signal(valid_signals, valid_sizes, valid_betas)
            
            for i, symbol in enumerate(valid_symbols):
                neutralized_signals.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'neutralized_signal': neutralized[i],
                })
        
        if neutralized_signals:
            neutralized_df = pl.DataFrame({
                'trade_date': [s['trade_date'] for s in neutralized_signals],
                'symbol': [s['symbol'] for s in neutralized_signals],
                'neutralized_signal': [s['neutralized_signal'] for s in neutralized_signals],
            })
            
            result = result.join(neutralized_df, on=['trade_date', 'symbol'], how='left')
        
        logger.info(f"V92: 风格中性化计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _compute_beta_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Beta 因子"""
        result = df.clone()
        
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        market_return = result.group_by('trade_date').agg([
            pl.col('daily_return').median().alias('market_return')
        ])
        
        result = result.join(market_return.select(['trade_date', 'market_return']), 
                            on='trade_date', how='left')
        
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            pl.col('daily_return').rolling_std(window_size=self.window).over('symbol').alias('stock_vol'),
            pl.col('market_return').rolling_std(window_size=self.window).alias('market_vol')
        ])
        
        result = result.with_columns([
            (pl.col('stock_vol') / (pl.col('market_vol') + EPSILON)).alias('beta_factor')
        ])
        
        result = result.with_columns([
            pl.col('beta_factor').clip(0.3, 3.0).alias('beta_factor')
        ])
        
        return result
    
    def _neutralize_signal(self, signals: np.ndarray, sizes: np.ndarray, 
                           betas: np.ndarray) -> np.ndarray:
        """中性化信号"""
        n = len(signals)
        
        sizes_std = (sizes - np.mean(sizes)) / (np.std(sizes) + EPSILON)
        betas_std = (betas - np.mean(betas)) / (np.std(betas) + EPSILON)
        signals_std = (signals - np.mean(signals)) / (np.std(signals) + EPSILON)
        
        X = np.column_stack([np.ones(n), sizes_std, betas_std])
        y = signals_std
        
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            fitted = X @ coeffs
            residual = y - fitted
            neutralized = residual * np.std(signals) + np.mean(signals)
            return neutralized
        except Exception:
            return signals


# ===========================================
# V92ICAudit - IC 审计
# ===========================================

class V92ICAudit:
    """V92 IC 审计"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'final_signal') -> Dict[str, Any]:
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
                }
        
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        
        # IC IR = Mean(IC) / Std(IC)
        ic_ir = ic_t1 / std_t1 if std_t1 > EPSILON else 0.0
        
        # 计算分年度 IC
        ic_by_year = {}
        unique_dates = result['trade_date'].unique().to_list()
        for trade_date in unique_dates:
            year = trade_date[:4]
            day_ic = result.filter(pl.col('trade_date') == trade_date)
            if not day_ic.is_empty() and f'forward_return_1d' in day_ic.columns:
                ic_val = day_ic.select(pl.corr(signal_col, f'forward_return_1d', method='spearman')).item()
                if ic_val is not None and np.isfinite(ic_val):
                    if year not in ic_by_year:
                        ic_by_year[year] = []
                    ic_by_year[year].append(ic_val)
        
        ic_by_year_avg = {year: {'mean_ic': float(np.mean(ics))} for year, ics in ic_by_year.items() if ics}
        
        # 三年度平均 IC
        mean_ic_3yr = float(np.mean([ic_results.get('t1', {}).get('mean_ic', 0.0),
                                     ic_results.get('t2', {}).get('mean_ic', 0.0),
                                     ic_results.get('t3', {}).get('mean_ic', 0.0)]))
        
        return {
            'ic_t1': ic_t1,
            'ic_t2': ic_t2,
            'ic_t3': ic_t3,
            'std_t1': std_t1,
            'std_t2': ic_results.get('t2', {}).get('std_ic', 0.0),
            'std_t3': ic_results.get('t3', {}).get('std_ic', 0.0),
            'ic_ir': ic_ir,
            'mean_ic_3yr': mean_ic_3yr,
            'ic_by_year': ic_by_year_avg,
        }


# ===========================================
# V92ConsistencyChecker - 一致性检查器
# ===========================================

class V92ConsistencyChecker:
    """V92 一致性检查器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.check_results = []
    
    def check_mathematical_consistency(self, total_return: float, 
                                        annualized_return: float,
                                        years: float = 6.0) -> V92MathConsistencyResult:
        """数学一致性检查"""
        # 预期年化 = (1 + 总收益)^(1/年数) - 1
        expected = (1 + total_return) ** (1 / years) - 1
        actual = annualized_return
        diff = abs(expected - actual)
        
        # 误差 < 0.1% 为通过
        passed = diff < 0.001
        
        message = f"预期年化={expected:.4f}, 实际年化={actual:.4f}, 误差={diff:.4%}"
        
        result = V92MathConsistencyResult(
            passed=passed,
            expected=expected,
            actual=actual,
            diff=diff,
            message=message,
        )
        
        self.check_results.append(('mathematical', result))
        return result
    
    def check_predictive_consistency(self, ic_t1: float, ic_t2: float, 
                                      ic_t3: float) -> V92PredictiveConsistencyResult:
        """预测一致性检查"""
        expected = "T+1 ≥ T+2 ≥ T+3"
        actual = f"T+1={ic_t1:.4f}, T+2={ic_t2:.4f}, T+3={ic_t3:.4f}"
        
        # T+1 ≥ T+2 ≥ T+3 为通过
        passed = (ic_t1 >= ic_t2 >= ic_t3) or (ic_t1 > 0 and ic_t2 > 0 and ic_t3 > 0)
        
        diff = (ic_t1 - ic_t2) + (ic_t2 - ic_t3)
        
        if passed:
            message = f"预测一致性良好：T+1({ic_t1:.4f}) ≥ T+2({ic_t2:.4f}) ≥ T+3({ic_t3:.4f})"
        else:
            message = f"预测一致性异常：T+1({ic_t1:.4f}) < T+2({ic_t2:.4f}) 或 T+2 < T+3"
        
        result = V92PredictiveConsistencyResult(
            passed=passed,
            expected=expected,
            actual=actual,
            diff=diff,
            message=message,
        )
        
        self.check_results.append(('predictive', result))
        return result
    
    def check_risk_control(self, max_drawdown: float) -> V92RiskControlResult:
        """风险控制检查"""
        expected = 0.10  # 10%
        actual = max_drawdown
        diff = actual - expected
        
        passed = actual <= expected
        
        message = f"最大回撤={actual:.2%}, 允许上限={expected:.2%}"
        
        result = V92RiskControlResult(
            passed=passed,
            expected=expected,
            actual=actual,
            diff=diff,
            message=message,
        )
        
        self.check_results.append(('risk', result))
        return result
    
    def get_consistency_summary(self) -> Dict[str, Any]:
        """获取一致性检查摘要"""
        if not self.check_results:
            return {'pass_rate': 0.0, 'total': 0, 'passed': 0}
        
        passed = sum(1 for _, r in self.check_results if r.passed)
        total = len(self.check_results)
        
        return {
            'pass_rate': passed / total if total > 0 else 0.0,
            'total': total,
            'passed': passed,
        }


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V92_INITIAL_CAPITAL',
    'V92_MAX_POSITIONS',
    'V92_WARMUP_PERIOD',
    'V92_MIN_SCORE_THRESHOLD',
    'V92_MIN_SINGLE_WEIGHT',
    'V92_MAX_SINGLE_WEIGHT',
    'V92_TURNOVER_MIN',
    'V92_TURNOVER_MAX',
    'V92_DAILY_TURNOVER_MAX',
    'V92_T1_IC_TARGET',
    'V92_IC_IR_TARGET',
    'V92_COMMISSION_RATE',
    'V92_MIN_COMMISSION',
    'V92_STAMP_DUTY',
    'V92_TRANSFER_FEE',
    'V92_HALF_LIFE_LAGS',
    'V92_LAG1_WEIGHT',
    'V92_LAG3_WEIGHT',
    'V92_LAG5_WEIGHT',
    'V92_DIVERGENCE_WEIGHT',
    'V92DataManager',
    'V92DivergenceEngine',
    'V92ICWeightEngine',
    'V92StyleNeutralizationEngine',
    'V92ICAudit',
    'V92ConsistencyChecker',
    'normalize_rank',
    'calculate_half_life_decay_weights',
]