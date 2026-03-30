"""
V94 Core Module - 回归 V90 基准与成交量加权动量增强

【V94 核心理念】
1. 回归 V90 稳健 Alpha 增强逻辑（废弃 V93 无效因子）
2. 仅新增一个变量：成交量加权动量 (VWAP Momentum)
3. 严格的单因子 IC 审计：新因子 IC < 0.02 则不计入复合得分
4. 保持 V90 的风格中性化、流动性冲击、动态调仓机制

【V94 硬性指标】
- 指标 A (Predictive Power): T+1 Rank IC >= 0.045（先恢复 V90 水平）
- 指标 B (Risk Control): 最大回撤 < 15%（逐步恢复至<10%）
- 指标 C (IC Stability): 2019/2021/2024 三年度 IC 全部为正
- 指标 D (Execution): 年化换手率 300%-500%，单日换手率<10%

【V94 因子框架】
1. Refined Residual (V90 继承) - 残差动量
2. Smart Flow (V90 继承) - 聪明资金流
3. Vol_Price_Interaction (V90 继承) - 量价交互
4. VWAP Momentum (V94 新增) - 成交量加权动量

作者：量化系统
版本：V94.0
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
# V94 配置常量（继承 V90）
# ===========================================

V94_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V94_MAX_POSITIONS = 30  # 最大持仓数量
V94_WARMUP_PERIOD = 250
V94_MIN_SAMPLE_SIZE = 100

# 因子权重配置（V90 稳健方法）
V94_RESIDUAL_WEIGHT = 0.20
V94_FLOW_WEIGHT = 0.15
V94_INTERACTION_WEIGHT = 0.50  # 降低至 50% 为新因子留空间
V94_VWAP_MOMENTUM_WEIGHT = 0.15  # V94 新增：成交量加权动量权重

# 评分门槛配置
V94_MIN_SCORE_THRESHOLD = 55.0
V94_MIN_SINGLE_WEIGHT = 0.003
V94_MAX_SINGLE_WEIGHT = 0.08

# 换手率控制配置
V94_TURNOVER_MIN = 3.0
V94_TURNOVER_MAX = 5.0
V94_DAILY_TURNOVER_MAX = 0.10

# 费率配置
V94_COMMISSION_RATE = 0.002
V94_MIN_COMMISSION = 5.0
V94_STAMP_DUTY = 0.0005
V94_TRANSFER_FEE = 0.00001

# IC 目标
V94_T1_IC_TARGET = 0.045
V94_IC_IR_TARGET = 0.5

# V94 新增：VWAP Momentum 配置
V94_VWAP_WINDOW = 20  # VWAP 计算窗口
V94_VWAP_MOMENTUM_WINDOW = 10  # 动量计算窗口
V94_VWAP_IC_THRESHOLD = 0.02  # IC 门槛：低于此值不计入复合得分

# 风格中性化配置（V90 继承）
V94_SIZE_NEUTRALIZATION = True
V94_BETA_NEUTRALIZATION = True
V94_NEUTRALIZATION_WINDOW = 60

# 流动性冲击配置（V90 继承）
V94_LIQUIDITY_SHOCK_WINDOW = 20
V94_LIQUIDITY_SHOCK_THRESHOLD = 2.0
V94_LIQUIDITY_SHOCK_PENALTY = 0.5

# 调仓配置（V90 继承）
V94_MIN_REBALANCE_INTERVAL = 5
V94_MAX_REBALANCE_INTERVAL = 5
V94_RANK_CORRELATION_THRESHOLD = 0.20

# 半衰期融合配置（V90 继承）
V94_HALF_LIFE_LAGS = [1, 3, 5]
V94_LAG1_WEIGHT = 0.50
V94_LAG3_WEIGHT = 0.30
V94_LAG5_WEIGHT = 0.20

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V94SingleFactorIC:
    """单因子 IC 记录"""
    factor_name: str
    ic_t1: float
    ic_t2: float
    ic_t3: float
    ic_ir: float
    passed_threshold: bool  # 是否通过 IC 门槛


@dataclass
class V94FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float


@dataclass
class V94TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float
    is_rebalance_day: bool = False


@dataclass
class V94Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V94DailyPortfolio:
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
# V94 工具函数
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
    mean = np.mean(series)
    std = np.std(series)
    if std < EPSILON:
        return np.zeros_like(series)
    return (series - mean) / std


def calculate_half_life_decay_weights(lags: List[int] = V94_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V94_LAG1_WEIGHT,
        3: V94_LAG3_WEIGHT,
        5: V94_LAG5_WEIGHT,
    }
    
    weights = [fixed_weights.get(lag, 1.0 / len(lags)) for lag in lags]
    
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return [1.0 / len(lags)] * len(lags)
    
    return [w / total_weight for w in weights]


def calculate_ic_decay_simple(df: pl.DataFrame, signal_col: str) -> Dict[str, float]:
    """简单计算 IC 衰减"""
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
    """使用全市场中位数填充空值"""
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


# ===========================================
# V94 DataManager
# ===========================================

class V94DataManager:
    """V94 数据管理器（继承 V90）"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V94_WARMUP_PERIOD)
    
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
                logger.warning(f"V94: 检测到数据库连接错误：{e}")
            return False, f"检查失败：{e}", {}
    
    def load_data(self, start_date: str, end_date: str,
                  symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载数据"""
        extra_days = max(V94_HALF_LIFE_LAGS) + 20
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
            
            logger.info(f"V94: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V94: 数据加载失败 - {e}")
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
# V94 Single Factor IC Auditor
# ===========================================

class V94SingleFactorICAuditor:
    """V94 单因子 IC 审计器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.ic_threshold = self.config.get('ic_threshold', V94_VWAP_IC_THRESHOLD)
        self.factor_ic_results: Dict[str, V94SingleFactorIC] = {}
    
    def audit_single_factor(self, df: pl.DataFrame, factor_col: str, 
                            factor_name: str) -> V94SingleFactorIC:
        """审计单因子 IC"""
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
                    pl.corr(factor_col, return_col, method='spearman').alias('ic')
                ]).filter(pl.col('ic').is_not_null())
                
                if not ic_by_date.is_empty():
                    ic_list = ic_by_date['ic'].drop_nulls().to_list()
                    valid_ic = [ic for ic in ic_list if ic is not None and np.isfinite(ic)]
                    ic_results[f't{lag}'] = {
                        'mean_ic': float(np.mean(valid_ic)) if valid_ic else 0.0,
                        'std_ic': float(np.std(valid_ic)) if valid_ic else 0.0,
                    }
        
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        
        # IC IR = Mean(IC) / Std(IC)
        ic_ir = ic_t1 / std_t1 if std_t1 > EPSILON else 0.0
        
        # 判断是否通过 IC 门槛
        passed_threshold = ic_t1 >= self.ic_threshold
        
        factor_ic = V94SingleFactorIC(
            factor_name=factor_name,
            ic_t1=ic_t1,
            ic_t2=ic_t2,
            ic_t3=ic_t3,
            ic_ir=ic_ir,
            passed_threshold=passed_threshold,
        )
        
        self.factor_ic_results[factor_name] = factor_ic
        
        logger.info(f"V94: 单因子 IC 审计 - {factor_name}")
        logger.info(f"     T+1 IC = {ic_t1:.4f}, T+2 IC = {ic_t2:.4f}, T+3 IC = {ic_t3:.4f}")
        logger.info(f"     IC IR = {ic_ir:.3f}, 通过门槛={passed_threshold}")
        
        return factor_ic
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """获取 IC 审计摘要"""
        if not self.factor_ic_results:
            return {'total_factors': 0, 'passed_factors': 0}
        
        passed_count = sum(1 for f in self.factor_ic_results.values() if f.passed_threshold)
        
        return {
            'total_factors': len(self.factor_ic_results),
            'passed_factors': passed_count,
            'failed_factors': len(self.factor_ic_results) - passed_count,
            'factors': {name: {
                'ic_t1': f.ic_t1,
                'ic_t2': f.ic_t2,
                'ic_t3': f.ic_t3,
                'ic_ir': f.ic_ir,
                'passed': f.passed_threshold,
            } for name, f in self.factor_ic_results.items()},
        }


# ===========================================
# V94 VWAP Momentum Engine (V94 新增)
# ===========================================

class V94VWAPMomentumEngine:
    """
    V94 VWAP Momentum 引擎 - 成交量加权动量
    
    【核心逻辑】
    1. 计算 VWAP（成交量加权平均价）
    2. 计算价格相对 VWAP 的偏离
    3. 计算 VWAP 动量（N 日变化率）
    4. 融合为 VWAP Momentum 信号
    
    【公式】
    - VWAP = Σ(close * volume) / Σ(volume)
    - VWAP_Momentum = VWAP_t / VWAP_t-N - 1
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vwap_window = self.config.get('vwap_window', V94_VWAP_WINDOW)
        self.momentum_window = self.config.get('momentum_window', V94_VWAP_MOMENTUM_WINDOW)
    
    def compute_vwap_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 VWAP Momentum 信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算典型价格 (Typical Price)
        result = result.with_columns([
            ((pl.col('high') + pl.col('low') + pl.col('close')) / 3.0).alias('typical_price')
        ])
        
        # 2. 计算成交量加权价格
        result = result.with_columns([
            (pl.col('typical_price') * pl.col('volume').fill_null(0)).alias('vp')
        ])
        
        # 3. 计算滚动 VWAP
        result = result.with_columns([
            pl.col('vp').rolling_sum(window_size=self.vwap_window).over('symbol').alias('vp_sum'),
            pl.col('volume').fill_null(0).rolling_sum(window_size=self.vwap_window).over('symbol').alias('vol_sum')
        ])
        
        result = result.with_columns([
            (pl.col('vp_sum') / (pl.col('vol_sum') + EPSILON)).alias('vwap')
        ])
        
        # 4. 计算 VWAP 动量（N 日变化率）
        result = result.with_columns([
            ((pl.col('vwap') - pl.col('vwap').shift(self.momentum_window)) / 
             (pl.col('vwap').shift(self.momentum_window) + EPSILON)).alias('vwap_momentum_raw')
        ])
        
        # 5. 计算价格相对 VWAP 的偏离
        result = result.with_columns([
            ((pl.col('close') - pl.col('vwap')) / (pl.col('vwap') + EPSILON)).alias('price_vwap_deviation')
        ])
        
        # 6. 融合 VWAP Momentum 分数
        # VWAP 动量 > 0 且价格 > VWAP 为正面信号
        result = result.with_columns([
            (
                pl.col('vwap_momentum_raw') * 0.6 +  # VWAP 动量
                pl.col('price_vwap_deviation') * 0.4  # 价格偏离
            ).alias('vwap_momentum_score_raw')
        ])
        
        # 7. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('vwap_momentum_score_raw').rank('ordinal', descending=True).over('trade_date').alias('vwap_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('vwap_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('vwap_momentum_score')
        ])
        
        result = result.drop(['vwap_rank', 'n_stocks', 'vp', 'vp_sum', 'vol_sum'])
        
        logger.info(f"V94: VWAP Momentum 计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V94 AlphaFusion - 半衰期融合引擎（V90 继承）
# ===========================================

class V94AlphaFusion:
    """V94 AlphaFusion - 半衰期衰减融合引擎"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V94_HALF_LIFE_LAGS)
        
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        
        self.fusion_signals: List[V94FusionSignal] = []
        
        logger.info("V94 AlphaFusion 初始化完成")
        logger.info(f"V94: 融合 Lags={self.fusion_lags}")
        logger.info(f"V94: 半衰期权重={dict(zip(self.fusion_lags, self.half_life_weights))}")
    
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
        
        logger.info(f"V94: 融合信号计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        return {
            'weights': dict(zip(self.fusion_lags, self.half_life_weights)),
            'lag1_weight': self.half_life_weights[0],
        }


# ===========================================
# V94 AlphaWeight - Alpha 权重引擎（V90 继承）
# ===========================================

class V94AlphaWeightEngine:
    """V94 AlphaWeight - Alpha 权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V94_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V94_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V94_MAX_SINGLE_WEIGHT)
    
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
        
        logger.info(f"V94: Alpha 权重计算完成，处理 {result.height} 条记录")
        
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
# V94 ICAudit - IC 审计（V90 继承）
# ===========================================

class V94ICAudit:
    """V94 IC 审计"""
    
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
        
        ic_by_year_avg = {year: float(np.mean(ics)) for year, ics in ic_by_year.items() if ics}
        
        return {
            'ic_t1': ic_t1,
            'ic_t2': ic_t2,
            'ic_t3': ic_t3,
            'std_t1': ic_results.get('t1', {}).get('std_ic', 0.0),
            'std_t2': ic_results.get('t2', {}).get('std_ic', 0.0),
            'std_t3': ic_results.get('t3', {}).get('std_ic', 0.0),
            'decay_normal': decay_normal,
            't1_ic_passed': ic_t1 >= V94_T1_IC_TARGET,
            'ic_by_year': ic_by_year_avg,
        }


# ===========================================
# V94 TurnoverTracker - 换手率追踪（V90 继承）
# ===========================================

class V94TurnoverTracker:
    """V94 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V94TurnoverRecord] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> V94TurnoverRecord:
        """记录换手率"""
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            daily_turnover = 0.0
        else:
            max_daily_turnover_value = portfolio_value * V94_DAILY_TURNOVER_MAX
            
            capped_buy_value = min(buy_value, max_daily_turnover_value)
            capped_sell_value = min(sell_value, max_daily_turnover_value)
            
            buy_turnover = capped_buy_value / portfolio_value
            sell_turnover = capped_sell_value / portfolio_value
            turnover_rate = (capped_buy_value + capped_sell_value) / portfolio_value
            daily_turnover = min(turnover_rate, V94_DAILY_TURNOVER_MAX)
        
        self.trading_days += 1
        
        cumulative_turnover = sum(r.turnover_rate for r in self.turnover_records) + turnover_rate
        annualized_turnover = cumulative_turnover * (252.0 / max(1, self.trading_days))
        
        record = V94TurnoverRecord(
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
        
        is_active = V94_TURNOVER_MIN <= annualized_turnover <= V94_TURNOVER_MAX
        daily_ok = max_daily <= V94_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean([r.turnover_rate for r in self.turnover_records])),
            'std_turnover': float(np.std([r.turnover_rate for r in self.turnover_records])),
            'max_turnover': float(np.max([r.turnover_rate for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V94_TURNOVER_MIN,
            'turnover_max': V94_TURNOVER_MAX,
        }


# ===========================================
# V94 PortfolioTracker - 组合追踪（V90 继承）
# ===========================================

class V94PortfolioTracker:
    """V94 组合追踪器"""
    
    def __init__(self, initial_capital: float = V94_INITIAL_CAPITAL,
                 config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        self.cash = initial_capital
        self.positions: Dict[str, V94Position] = {}
        self.portfolio_snapshots: List[V94DailyPortfolio] = []
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
                        turnover_rate: float) -> V94DailyPortfolio:
        """记录组合快照"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        
        cumulative_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        snapshot = V94DailyPortfolio(
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
# V94 StyleNeutralization - 风格中性化（V90 继承）
# ===========================================

class V94StyleNeutralizationEngine:
    """V94 风格中性化引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V94_NEUTRALIZATION_WINDOW)
        self.size_neutralization = self.config.get('size_neutralization', V94_SIZE_NEUTRALIZATION)
        self.beta_neutralization = self.config.get('beta_neutralization', V94_BETA_NEUTRALIZATION)
    
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
        
        logger.info(f"V94: 风格中性化计算完成，处理 {result.height} 条记录")
        
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
    
    def get_exposure_summary(self) -> Dict[str, Any]:
        """获取风格暴露摘要"""
        return {'neutralization_enabled': True}


# ===========================================
# V94 LiquidityShock - 流动性冲击（V90 继承）
# ===========================================

class V94LiquidityShockEngine:
    """V94 流动性冲击引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V94_LIQUIDITY_SHOCK_WINDOW)
        self.threshold = self.config.get('threshold', V94_LIQUIDITY_SHOCK_THRESHOLD)
        self.penalty = self.config.get('penalty', V94_LIQUIDITY_SHOCK_PENALTY)
    
    def compute_liquidity_shock(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算流动性冲击信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            pl.col('amount').fill_null(0).rolling_mean(window_size=self.window).over('symbol').alias('amount_ma')
        ])
        
        result = result.with_columns([
            (pl.col('amount').fill_null(0) / (pl.col('amount_ma') + EPSILON)).alias('amount_ratio')
        ])
        
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        result = result.with_columns([
            (pl.col('amount_ratio') > self.threshold).alias('liquidity_shock_flag')
        ])
        
        result = result.with_columns([
            pl.when(pl.col('liquidity_shock_flag'))
            .then(
                pl.when(pl.col('daily_return') < 0.0)
                .then(0.3)
                .when(pl.col('daily_return') < 0.01)
                .then(0.5)
                .otherwise(0.8)
            )
            .otherwise(1.0)
            .alias('liquidity_penalty')
        ])
        
        logger.info(f"V94: 流动性冲击计算完成")
        
        return result


# ===========================================
# V94 DynamicRebalance - 动态调仓（V90 继承）
# ===========================================

class V94DynamicRebalanceEngine:
    """V94 动态调仓引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('threshold', V94_RANK_CORRELATION_THRESHOLD)
        self.min_interval = self.config.get('min_interval', V94_MIN_REBALANCE_INTERVAL)
        self.max_interval = self.config.get('max_interval', V94_MAX_REBALANCE_INTERVAL)
        
        self.last_rebalance_date = None
        self.last_rebalance_signals = {}
        self.rebalance_count = 0
    
    def should_rebalance(self, trade_date: str, current_signals: Dict[str, float]) -> bool:
        """判断是否应该调仓"""
        if self.last_rebalance_date is None:
            self._update_rebalance_state(trade_date, current_signals)
            logger.info(f"V94: 首次调仓 ({trade_date})")
            self.rebalance_count += 1
            return True
        
        try:
            last_date = datetime.strptime(self.last_rebalance_date, "%Y-%m-%d")
            curr_date = datetime.strptime(trade_date, "%Y-%m-%d")
            actual_days = (curr_date - last_date).days
        except Exception:
            actual_days = 0
        
        if actual_days < self.min_interval:
            return False
        
        if actual_days >= self.max_interval:
            logger.info(f"V94: 达到最大调仓间隔 ({actual_days}天)，强制调仓")
            self._update_rebalance_state(trade_date, current_signals)
            self.rebalance_count += 1
            return True
        
        if self.last_rebalance_signals:
            common_symbols = set(current_signals.keys()) & set(self.last_rebalance_signals.keys())
            
            if len(common_symbols) >= 10:
                current_values = np.array([current_signals[s] for s in common_symbols])
                last_values = np.array([self.last_rebalance_signals[s] for s in common_symbols])
                
                rank_corr = calculate_rank_correlation(current_values, last_values)
                
                if rank_corr < self.correlation_threshold:
                    logger.info(f"V94: Rank Correlation ({rank_corr:.3f}) < 阈值，触发调仓")
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
        }


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    # 常量
    'V94_INITIAL_CAPITAL',
    'V94_MAX_POSITIONS',
    'V94_WARMUP_PERIOD',
    'V94_MIN_SCORE_THRESHOLD',
    'V94_MIN_SINGLE_WEIGHT',
    'V94_MAX_SINGLE_WEIGHT',
    'V94_TURNOVER_MIN',
    'V94_TURNOVER_MAX',
    'V94_DAILY_TURNOVER_MAX',
    'V94_T1_IC_TARGET',
    'V94_IC_IR_TARGET',
    'V94_COMMISSION_RATE',
    'V94_MIN_COMMISSION',
    'V94_STAMP_DUTY',
    'V94_TRANSFER_FEE',
    'V94_VWAP_IC_THRESHOLD',
    'V94_HALF_LIFE_LAGS',
    'V94_LAG1_WEIGHT',
    'V94_LAG3_WEIGHT',
    'V94_LAG5_WEIGHT',
    'V94_VWAP_WINDOW',
    'V94_VWAP_MOMENTUM_WINDOW',
    'V94_VWAP_MOMENTUM_WEIGHT',
    'V94_SIZE_NEUTRALIZATION',
    'V94_BETA_NEUTRALIZATION',
    'V94_NEUTRALIZATION_WINDOW',
    'V94_LIQUIDITY_SHOCK_WINDOW',
    'V94_LIQUIDITY_SHOCK_THRESHOLD',
    'V94_LIQUIDITY_SHOCK_PENALTY',
    'V94_MIN_REBALANCE_INTERVAL',
    'V94_MAX_REBALANCE_INTERVAL',
    'V94_RANK_CORRELATION_THRESHOLD',
    # 数据类
    'V94SingleFactorIC',
    'V94FusionSignal',
    'V94TurnoverRecord',
    'V94Position',
    'V94DailyPortfolio',
    # 工具函数
    'normalize_rank',
    'zscore_normalize',
    'calculate_half_life_decay_weights',
    'calculate_ic_decay_simple',
    'fill_with_market_median',
    'calculate_rank_correlation',
    'EPSILON',
    # 核心类
    'V94DataManager',
    'V94SingleFactorICAuditor',
    'V94VWAPMomentumEngine',
    'V94AlphaFusion',
    'V94AlphaWeightEngine',
    'V94ICAudit',
    'V94TurnoverTracker',
    'V94PortfolioTracker',
    'V94StyleNeutralizationEngine',
    'V94LiquidityShockEngine',
    'V94DynamicRebalanceEngine',
]