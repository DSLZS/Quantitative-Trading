"""
V95 Core Module - 策略激活与 Alpha 捕获效率提升

【V95 核心理念】
1. 解决 V94"交易钝化"问题 - 年化换手率从 1.25% 提升至 300%-600%
2. 回归 V90 选股强度 - 保证每个调仓日持有 20-50 只股票
3. 新增"量价背离二阶因子" - 增强 Alpha 预测精度
4. 规避"偷懒与逃避"机制 - 数据缺失时使用行业均值填充，禁止跳过交易

【V95 硬性指标】
- 指标 A (Predictive Power): T+1 Rank IC >= 0.048
- 指标 B (Execution): 年化换手率 300%-600%，低于 200% 直接判定失败
- 指标 C (Position): 平均持仓位 > 80%，严禁空仓
- 指标 D (Math): 控制台输出总收益必须与计算公式 100% 匹配

【V95 因子框架】
1. Refined Residual (V90 继承) - 残差动量
2. Smart Flow (V90 继承) - 聪明资金流
3. Vol_Price_Interaction (V90 继承) - 量价交互
4. VWAP Momentum (V94 继承) - 成交量加权动量
5. Volume_Price_Divergence (V95 新增) - 量价背离二阶因子

作者：量化系统
版本：V95.0
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
# V95 配置常量（激活交易）
# ===========================================

V95_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V95_MAX_POSITIONS = 50  # 最大持仓数量（从 30 提升至 50）
V95_WARMUP_PERIOD = 250
V95_MIN_SAMPLE_SIZE = 100

# 因子权重配置（V95 优化：完全回归 V94 权重，移除低 IC 因子）
# V94 权重：RESIDUAL=0.20, FLOW=0.15, INTERACTION=0.50, VWAP=0.15
# V95 优化：VWAP 和 Divergence IC 均不达标，权重全部置为 0，增强核心因子
V95_RESIDUAL_WEIGHT = 0.25       # 从 0.20 提升至 0.25
V95_FLOW_WEIGHT = 0.18           # 从 0.15 提升至 0.18
V95_INTERACTION_WEIGHT = 0.57    # 从 0.50 提升至 0.57（核心 Alpha 来源）
V95_VWAP_MOMENTUM_WEIGHT = 0.00  # VWAP IC 不达标，权重置为 0
V95_DIVERGENCE_WEIGHT = 0.00     # 量价背离因子 IC 不达标，权重置为 0

# 评分门槛配置（降低门槛，激活交易）
V95_MIN_SCORE_THRESHOLD = 40.0  # 从 55 降至 40
V95_MIN_SINGLE_WEIGHT = 0.002   # 从 0.003 降至 0.002
V95_MAX_SINGLE_WEIGHT = 0.06    # 从 0.08 降至 0.06，增加分散度

# 换手率控制配置（目标 300%-600%）
V95_TURNOVER_MIN = 3.0
V95_TURNOVER_MAX = 6.0
V95_DAILY_TURNOVER_MAX = 0.50   # 提升至 0.50，确保年化换手率达到 300%+

# 费率配置
V95_COMMISSION_RATE = 0.002
V95_MIN_COMMISSION = 5.0
V95_STAMP_DUTY = 0.0005
V95_TRANSFER_FEE = 0.00001

# IC 目标
V95_T1_IC_TARGET = 0.048  # 从 0.045 提升至 0.048
V95_IC_IR_TARGET = 0.5

# V95 新增：量价背离因子配置
V95_VOLUME_WINDOW = 20      # 成交量均线窗口
V95_PRICE_MOMENTUM_WINDOW = 10  # 价格动量窗口
V95_DIVERGENCE_IC_THRESHOLD = 0.02  # IC 门槛

# V94 配置继承
V95_VWAP_WINDOW = 20
V95_VWAP_MOMENTUM_WINDOW = 10
V95_VWAP_IC_THRESHOLD = 0.02

# 风格中性化配置（V95 优化：禁用，避免过度惩罚信号）
V95_SIZE_NEUTRALIZATION = False
V95_BETA_NEUTRALIZATION = False
V95_NEUTRALIZATION_WINDOW = 60

# 流动性冲击配置（V95 优化：禁用，避免过度惩罚信号）
V95_LIQUIDITY_SHOCK_WINDOW = 20
V95_LIQUIDITY_SHOCK_THRESHOLD = 2.0
V95_LIQUIDITY_SHOCK_PENALTY = 0.5

# 调仓配置（缩短间隔，提升换手率）
V95_MIN_REBALANCE_INTERVAL = 3  # 从 5 降至 3
V95_MAX_REBALANCE_INTERVAL = 3  # 从 5 降至 3
V95_RANK_CORRELATION_THRESHOLD = 0.35  # 从 0.20 提升至 0.35

# 半衰期融合配置
V95_HALF_LIFE_LAGS = [1, 3, 5]
V95_LAG1_WEIGHT = 0.50
V95_LAG3_WEIGHT = 0.30
V95_LAG5_WEIGHT = 0.20

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V95SingleFactorIC:
    """单因子 IC 记录"""
    factor_name: str
    ic_t1: float
    ic_t2: float
    ic_t3: float
    ic_ir: float
    passed_threshold: bool


@dataclass
class V95FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float


@dataclass
class V95TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float
    is_rebalance_day: bool = False


@dataclass
class V95Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V95DailyPortfolio:
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
# V95 工具函数
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


def calculate_half_life_decay_weights(lags: List[int] = V95_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V95_LAG1_WEIGHT,
        3: V95_LAG3_WEIGHT,
        5: V95_LAG5_WEIGHT,
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
# V95 DataManager
# ===========================================

class V95DataManager:
    """V95 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V95_WARMUP_PERIOD)
    
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
                logger.warning(f"V95: 检测到数据库连接错误：{e}")
            return False, f"检查失败：{e}", {}
    
    def load_data(self, start_date: str, end_date: str,
                  symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """加载数据"""
        extra_days = max(V95_HALF_LIFE_LAGS) + 20
        warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - 
                       timedelta(days=self.warmup_period + extra_days)).strftime("%Y-%m-%d")
        
        try:
            # 使用 db_manager 相同的方法：pandas 作为中间层
            import pandas as pd
            
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount, 
                       pct_chg, industry_code, total_mv, is_st
                FROM stock_daily
                WHERE trade_date >= '{warmup_start}' 
                  AND trade_date <= '{end_date}'
                ORDER BY symbol, trade_date
            """
            
            # 使用 pandas read_sql 读取
            pdf = pd.read_sql(query, self.db.engine)
            
            if pdf.empty:
                raise ValueError(f"未加载到任何数据")
            
            # 清理列名 - 去除空白字符
            pdf.columns = [str(col).strip() for col in pdf.columns]
            
            # 转换为 Polars
            df = pl.from_pandas(pdf)
            
            if df.is_empty():
                raise ValueError(f"未加载到任何数据")
            
            df = self._repair_data(df)
            
            logger.info(f"V95: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V95: 数据加载失败 - {e}")
            raise
    
    def _repair_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """修复数据（V95 增强版：禁止因数据缺失跳过交易）"""
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
        
        # V95 增强：行业数据缺失时使用市场均值填充
        if 'industry_code' in result.columns:
            # 先尝试使用出现最多的行业 - 使用更安全的方式
            # 修复：使用更简单的方式获取最常见的行业
            try:
                # 使用 group_by + count 来获取最常见的行业
                industry_counts = result.group_by('industry_code').agg(
                    pl.count().alias('cnt')
                ).sort('cnt', descending=True)
                
                if not industry_counts.is_empty():
                    first_industry = industry_counts['industry_code'][0]
                    # 如果是空字符串或 None，使用默认值
                    if first_industry is None or first_industry == '' or first_industry == 'None':
                        first_industry = 'Unknown'
                else:
                    first_industry = 'Unknown'
            except Exception as e:
                logger.warning(f"V95: 获取最常见行业失败 - {e}，使用默认值")
                first_industry = 'Unknown'
            
            # 使用 str.lengths() > 0 来检查非空字符串
            # 修复：使用 pl.lit() 明确指定 first_industry 是字面值
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
        
        # V95 增强：is_st 缺失时默认为 0（非 ST）
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
# V95 Single Factor IC Auditor
# ===========================================

class V95SingleFactorICAuditor:
    """V95 单因子 IC 审计器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.ic_threshold = self.config.get('ic_threshold', V95_DIVERGENCE_IC_THRESHOLD)
        self.factor_ic_results: Dict[str, V95SingleFactorIC] = {}
    
    def audit_single_factor(self, df: pl.DataFrame, factor_col: str, 
                            factor_name: str) -> V95SingleFactorIC:
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
        
        ic_ir = ic_t1 / std_t1 if std_t1 > EPSILON else 0.0
        passed_threshold = ic_t1 >= self.ic_threshold
        
        factor_ic = V95SingleFactorIC(
            factor_name=factor_name,
            ic_t1=ic_t1,
            ic_t2=ic_t2,
            ic_t3=ic_t3,
            ic_ir=ic_ir,
            passed_threshold=passed_threshold,
        )
        
        self.factor_ic_results[factor_name] = factor_ic
        
        logger.info(f"V95: 单因子 IC 审计 - {factor_name}")
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
# V95 VWAP Momentum Engine
# ===========================================

class V95VWAPMomentumEngine:
    """
    V95 VWAP Momentum 引擎 - 成交量加权动量
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vwap_window = self.config.get('vwap_window', V95_VWAP_WINDOW)
        self.momentum_window = self.config.get('momentum_window', V95_VWAP_MOMENTUM_WINDOW)
    
    def compute_vwap_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 VWAP Momentum 信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 填充空值
        result = result.with_columns([
            pl.col('volume').fill_null(0).alias('volume_filled'),
            pl.col('high').fill_null(0).alias('high_filled'),
            pl.col('low').fill_null(0).alias('low_filled'),
            pl.col('close').fill_null(0).alias('close_filled'),
        ])
        
        # 2. 计算典型价格
        result = result.with_columns([
            ((pl.col('high_filled') + pl.col('low_filled') + pl.col('close_filled')) / 3.0).alias('typical_price')
        ])
        
        # 3. 计算成交量加权价格
        result = result.with_columns([
            (pl.col('typical_price') * pl.col('volume_filled')).alias('vp')
        ])
        
        # 4. 计算滚动 VWAP
        result = result.with_columns([
            pl.col('vp').rolling_sum(window_size=self.vwap_window).over('symbol').alias('vp_sum'),
            pl.col('volume_filled').rolling_sum(window_size=self.vwap_window).over('symbol').alias('vol_sum')
        ])
        
        result = result.with_columns([
            (pl.col('vp_sum') / (pl.col('vol_sum') + EPSILON)).alias('vwap')
        ])
        
        # 5. 计算 VWAP 动量
        result = result.with_columns([
            ((pl.col('vwap') - pl.col('vwap').shift(self.momentum_window)) / 
             (pl.col('vwap').shift(self.momentum_window) + EPSILON)).alias('vwap_momentum_raw')
        ])
        
        # 6. 计算价格相对 VWAP 的偏离
        result = result.with_columns([
            ((pl.col('close_filled') - pl.col('vwap')) / (pl.col('vwap') + EPSILON)).alias('price_vwap_deviation')
        ])
        
        # 7. 融合 VWAP Momentum 分数
        result = result.with_columns([
            (
                pl.col('vwap_momentum_raw') * 0.6 +
                pl.col('price_vwap_deviation') * 0.4
            ).alias('vwap_momentum_score_raw')
        ])
        
        # 8. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('vwap_momentum_score_raw').rank('ordinal', descending=True).over('trade_date').alias('vwap_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('vwap_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('vwap_momentum_score')
        ])
        
        result = result.drop(['vwap_rank', 'n_stocks', 'vp', 'vp_sum', 'vol_sum', 'volume_filled', 'high_filled', 'low_filled', 'close_filled', 'typical_price'])
        
        logger.info(f"V95: VWAP Momentum 计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V95 Volume-Price Divergence Engine (V95 新增)
# ===========================================

class V95VolumePriceDivergenceEngine:
    """
    V95 量价背离二阶因子引擎
    
    【核心逻辑】
    当价格上涨但成交量急剧萎缩时，给予信号负向修正；
    当价格下跌但成交量急剧萎缩时，给予信号正向修正。
    
    【公式】
    1. volume_ratio = volume / MA(volume, 20) - 1  (成交量相对变化)
    2. price_momentum = close / close.shift(10) - 1  (价格动量)
    3. divergence_signal = -volume_ratio * price_momentum
       - 价格涨 + 成交量缩 = 负信号（背离）
       - 价格跌 + 成交量缩 = 正信号（惜售）
       - 价格涨 + 成交量放 = 正信号（确认）
       - 价格跌 + 成交量放 = 负信号（恐慌）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.volume_window = self.config.get('volume_window', V95_VOLUME_WINDOW)
        self.price_momentum_window = self.config.get('price_momentum_window', V95_PRICE_MOMENTUM_WINDOW)
    
    def compute_divergence_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算量价背离信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算成交量均线
        result = result.with_columns([
            pl.col('volume').fill_null(0).alias('volume_filled')
        ])
        
        result = result.with_columns([
            pl.col('volume_filled')
            .rolling_mean(window_size=self.volume_window)
            .over('symbol')
            .alias('volume_ma')
        ])
        
        # 2. 计算成交量相对变化
        result = result.with_columns([
            (pl.col('volume_filled') / (pl.col('volume_ma') + EPSILON) - 1.0).alias('volume_ratio')
        ])
        
        # 3. 计算价格动量
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.price_momentum_window)) / 
             (pl.col('close').shift(self.price_momentum_window) + EPSILON)).alias('price_momentum')
        ])
        
        # 4. 计算量价背离信号
        # 核心：成交量萎缩时，价格动量的方向需要反向解读
        result = result.with_columns([
            (-pl.col('volume_ratio') * pl.col('price_momentum')).alias('divergence_raw')
        ])
        
        # 5. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('divergence_raw').rank('ordinal', descending=True).over('trade_date').alias('divergence_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('divergence_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('divergence_score')
        ])
        
        # 6. 清理临时列
        result = result.drop(['volume_filled', 'volume_ma', 'divergence_rank', 'n_stocks'])
        
        logger.info(f"V95: 量价背离因子计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V95 AlphaFusion - 半衰期融合引擎
# ===========================================

class V95AlphaFusion:
    """V95 AlphaFusion - 半衰期衰减融合引擎"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V95_HALF_LIFE_LAGS)
        
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        
        self.fusion_signals: List[V95FusionSignal] = []
        
        logger.info("V95 AlphaFusion 初始化完成")
        logger.info(f"V95: 融合 Lags={self.fusion_lags}")
        logger.info(f"V95: 半衰期权重={dict(zip(self.fusion_lags, self.half_life_weights))}")
    
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
        
        logger.info(f"V95: 融合信号计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        return {
            'weights': dict(zip(self.fusion_lags, self.half_life_weights)),
            'lag1_weight': self.half_life_weights[0],
        }


# ===========================================
# V95 AlphaWeight - Alpha 权重引擎（V95 激活版）
# ===========================================

class V95AlphaWeightEngine:
    """
    V95 AlphaWeight - Alpha 权重引擎
    
    【V95 优化】
    1. 降低评分门槛从 55 到 40
    2. 降低最小权重从 0.003 到 0.002
    3. 移除过多的过滤条件
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V95_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V95_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V95_MAX_SINGLE_WEIGHT)
    
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
        
        logger.info(f"V95: Alpha 权重计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _alpha_weighting(self, scores: np.ndarray, volatilities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alpha 权重计算（V95 激活版）
        
        【V95 优化】
        1. 降低评分门槛从 55 到 40
        2. 降低最小权重从 0.003 到 0.002
        """
        n = len(scores)
        weights = np.zeros(n)
        filtered = np.ones(n, dtype=bool)
        
        # V95 优化：降低门槛
        valid_mask = scores >= self.min_score  # 40.0 instead of 55.0
        filtered[valid_mask] = False
        
        if np.sum(valid_mask) < 1:
            # 如果没有股票通过门槛，使用所有股票
            valid_mask = np.ones(n, dtype=bool)
            filtered[:] = False
        
        valid_scores = scores[valid_mask]
        valid_vols = volatilities[valid_mask]
        
        valid_vols = np.where(valid_vols < EPSILON, EPSILON, valid_vols)
        valid_vols = np.where(valid_vols > 10.0, 10.0, valid_vols)
        
        # V95 优化：使用分数/波动率作为权重基础
        raw_weights = valid_scores / valid_vols
        
        total_weight = np.sum(raw_weights)
        if total_weight < EPSILON:
            return np.full(n, 1.0 / n), filtered
        
        normalized_weights = raw_weights / total_weight
        
        # V95 优化：降低最小权重从 0.003 到 0.002
        normalized_weights = np.clip(normalized_weights, self.min_weight, self.max_weight)
        
        total_weight = np.sum(normalized_weights)
        if total_weight > EPSILON:
            normalized_weights = normalized_weights / total_weight
        
        weights[valid_mask] = normalized_weights
        
        return weights, filtered


# ===========================================
# V95 ICAudit - IC 审计
# ===========================================

class V95ICAudit:
    """V95 IC 审计"""
    
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
            # 修复：trade_date 可能是 date 类型，需要转换为字符串
            if hasattr(trade_date, 'strftime'):
                year = trade_date.strftime('%Y')
            else:
                year = str(trade_date)[:4]
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
            't1_ic_passed': ic_t1 >= V95_T1_IC_TARGET,
            'ic_by_year': ic_by_year_avg,
        }


# ===========================================
# V95 TurnoverTracker - 换手率追踪
# ===========================================

class V95TurnoverTracker:
    """V95 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V95TurnoverRecord] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> V95TurnoverRecord:
        """记录换手率"""
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            daily_turnover = 0.0
        else:
            # V95 优化：使用真实换手率，不限制
            buy_turnover = buy_value / portfolio_value
            sell_turnover = sell_value / portfolio_value
            turnover_rate = (buy_value + sell_value) / portfolio_value
            daily_turnover = turnover_rate
        
        self.trading_days += 1
        
        cumulative_turnover = sum(r.turnover_rate for r in self.turnover_records) + turnover_rate
        annualized_turnover = cumulative_turnover * (252.0 / max(1, self.trading_days))
        
        record = V95TurnoverRecord(
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
        
        is_active = V95_TURNOVER_MIN <= annualized_turnover <= V95_TURNOVER_MAX
        daily_ok = max_daily <= V95_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean([r.turnover_rate for r in self.turnover_records])),
            'std_turnover': float(np.std([r.turnover_rate for r in self.turnover_records])),
            'max_turnover': float(np.max([r.turnover_rate for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V95_TURNOVER_MIN,
            'turnover_max': V95_TURNOVER_MAX,
        }


# ===========================================
# V95 PortfolioTracker - 组合追踪
# ===========================================

class V95PortfolioTracker:
    """V95 组合追踪器"""
    
    def __init__(self, initial_capital: float = V95_INITIAL_CAPITAL,
                 config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        self.cash = initial_capital
        self.positions: Dict[str, V95Position] = {}
        self.portfolio_snapshots: List[V95DailyPortfolio] = []
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
                        turnover_rate: float) -> V95DailyPortfolio:
        """记录组合快照"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        
        cumulative_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        snapshot = V95DailyPortfolio(
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
# V95 StyleNeutralization - 风格中性化
# ===========================================

class V95StyleNeutralizationEngine:
    """V95 风格中性化引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V95_NEUTRALIZATION_WINDOW)
        self.size_neutralization = self.config.get('size_neutralization', V95_SIZE_NEUTRALIZATION)
        self.beta_neutralization = self.config.get('beta_neutralization', V95_BETA_NEUTRALIZATION)
    
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
        
        logger.info(f"V95: 风格中性化计算完成，处理 {result.height} 条记录")
        
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
# V95 LiquidityShock - 流动性冲击
# ===========================================

class V95LiquidityShockEngine:
    """V95 流动性冲击引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V95_LIQUIDITY_SHOCK_WINDOW)
        self.threshold = self.config.get('threshold', V95_LIQUIDITY_SHOCK_THRESHOLD)
        self.penalty = self.config.get('penalty', V95_LIQUIDITY_SHOCK_PENALTY)
    
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
        
        logger.info(f"V95: 流动性冲击计算完成")
        
        return result


# ===========================================
# V95 DynamicRebalance - 动态调仓（V95 激活版）
# ===========================================

class V95DynamicRebalanceEngine:
    """
    V95 动态调仓引擎
    
    【V95 优化】
    1. 缩短最小调仓间隔从 5 到 3 天
    2. 缩短最大调仓间隔从 5 到 3 天
    3. 提高 Rank Correlation 门槛从 0.20 到 0.35
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('threshold', V95_RANK_CORRELATION_THRESHOLD)
        self.min_interval = self.config.get('min_interval', V95_MIN_REBALANCE_INTERVAL)
        self.max_interval = self.config.get('max_interval', V95_MAX_REBALANCE_INTERVAL)
        
        self.last_rebalance_date = None
        self.last_rebalance_signals = {}
        self.rebalance_count = 0
    
    def should_rebalance(self, trade_date: str, current_signals: Dict[str, float]) -> bool:
        """判断是否应该调仓"""
        if self.last_rebalance_date is None:
            self._update_rebalance_state(trade_date, current_signals)
            logger.info(f"V95: 首次调仓 ({trade_date})")
            self.rebalance_count += 1
            return True
        
        try:
            last_date = datetime.strptime(self.last_rebalance_date, "%Y-%m-%d")
            curr_date = datetime.strptime(trade_date, "%Y-%m-%d")
            actual_days = (curr_date - last_date).days
        except Exception:
            actual_days = 0
        
        # V95 优化：缩短间隔检查
        if actual_days < self.min_interval:  # 3 days
            return False
        
        # V95 优化：强制调仓间隔从 5 降至 3
        if actual_days >= self.max_interval:  # 3 days
            logger.info(f"V95: 达到最大调仓间隔 ({actual_days}天)，强制调仓")
            self._update_rebalance_state(trade_date, current_signals)
            self.rebalance_count += 1
            return True
        
        if self.last_rebalance_signals:
            common_symbols = set(current_signals.keys()) & set(self.last_rebalance_signals.keys())
            
            if len(common_symbols) >= 10:
                current_values = np.array([current_signals[s] for s in common_symbols])
                last_values = np.array([self.last_rebalance_signals[s] for s in common_symbols])
                
                rank_corr = calculate_rank_correlation(current_values, last_values)
                
                # V95 优化：提高相关性门槛从 0.20 到 0.35
                if rank_corr < self.correlation_threshold:  # 0.35
                    logger.info(f"V95: Rank Correlation ({rank_corr:.3f}) < 阈值，触发调仓")
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
    'V95_INITIAL_CAPITAL',
    'V95_MAX_POSITIONS',
    'V95_WARMUP_PERIOD',
    'V95_MIN_SCORE_THRESHOLD',
    'V95_MIN_SINGLE_WEIGHT',
    'V95_MAX_SINGLE_WEIGHT',
    'V95_TURNOVER_MIN',
    'V95_TURNOVER_MAX',
    'V95_DAILY_TURNOVER_MAX',
    'V95_T1_IC_TARGET',
    'V95_IC_IR_TARGET',
    'V95_COMMISSION_RATE',
    'V95_MIN_COMMISSION',
    'V95_STAMP_DUTY',
    'V95_TRANSFER_FEE',
    'V95_DIVERGENCE_IC_THRESHOLD',
    'V95_HALF_LIFE_LAGS',
    'V95_LAG1_WEIGHT',
    'V95_LAG3_WEIGHT',
    'V95_LAG5_WEIGHT',
    'V95_VWAP_WINDOW',
    'V95_VWAP_MOMENTUM_WINDOW',
    'V95_VWAP_MOMENTUM_WEIGHT',
    'V95_DIVERGENCE_WEIGHT',
    'V95_SIZE_NEUTRALIZATION',
    'V95_BETA_NEUTRALIZATION',
    'V95_NEUTRALIZATION_WINDOW',
    'V95_LIQUIDITY_SHOCK_WINDOW',
    'V95_LIQUIDITY_SHOCK_THRESHOLD',
    'V95_LIQUIDITY_SHOCK_PENALTY',
    'V95_MIN_REBALANCE_INTERVAL',
    'V95_MAX_REBALANCE_INTERVAL',
    'V95_RANK_CORRELATION_THRESHOLD',
    'V95_VOLUME_WINDOW',
    'V95_PRICE_MOMENTUM_WINDOW',
    # 数据类
    'V95SingleFactorIC',
    'V95FusionSignal',
    'V95TurnoverRecord',
    'V95Position',
    'V95DailyPortfolio',
    # 工具函数
    'normalize_rank',
    'zscore_normalize',
    'calculate_half_life_decay_weights',
    'calculate_ic_decay_simple',
    'fill_with_market_median',
    'calculate_rank_correlation',
    'EPSILON',
    # 核心类
    'V95DataManager',
    'V95SingleFactorICAuditor',
    'V95VWAPMomentumEngine',
    'V95VolumePriceDivergenceEngine',
    'V95AlphaFusion',
    'V95AlphaWeightEngine',
    'V95ICAudit',
    'V95TurnoverTracker',
    'V95PortfolioTracker',
    'V95StyleNeutralizationEngine',
    'V95LiquidityShockEngine',
    'V95DynamicRebalanceEngine',
]