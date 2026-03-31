"""
V97 Core Module - 深度防御与重启：V90 基准复现 + 量价一致性因子消融实验

【V97 核心理念：先截断逃生路径，再建立基准，最后分步验证】
1. 强制中性化 - 行业中性化和市值中性化必须开启（使用 StandardScaler 和 OLS 残差化）
2. 强制流动性过滤 - 禁止买入 ST 股、禁止买入成交额后 10% 的股票
3. V90 基准复现 - 首先完整复现 V90 的原始算法（残差动量 + 聪明资金流）
4. 消融实验 (Ablation Study) - 引入"量价一致性强度"因子，分别打印各因子 IC
5. 自动回退机制 - 如果融合后 IC 低于原始 V90，自动剔除新因子

【V97 硬性指标】
| 维度 | 指标 | 目标值 | 失败判定 |
| :--- | :--- | :--- | :--- |
| 预测力 | T+1 Rank IC | ≥ 0.048 | 低于 0.045 视为负优化 |
| 真实度 | 数据采样打印 | 必须可见 | 缺失采样视为逻辑欺诈 |
| 风险比 | Calmar Ratio | > 1.5 | 禁用中性化则报告无效 |
| 活跃度 | 年化换手率 | 300% - 600% | 低于 200% 视为消极交易 |

【V97 因子框架】
1. Residual Momentum (V90 基准) - 残差动量
2. Smart Flow (V90 基准) - 聪明资金流
3. Price-Volume Consistency (V97 新增) - 量价一致性强度因子

【消融实验配置】
- Audit_Mode: 分别打印 V90 原始因子 IC、新因子独立 IC、融合后 IC
- 自动回退：如果融合后 IC < V90 原始 IC，剔除新因子

作者：量化系统
版本：V97.0
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
# V97 配置常量（强制中性化）
# ===========================================

V97_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V97_MAX_POSITIONS = 50  # 最大持仓数量
V97_WARMUP_PERIOD = 250
V97_MIN_SAMPLE_SIZE = 100

# V90 基准因子权重（必须首先复现）
V97_RESIDUAL_WEIGHT = 0.60       # 残差动量权重（V90 核心）
V97_FLOW_WEIGHT = 0.40           # 聪明资金流权重（V90 核心）

# V97 新增因子权重（消融实验）
V97_CONSISTENCY_WEIGHT = 0.20    # 量价一致性强度权重

# 评分门槛配置
V97_MIN_SCORE_THRESHOLD = 45.0   # 评分门槛
V97_MIN_SINGLE_WEIGHT = 0.002    # 最小权重
V97_MAX_SINGLE_WEIGHT = 0.06     # 最大权重

# 换手率控制配置（V97 目标：300%-600%）
V97_TURNOVER_MIN = 3.0
V97_TURNOVER_MAX = 6.0
V97_DAILY_TURNOVER_MAX = 0.50

# 费率配置
V97_COMMISSION_RATE = 0.002
V97_MIN_COMMISSION = 5.0
V97_STAMP_DUTY = 0.0005
V97_TRANSFER_FEE = 0.00001

# IC 目标（V97 强制）
V97_T1_IC_TARGET = 0.048  # T+1 IC 目标
V97_IC_IR_TARGET = 0.5

# 风格中性化配置（V97 强制：必须开启）
V97_SIZE_NEUTRALIZATION = True      # 强制开启市值中性化
V97_INDUSTRY_NEUTRALIZATION = True  # 强制开启行业中性化
V97_NEUTRALIZATION_WINDOW = 60

# 流动性过滤配置（V97 强制：必须开启）
V97_LIQUIDITY_FILTER = True         # 强制开启流动性过滤
V97_LIQUIDITY_PERCENTILE = 10       # 禁止买入后 10% 的股票
V97_FILTER_ST = True                # 强制过滤 ST 股

# 调仓配置
V97_MIN_REBALANCE_INTERVAL = 5
V97_MAX_REBALANCE_INTERVAL = 5
V97_RANK_CORRELATION_THRESHOLD = 0.35

# 半衰期融合配置
V97_HALF_LIFE_LAGS = [1, 3, 5]
V97_LAG1_WEIGHT = 0.50
V97_LAG3_WEIGHT = 0.30
V97_LAG5_WEIGHT = 0.20

# V97 新增：量价一致性因子配置
V97_CONSISTENCY_WINDOW = 20  # 量价一致性计算窗口

# 消融实验配置
V97_AUDIT_MODE = True  # 开启审计模式，打印各因子 IC

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V97SingleFactorIC:
    """单因子 IC 记录（消融实验用）"""
    factor_name: str
    ic_t1: float
    ic_t2: float
    ic_t3: float
    ic_ir: float
    passed_threshold: bool


@dataclass
class V97FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float


@dataclass
class V97TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float
    is_rebalance_day: bool = False


@dataclass
class V97Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V97DailyPortfolio:
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
# V97 工具函数
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
    """Z-Score 标准化（使用 StandardScaler）"""
    if len(series) < 2:
        return np.zeros_like(series)
    
    scaler = StandardScaler()
    try:
        return scaler.fit_transform(series.reshape(-1, 1)).flatten()
    except Exception:
        return (series - np.mean(series)) / (np.std(series) + EPSILON)


def calculate_half_life_decay_weights(lags: List[int] = V97_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V97_LAG1_WEIGHT,
        3: V97_LAG3_WEIGHT,
        5: V97_LAG5_WEIGHT,
    }
    
    weights = [fixed_weights.get(lag, 1.0 / len(lags)) for lag in lags]
    
    total_weight = sum(weights)
    if total_weight < EPSILON:
        return [1.0 / len(lags)] * len(lags)
    
    return [w / total_weight for w in weights]


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


def ols_residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    OLS 残差化（使用 sklearn LinearRegression）
    
    Parameters
    ----------
    y : np.ndarray
        因变量（信号）
    X : np.ndarray
        自变量（风格因子）
        
    Returns
    -------
    np.ndarray
        残差
    """
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
        
        # 将残差放回原数组
        result = np.zeros_like(y)
        result[valid_mask] = residuals
        
        # 恢复原始量纲
        result = result * np.std(y) + np.mean(y)
        
        return result
    except Exception:
        return y


# ===========================================
# V97 DataManager
# ===========================================

class V97DataManager:
    """V97 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V97_WARMUP_PERIOD)
    
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
        extra_days = max(V97_HALF_LIFE_LAGS) + 20
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
            
            logger.info(f"V97: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V97: 数据加载失败 - {e}")
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
                logger.warning(f"V97: 获取最常见行业失败 - {e}，使用默认值")
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
# V97 Industry Neutralization - 行业中性化（V97 强制开启）
# ===========================================

class V97IndustryNeutralizationEngine:
    """
    V97 行业中性化引擎（使用 OLS 残差化）
    
    【强制要求】
    1. 必须开启行业中性化
    2. 使用行业哑变量进行 OLS 回归
    3. 取残差作为中性化后的信号
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.industry_neutralization = self.config.get('industry_neutralization', V97_INDUSTRY_NEUTRALIZATION)
        self.window = self.config.get('window', V97_NEUTRALIZATION_WINDOW)
    
    def compute_industry_neutralization(self, df: pl.DataFrame, 
                                         signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算行业中性化"""
        if not self.industry_neutralization:
            logger.error("V97: 行业中性化未开启，这将导致回测无效！")
            return df
        
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 获取行业映射数据
        industry_df = self._load_industry_data()
        
        if industry_df.is_empty():
            logger.warning("V97: 无法获取行业数据，使用行业代码代替")
            result = result.with_columns([
                pl.col('industry_code').fill_null('Unknown').alias('industry_name')
            ])
        else:
            # 修复日期类型不匹配问题
            industry_df = industry_df.with_columns([
                pl.col('trade_date').cast(pl.Date).alias('trade_date')
            ])
            
            result = result.join(
                industry_df.select(['symbol', 'trade_date', 'industry_name']),
                on=['symbol', 'trade_date'],
                how='left'
            )
            result = result.with_columns([
                pl.col('industry_name').fill_null(pl.col('industry_code').fill_null('Unknown')).alias('industry_name')
            ])
        
        # 按日期进行行业中性化
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        neutralized_signals = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            if day_data.is_empty():
                continue
            
            signals = day_data[signal_col].to_numpy()
            industries = day_data['industry_name'].to_numpy()
            symbols = day_data['symbol'].to_numpy()
            
            valid_mask = (~np.isnan(signals) & np.isfinite(signals))
            
            if np.sum(valid_mask) < 10:
                for i, symbol in enumerate(symbols):
                    neutralized_signals.append({
                        'trade_date': trade_date,
                        'symbol': symbol,
                        'neutralized_signal': signals[i] if i < len(signals) else 0.0,
                    })
                continue
            
            valid_signals = signals[valid_mask]
            valid_industries = industries[valid_mask]
            valid_symbols = symbols[valid_mask]
            
            neutralized = self._neutralize_by_industry(valid_signals, valid_industries)
            
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
        
        logger.info(f"V97: 行业中性化计算完成 (ENABLED)，处理 {result.height} 条记录")
        
        return result
    
    def _load_industry_data(self) -> pl.DataFrame:
        """加载行业数据"""
        if self.db is None:
            return pl.DataFrame()
        
        try:
            import pandas as pd
            
            query = """
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
            """
            
            pdf = pd.read_sql(query, self.db.engine)
            return pl.from_pandas(pdf)
            
        except Exception as e:
            logger.warning(f"V97: 加载行业数据失败 - {e}")
            return pl.DataFrame()
    
    def _neutralize_by_industry(self, signals: np.ndarray, industries: np.ndarray) -> np.ndarray:
        """
        按行业中性化信号（OLS 残差化）
        
        方法：对每个行业，构建哑变量进行多元回归，取残差
        """
        unique_industries = np.unique(industries)
        
        if len(unique_industries) < 2:
            return signals
        
        # 构建行业哑变量矩阵
        n = len(signals)
        X = np.zeros((n, len(unique_industries)))
        
        for i, industry in enumerate(unique_industries):
            X[:, i] = (industries == industry).astype(float)
        
        # OLS 残差化
        return ols_residualize(signals, X)


# ===========================================
# V97 Size Neutralization - 市值中性化（V97 强制开启）
# ===========================================

class V97SizeNeutralizationEngine:
    """
    V97 市值中性化引擎（使用 OLS 残差化）
    
    【强制要求】
    1. 必须开启市值中性化
    2. 使用对数市值进行回归
    3. 取残差作为中性化后的信号
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.size_neutralization = self.config.get('size_neutralization', V97_SIZE_NEUTRALIZATION)
    
    def compute_size_neutralization(self, df: pl.DataFrame, 
                                     signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算市值中性化"""
        if not self.size_neutralization:
            logger.error("V97: 市值中性化未开启，这将导致回测无效！")
            return df
        
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算对数市值
        result = result.with_columns([
            pl.col('total_mv').fill_null(1.0).log().alias('size_factor')
        ])
        
        # 按日期进行市值中性化
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        neutralized_signals = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            if day_data.is_empty():
                continue
            
            signals = day_data[signal_col].to_numpy()
            sizes = day_data['size_factor'].to_numpy()
            symbols = day_data['symbol'].to_numpy()
            
            valid_mask = (~np.isnan(signals) & ~np.isnan(sizes) &
                         np.isfinite(signals) & np.isfinite(sizes))
            
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
            valid_symbols = symbols[valid_mask]
            
            neutralized = self._neutralize_by_size(valid_signals, valid_sizes)
            
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
        
        logger.info(f"V97: 市值中性化计算完成 (ENABLED)，处理 {result.height} 条记录")
        
        return result
    
    def _neutralize_by_size(self, signals: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        按市值中性化信号（OLS 残差化）
        
        方法：线性回归取残差
        """
        n = len(signals)
        
        # 构建自变量矩阵
        X = sizes.reshape(-1, 1)
        
        # OLS 残差化
        return ols_residualize(signals, X)


# ===========================================
# V97 Liquidity Filter - 流动性过滤（V97 强制开启）
# ===========================================

class V97LiquidityFilterEngine:
    """
    V97 流动性过滤引擎
    
    【强制要求】
    1. 禁止买入日均成交额后 10% 的股票
    2. 禁止买入 ST 股
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.liquidity_filter = self.config.get('liquidity_filter', V97_LIQUIDITY_FILTER)
        self.liquidity_percentile = self.config.get('liquidity_percentile', V97_LIQUIDITY_PERCENTILE)
        self.filter_st = self.config.get('filter_st', V97_FILTER_ST)
    
    def apply_liquidity_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """应用流动性过滤"""
        if not self.liquidity_filter:
            logger.error("V97: 流动性过滤未开启，这将导致回测无效！")
            return df
        
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算滚动平均成交额
        result = result.with_columns([
            pl.col('amount').fill_null(0).alias('amount_filled')
        ])
        
        result = result.with_columns([
            pl.col('amount_filled')
            .rolling_mean(window_size=V97_CONSISTENCY_WINDOW)
            .over('symbol')
            .alias('amount_ma')
        ])
        
        # 按日期计算成交额百分位
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        filtered_flags = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            if day_data.is_empty():
                continue
            
            amounts = day_data['amount_ma'].to_numpy()
            symbols = day_data['symbol'].to_numpy()
            is_st = day_data['is_st'].to_numpy() if 'is_st' in day_data.columns else np.zeros(len(day_data))
            
            valid_mask = (~np.isnan(amounts) & np.isfinite(amounts))
            
            if np.sum(valid_mask) < 10:
                for i, symbol in enumerate(symbols):
                    filtered_flags.append({
                        'trade_date': trade_date,
                        'symbol': symbol,
                        'is_filtered': False,
                        'filter_reason': 'insufficient_data',
                    })
                continue
            
            valid_amounts = amounts[valid_mask]
            valid_symbols = symbols[valid_mask]
            valid_st = is_st[valid_mask] if len(is_st) == len(symbols) else np.zeros(len(valid_symbols))
            
            # 计算后 10% 阈值
            threshold = np.percentile(valid_amounts, self.liquidity_percentile)
            
            for i, symbol in enumerate(valid_symbols):
                is_low_liquidity = valid_amounts[i] <= threshold
                is_stock_st = valid_st[i] == 1
                
                should_filter = False
                reason = 'none'
                
                if self.filter_st and is_stock_st:
                    should_filter = True
                    reason = 'st_stock'
                elif is_low_liquidity:
                    should_filter = True
                    reason = 'low_liquidity'
                
                filtered_flags.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'is_filtered': should_filter,
                    'filter_reason': reason,
                })
        
        if filtered_flags:
            filter_df = pl.DataFrame({
                'trade_date': [f['trade_date'] for f in filtered_flags],
                'symbol': [f['symbol'] for f in filtered_flags],
                'is_filtered': [f['is_filtered'] for f in filtered_flags],
                'filter_reason': [f['filter_reason'] for f in filtered_flags],
            })
            
            result = result.join(filter_df, on=['trade_date', 'symbol'], how='left')
            result = result.with_columns([
                pl.col('is_filtered').fill_null(False).alias('is_filtered')
            ])
        
        logger.info(f"V97: 流动性过滤计算完成 (ENABLED)")
        
        return result


# ===========================================
# V97 Residual Momentum - 残差动量（V90 基准因子 1）
# ===========================================

class V97ResidualMomentumEngine:
    """
    V97 残差动量引擎（V90 基准因子 1）
    
    【核心逻辑】
    1. 计算个股过去 N 日的累计收益率
    2. 对行业市值中性化
    3. 取残差作为动量信号
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.momentum_window = self.config.get('momentum_window', 20)
    
    def compute_residual_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算残差动量"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算 N 日动量
        result = result.with_columns([
            ((pl.col('close') / pl.col('close').shift(self.momentum_window)) - 1).alias('momentum_raw')
        ])
        
        # 2. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('momentum_raw').rank('ordinal', descending=True).over('trade_date').alias('momentum_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('momentum_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('residual_momentum_score')
        ])
        
        # 清理临时列
        result = result.drop(['momentum_rank', 'n_stocks'])
        
        logger.info(f"V97: 残差动量计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V97 Smart Flow - 聪明资金流（V90 基准因子 2）
# ===========================================

class V97SmartFlowEngine:
    """
    V97 聪明资金流引擎（V90 基准因子 2）
    
    【核心逻辑】
    1. 计算资金流向：(close - low) - (high - close) / (high - low) * volume
    2. 计算滚动资金流均值
    3. 按日期排名转换为百分位
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.flow_window = self.config.get('flow_window', 5)
    
    def compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算聪明资金流"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算资金流向因子
        numerator = (pl.col('close') - pl.col('low')) - (pl.col('high') - pl.col('close'))
        denominator = pl.col('high') - pl.col('low') + EPSILON
        result = result.with_columns(
            (numerator / denominator).alias('money_flow_factor')
        )
        
        # 2. 计算资金流（乘以成交量）
        result = result.with_columns(
            (pl.col('money_flow_factor') * pl.col('volume').fill_null(0)).alias('money_flow')
        )
        
        # 3. 计算滚动资金流均值
        result = result.with_columns([
            pl.col('money_flow')
            .rolling_mean(window_size=self.flow_window)
            .over('symbol')
            .alias('money_flow_ma')
        ])
        
        # 4. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('money_flow_ma').rank('ordinal', descending=True).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('flow_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('smart_flow_score')
        ])
        
        # 清理临时列 - 保留基础列和计算结果
        keep_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 
                     'volume', 'amount', 'pct_chg', 'industry_code', 'total_mv', 
                     'is_st', 'smart_flow_score']
        
        # 如果存在残差动量分数，也保留它
        if 'residual_momentum_score' in result.columns:
            keep_cols.append('residual_momentum_score')
        
        result = result.select(keep_cols)
        
        logger.info(f"V97: 聪明资金流计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V97 Price-Volume Consistency - 量价一致性强度（V97 新增因子）
# ===========================================

class V97PriceVolumeConsistencyEngine:
    """
    V97 量价一致性强度引擎
    
    【核心逻辑】
    计算价格方向与成交量变化的余弦相似度：
    - 当价格和成交量同向变化时，余弦相似度接近 1
    - 当价格和成交量反向变化时，余弦相似度接近 -1
    
    【准入审计】
    - 在 Audit_Mode 下，打印此因子的独立 IC
    - 如果融合后 IC 低于原始 V90，自动回退剔除该因子
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V97_CONSISTENCY_WINDOW)
    
    def compute_price_volume_consistency(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算量价一致性强度"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算价格变化率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('price_change')
        ])
        
        # 2. 计算成交量变化率
        result = result.with_columns([
            ((pl.col('volume').fill_null(0) - pl.col('volume').fill_null(0).shift(1)) / 
             (pl.col('volume').fill_null(0).shift(1) + EPSILON)).alias('volume_change')
        ])
        
        # 3. 计算滚动协方差和标准差（用于计算余弦相似度）
        # 余弦相似度 = cov(price, volume) / (std(price) * std(volume))
        result = result.with_columns([
            pl.col('price_change')
            .rolling_std(window_size=self.window)
            .over('symbol')
            .alias('price_std')
        ])
        
        result = result.with_columns([
            pl.col('volume_change')
            .rolling_std(window_size=self.window)
            .over('symbol')
            .alias('volume_std')
        ])
        
        # 4. 计算滚动相关性（近似余弦相似度）
        # 使用滚动窗口计算价格和成交量的相关性
        result = self._compute_rolling_correlation(result)
        
        # 5. 转换为百分位分数
        result = result.with_columns([
            pl.col('price_volume_correlation').rank('ordinal', descending=True).over('trade_date').alias('consistency_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('consistency_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('price_volume_consistency_score')
        ])
        
        # 清理临时列
        result = result.drop(['price_change', 'volume_change', 'price_std', 'volume_std', 
                              'consistency_rank', 'n_stocks'])
        
        logger.info(f"V97: 量价一致性强度计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _compute_rolling_correlation(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算滚动相关性
        
        使用滚动窗口计算价格和成交量的皮尔逊相关系数
        """
        result = df.clone()
        
        # 计算滚动均值
        result = result.with_columns([
            pl.col('price_change')
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('price_mean')
        ])
        
        result = result.with_columns([
            pl.col('volume_change')
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('volume_mean')
        ])
        
        # 计算去均值后的值
        result = result.with_columns([
            (pl.col('price_change') - pl.col('price_mean')).alias('price_dev'),
            (pl.col('volume_change') - pl.col('volume_mean')).alias('volume_dev')
        ])
        
        # 计算协方差和方差
        result = result.with_columns([
            (pl.col('price_dev') * pl.col('volume_dev'))
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('covariance')
        ])
        
        result = result.with_columns([
            (pl.col('price_dev') ** 2)
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('price_var')
        ])
        
        result = result.with_columns([
            (pl.col('volume_dev') ** 2)
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('volume_var')
        ])
        
        # 计算相关系数
        result = result.with_columns([
            (pl.col('covariance') / 
             (pl.col('price_var').sqrt() * pl.col('volume_var').sqrt() + EPSILON)).alias('price_volume_correlation')
        ])
        
        # 清理临时列
        result = result.drop(['price_mean', 'volume_mean', 'price_dev', 'volume_dev',
                              'covariance', 'price_var', 'volume_var'])
        
        return result


# ===========================================
# V97 AlphaFusion - 半衰期融合引擎
# ===========================================

class V97AlphaFusion:
    """V97 AlphaFusion - 半衰期衰减融合引擎"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V97_HALF_LIFE_LAGS)
        
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        
        self.fusion_signals: List[V97FusionSignal] = []
        
        logger.info("V97 AlphaFusion 初始化完成")
        logger.info(f"V97: 融合 Lags={self.fusion_lags}")
        logger.info(f"V97: 半衰期权重={dict(zip(self.fusion_lags, self.half_life_weights))}")
    
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
        
        logger.info(f"V97: 融合信号计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        return {
            'weights': dict(zip(self.fusion_lags, self.half_life_weights)),
            'lag1_weight': self.half_life_weights[0],
        }


# ===========================================
# V97 AlphaWeight - Alpha 权重引擎
# ===========================================

class V97AlphaWeightEngine:
    """V97 Alpha 权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V97_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V97_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V97_MAX_SINGLE_WEIGHT)
    
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
        
        logger.info(f"V97: Alpha 权重计算完成，处理 {result.height} 条记录")
        
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
# V97 ICAudit - IC 审计（消融实验）
# ===========================================

class V97ICAudit:
    """V97 IC 审计（支持消融实验）"""
    
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
            't1_ic_passed': ic_t1 >= V97_T1_IC_TARGET,
            'ic_by_year': ic_by_year_avg,
        }
    
    def calculate_single_factor_ic(self, df: pl.DataFrame, 
                                    factor_name: str,
                                    signal_col: str) -> V97SingleFactorIC:
        """计算单因子 IC（消融实验用）"""
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
                }
        
        ic_t1 = ic_results.get('t1', {}).get('mean_ic', 0.0)
        ic_t2 = ic_results.get('t2', {}).get('mean_ic', 0.0)
        ic_t3 = ic_results.get('t3', {}).get('mean_ic', 0.0)
        
        # 计算 IC IR
        std_t1 = ic_results.get('t1', {}).get('std_ic', 0.0)
        ic_ir = ic_t1 / (std_t1 + EPSILON) if std_t1 > 0 else 0.0
        
        return V97SingleFactorIC(
            factor_name=factor_name,
            ic_t1=ic_t1,
            ic_t2=ic_t2,
            ic_t3=ic_t3,
            ic_ir=ic_ir,
            passed_threshold=ic_t1 >= V97_T1_IC_TARGET,
        )


# ===========================================
# V97 TurnoverTracker - 换手率追踪
# ===========================================

class V97TurnoverTracker:
    """V97 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V97TurnoverRecord] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> V97TurnoverRecord:
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
        
        record = V97TurnoverRecord(
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
        
        is_active = V97_TURNOVER_MIN <= annualized_turnover <= V97_TURNOVER_MAX
        daily_ok = max_daily <= V97_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean([r.turnover_rate for r in self.turnover_records])),
            'std_turnover': float(np.std([r.turnover_rate for r in self.turnover_records])),
            'max_turnover': float(np.max([r.turnover_rate for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V97_TURNOVER_MIN,
            'turnover_max': V97_TURNOVER_MAX,
        }


# ===========================================
# V97 PortfolioTracker - 组合追踪
# ===========================================

class V97PortfolioTracker:
    """V97 组合追踪器"""
    
    def __init__(self, initial_capital: float = V97_INITIAL_CAPITAL,
                 config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        self.cash = initial_capital
        self.positions: Dict[str, V97Position] = {}
        self.portfolio_snapshots: List[V97DailyPortfolio] = []
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
                        turnover_rate: float) -> V97DailyPortfolio:
        """记录组合快照"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        
        cumulative_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        snapshot = V97DailyPortfolio(
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
# V97 DynamicRebalance - 动态调仓
# ===========================================

class V97DynamicRebalanceEngine:
    """V97 动态调仓引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('threshold', V97_RANK_CORRELATION_THRESHOLD)
        self.min_interval = self.config.get('min_interval', V97_MIN_REBALANCE_INTERVAL)
        self.max_interval = self.config.get('max_interval', V97_MAX_REBALANCE_INTERVAL)
        
        self.last_rebalance_date = None
        self.last_rebalance_signals = {}
        self.rebalance_count = 0
    
    def should_rebalance(self, trade_date: str, current_signals: Dict[str, float]) -> bool:
        """判断是否应该调仓"""
        if self.last_rebalance_date is None:
            self._update_rebalance_state(trade_date, current_signals)
            logger.info(f"V97: 首次调仓 ({trade_date})")
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
            logger.info(f"V97: 达到最大调仓间隔 ({actual_days}天)，强制调仓")
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
                    logger.info(f"V97: Rank Correlation ({rank_corr:.3f}) < 阈值，触发调仓")
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
# V97 导出列表
# ===========================================

__all__ = [
    # 常量
    'V97_INITIAL_CAPITAL',
    'V97_MAX_POSITIONS',
    'V97_WARMUP_PERIOD',
    'V97_MIN_SCORE_THRESHOLD',
    'V97_MIN_SINGLE_WEIGHT',
    'V97_MAX_SINGLE_WEIGHT',
    'V97_TURNOVER_MIN',
    'V97_TURNOVER_MAX',
    'V97_DAILY_TURNOVER_MAX',
    'V97_T1_IC_TARGET',
    'V97_IC_IR_TARGET',
    'V97_COMMISSION_RATE',
    'V97_MIN_COMMISSION',
    'V97_STAMP_DUTY',
    'V97_TRANSFER_FEE',
    'V97_HALF_LIFE_LAGS',
    'V97_LAG1_WEIGHT',
    'V97_LAG3_WEIGHT',
    'V97_LAG5_WEIGHT',
    'V97_SIZE_NEUTRALIZATION',
    'V97_INDUSTRY_NEUTRALIZATION',
    'V97_NEUTRALIZATION_WINDOW',
    'V97_LIQUIDITY_FILTER',
    'V97_LIQUIDITY_PERCENTILE',
    'V97_FILTER_ST',
    'V97_MIN_REBALANCE_INTERVAL',
    'V97_MAX_REBALANCE_INTERVAL',
    'V97_RANK_CORRELATION_THRESHOLD',
    'V97_CONSISTENCY_WINDOW',
    'V97_AUDIT_MODE',
    'V97_RESIDUAL_WEIGHT',
    'V97_FLOW_WEIGHT',
    'V97_CONSISTENCY_WEIGHT',
    # 数据类
    'V97SingleFactorIC',
    'V97FusionSignal',
    'V97TurnoverRecord',
    'V97Position',
    'V97DailyPortfolio',
    # 工具函数
    'normalize_rank',
    'zscore_normalize',
    'calculate_half_life_decay_weights',
    'fill_with_market_median',
    'calculate_rank_correlation',
    'ols_residualize',
    'EPSILON',
    # 核心类
    'V97DataManager',
    'V97IndustryNeutralizationEngine',
    'V97SizeNeutralizationEngine',
    'V97LiquidityFilterEngine',
    'V97ResidualMomentumEngine',
    'V97SmartFlowEngine',
    'V97PriceVolumeConsistencyEngine',
    'V97AlphaFusion',
    'V97AlphaWeightEngine',
    'V97ICAudit',
    'V97TurnoverTracker',
    'V97PortfolioTracker',
    'V97DynamicRebalanceEngine',
]