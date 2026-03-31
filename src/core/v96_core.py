"""
V96 Core Module - 强制中性化下的 Alpha 深度挖掘

【V96 核心理念】
1. 强制开启中性化 - 行业中性化和市值中性化必须开启
2. 强制开启流动性过滤 - 禁止买入日均成交额后 10% 的股票，禁止买入 ST 股
3. 丢弃无效因子 - V95 的"量价背离"IC 太低，予以丢弃
4. 新尝试 - 引入"日内收益率分布偏度 (Intraday Return Skewness)"因子
5. 特征交互 - 将该因子与 V90 的残差动量进行非线性叠加

【V96 硬性指标】
| 指标 | 目标值 | 惩罚 |
| :--- | :--- | :--- |
| T+1 Rank IC | > 0.048 | 低于此值则版本判定为失败 |
| 中性化状态 | 必须为 ENABLED | 若禁用，整个回测报告无效 |
| 最大回撤 | < 12% | 若超过，说明 Alpha 质量极差 |
| 数学一致性 | 误差 < 0.01% | 必须严格匹配 |

【V96 因子框架】
1. Refined Residual (V90 继承) - 残差动量
2. Smart Flow (V90 继承) - 聪明资金流
3. Vol_Price_Interaction (V90 继承) - 量价交互
4. Intraday Skewness (V96 新增) - 日内收益率分布偏度

作者：量化系统
版本：V96.0
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
from loguru import logger

# ===========================================
# V96 配置常量（强制中性化）
# ===========================================

V96_INITIAL_CAPITAL = 100000.00  # 初始资金（严禁修改）
V96_MAX_POSITIONS = 50  # 最大持仓数量
V96_WARMUP_PERIOD = 250
V96_MIN_SAMPLE_SIZE = 100

# 因子权重配置（V96 优化：丢弃量价背离因子，新增日内偏度因子）
V96_RESIDUAL_WEIGHT = 0.30       # 残差动量权重
V96_FLOW_WEIGHT = 0.20           # 聪明资金流权重
V96_INTERACTION_WEIGHT = 0.40    # 量价交互权重
V96_SKEWNESS_WEIGHT = 0.10       # 日内偏度因子权重（新增）

# 评分门槛配置
V96_MIN_SCORE_THRESHOLD = 45.0   # 评分门槛
V96_MIN_SINGLE_WEIGHT = 0.002    # 最小权重
V96_MAX_SINGLE_WEIGHT = 0.06     # 最大权重

# 换手率控制配置
V96_TURNOVER_MIN = 3.0
V96_TURNOVER_MAX = 6.0
V96_DAILY_TURNOVER_MAX = 0.50

# 费率配置
V96_COMMISSION_RATE = 0.002
V96_MIN_COMMISSION = 5.0
V96_STAMP_DUTY = 0.0005
V96_TRANSFER_FEE = 0.00001

# IC 目标
V96_T1_IC_TARGET = 0.048  # T+1 IC 目标
V96_IC_IR_TARGET = 0.5

# V96 新增：日内偏度因子配置
V96_INTRADAY_WINDOW = 20      # 日内数据窗口
V96_SKEWNESS_THRESHOLD = 0.5  # 偏度阈值

# 风格中性化配置（V96 强制：必须开启）
V96_SIZE_NEUTRALIZATION = True   # 强制开启市值中性化
V96_INDUSTRY_NEUTRALIZATION = True  # 强制开启行业中性化
V96_NEUTRALIZATION_WINDOW = 60

# 流动性过滤配置（V96 强制：必须开启）
V96_LIQUIDITY_FILTER = True      # 强制开启流动性过滤
V96_LIQUIDITY_PERCENTILE = 10    # 禁止买入后 10% 的股票
V96_FILTER_ST = True             # 强制过滤 ST 股

# 调仓配置
V96_MIN_REBALANCE_INTERVAL = 3
V96_MAX_REBALANCE_INTERVAL = 3
V96_RANK_CORRELATION_THRESHOLD = 0.35

# 半衰期融合配置
V96_HALF_LIFE_LAGS = [1, 3, 5]
V96_LAG1_WEIGHT = 0.50
V96_LAG3_WEIGHT = 0.30
V96_LAG5_WEIGHT = 0.20

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V96SingleFactorIC:
    """单因子 IC 记录"""
    factor_name: str
    ic_t1: float
    ic_t2: float
    ic_t3: float
    ic_ir: float
    passed_threshold: bool


@dataclass
class V96FusionSignal:
    """半衰期融合信号"""
    trade_date: str
    symbol: str
    signal_t1: float
    signal_t3: float
    signal_t5: float
    fused_signal: float


@dataclass
class V96TurnoverRecord:
    """换手记录"""
    trade_date: str
    turnover_rate: float
    buy_turnover: float
    sell_turnover: float
    annualized_turnover: float
    daily_turnover: float
    is_rebalance_day: bool = False


@dataclass
class V96Position:
    """持仓记录"""
    symbol: str
    entry_date: str
    entry_price: float
    quantity: int
    weight: float
    current_price: float = 0.0
    pnl: float = 0.0


@dataclass
class V96DailyPortfolio:
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
# V96 工具函数
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


def calculate_skewness(series: np.ndarray) -> float:
    """计算偏度"""
    if len(series) < 3:
        return 0.0
    return float(stats.skew(series))


def calculate_half_life_decay_weights(lags: List[int] = V96_HALF_LIFE_LAGS) -> List[float]:
    """计算半衰期衰减权重"""
    fixed_weights = {
        1: V96_LAG1_WEIGHT,
        3: V96_LAG3_WEIGHT,
        5: V96_LAG5_WEIGHT,
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


# ===========================================
# V96 DataManager
# ===========================================

class V96DataManager:
    """V96 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V96_WARMUP_PERIOD)
    
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
        extra_days = max(V96_HALF_LIFE_LAGS) + 20
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
            
            logger.info(f"V96: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V96: 数据加载失败 - {e}")
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
                logger.warning(f"V96: 获取最常见行业失败 - {e}，使用默认值")
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
# V96 Industry Neutralization - 行业中性化（V96 强制开启）
# ===========================================

class V96IndustryNeutralizationEngine:
    """
    V96 行业中性化引擎
    
    【强制要求】
    1. 必须开启行业中性化
    2. 使用行业哑变量进行回归
    3. 取残差作为中性化后的信号
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.industry_neutralization = self.config.get('industry_neutralization', V96_INDUSTRY_NEUTRALIZATION)
        self.window = self.config.get('window', V96_NEUTRALIZATION_WINDOW)
    
    def compute_industry_neutralization(self, df: pl.DataFrame, 
                                         signal_col: str = 'composite_score') -> pl.DataFrame:
        """
        计算行业中性化
        
        核心逻辑：
        1. 对每个交易日，将信号对行业哑变量回归
        2. 取残差作为中性化后的信号
        """
        if not self.industry_neutralization:
            logger.warning("V96: 行业中性化未开启，这将导致回测无效！")
            return df
        
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 获取行业映射数据
        industry_df = self._load_industry_data()
        
        if industry_df.is_empty():
            logger.warning("V96: 无法获取行业数据，使用行业代码代替")
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
        
        logger.info(f"V96: 行业中性化计算完成 (ENABLED)，处理 {result.height} 条记录")
        
        return result
    
    def _load_industry_data(self) -> pl.DataFrame:
        """加载行业数据"""
        if self.db is None:
            return pl.DataFrame()
        
        try:
            import connectorx as cx
            import pandas as pd
            
            db_url = f"mysql://{os.getenv('MYSQL_USER', 'root')}:{os.getenv('MYSQL_PASSWORD', '')}@{os.getenv('MYSQL_HOST', 'localhost')}:{os.getenv('MYSQL_PORT', '3306')}/{os.getenv('MYSQL_DATABASE', 'quantitative_trading')}"
            
            query = """
                SELECT symbol, trade_date, industry_name
                FROM stock_industry_daily
            """
            
            pdf = cx.read_sql(db_url, query)
            return pl.from_pandas(pdf)
            
        except Exception as e:
            logger.warning(f"V96: 加载行业数据失败 - {e}")
            return pl.DataFrame()
    
    def _neutralize_by_industry(self, signals: np.ndarray, industries: np.ndarray) -> np.ndarray:
        """
        按行业中性化信号
        
        方法：对每个行业，减去行业均值，得到残差
        """
        unique_industries = np.unique(industries)
        
        if len(unique_industries) < 2:
            return signals
        
        neutralized = np.zeros_like(signals)
        
        for industry in unique_industries:
            mask = industries == industry
            industry_signals = signals[mask]
            
            if len(industry_signals) < 2:
                neutralized[mask] = industry_signals
                continue
            
            # 计算行业均值和标准差
            industry_mean = np.mean(industry_signals)
            industry_std = np.std(industry_signals)
            
            if industry_std < EPSILON:
                neutralized[mask] = np.zeros_like(industry_signals)
            else:
                # 取残差
                neutralized[mask] = (industry_signals - industry_mean) / industry_std
        
        # 恢复原始量纲
        neutralized = neutralized * np.std(signals) + np.mean(signals)
        
        return neutralized


# ===========================================
# V96 Size Neutralization - 市值中性化（V96 强制开启）
# ===========================================

class V96SizeNeutralizationEngine:
    """
    V96 市值中性化引擎
    
    【强制要求】
    1. 必须开启市值中性化
    2. 使用对数市值进行回归
    3. 取残差作为中性化后的信号
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.size_neutralization = self.config.get('size_neutralization', V96_SIZE_NEUTRALIZATION)
    
    def compute_size_neutralization(self, df: pl.DataFrame, 
                                     signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算市值中性化"""
        if not self.size_neutralization:
            logger.warning("V96: 市值中性化未开启，这将导致回测无效！")
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
        
        logger.info(f"V96: 市值中性化计算完成 (ENABLED)，处理 {result.height} 条记录")
        
        return result
    
    def _neutralize_by_size(self, signals: np.ndarray, sizes: np.ndarray) -> np.ndarray:
        """
        按市值中性化信号
        
        方法：线性回归取残差
        """
        n = len(signals)
        
        sizes_std = (sizes - np.mean(sizes)) / (np.std(sizes) + EPSILON)
        signals_std = (signals - np.mean(signals)) / (np.std(signals) + EPSILON)
        
        # 简单线性回归
        try:
            # y = a + b*x
            # b = cov(x,y) / var(x)
            b = np.cov(sizes_std, signals_std)[0, 1] / (np.var(sizes_std) + EPSILON)
            a = np.mean(signals_std) - b * np.mean(sizes_std)
            
            fitted = a + b * sizes_std
            residual = signals_std - fitted
            
            # 恢复原始量纲
            neutralized = residual * np.std(signals) + np.mean(signals)
            return neutralized
        except Exception:
            return signals


# ===========================================
# V96 Liquidity Filter - 流动性过滤（V96 强制开启）
# ===========================================

class V96LiquidityFilterEngine:
    """
    V96 流动性过滤引擎
    
    【强制要求】
    1. 禁止买入日均成交额后 10% 的股票
    2. 禁止买入 ST 股
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.liquidity_filter = self.config.get('liquidity_filter', V96_LIQUIDITY_FILTER)
        self.liquidity_percentile = self.config.get('liquidity_percentile', V96_LIQUIDITY_PERCENTILE)
        self.filter_st = self.config.get('filter_st', V96_FILTER_ST)
    
    def apply_liquidity_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """应用流动性过滤"""
        if not self.liquidity_filter:
            logger.warning("V96: 流动性过滤未开启，这将导致回测无效！")
            return df
        
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 计算滚动平均成交额
        result = result.with_columns([
            pl.col('amount').fill_null(0).alias('amount_filled')
        ])
        
        result = result.with_columns([
            pl.col('amount_filled')
            .rolling_mean(window_size=V96_INTRADAY_WINDOW)
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
        
        logger.info(f"V96: 流动性过滤计算完成 (ENABLED)")
        
        return result


# ===========================================
# V96 Intraday Skewness Engine - 日内偏度因子（V96 新增）
# ===========================================

class V96IntradaySkewnessEngine:
    """
    V96 日内收益率分布偏度因子引擎
    
    【核心逻辑】
    研究日内分钟级收益的分布：
    - 正偏 (Positive Skew) 代表资金边拉边撤
    - 负偏 (Negative Skew) 代表资金暗中吸筹
    
    【计算公式】
    1. 使用 (high - low) / close 作为日内收益率的代理
    2. 计算滚动窗口内的偏度
    3. 负偏给予正向信号（吸筹），正偏给予负向信号（派发）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.window = self.config.get('window', V96_INTRADAY_WINDOW)
        self.threshold = self.config.get('threshold', V96_SKEWNESS_THRESHOLD)
    
    def compute_intraday_skewness(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算日内偏度信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算日内收益率代理：(high - low) / close
        result = result.with_columns([
            ((pl.col('high') - pl.col('low')) / (pl.col('close') + EPSILON)).alias('intraday_return')
        ])
        
        # 2. 计算滚动偏度（使用 Polars 的 skew 函数）
        # 注意：Polars 没有直接的 skew 函数，我们使用近似方法
        result = self._compute_rolling_skewness(result)
        
        # 3. 偏度信号转换
        # 负偏 -> 吸筹 -> 正信号
        # 正偏 -> 派发 -> 负信号
        result = result.with_columns([
            (-pl.col('rolling_skew')).alias('skewness_signal_raw')
        ])
        
        # 4. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('skewness_signal_raw').rank('ordinal', descending=True).over('trade_date').alias('skewness_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('skewness_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('intraday_skewness_score')
        ])
        
        # 5. 清理临时列
        result = result.drop(['intraday_return', 'skewness_rank', 'n_stocks'])
        
        logger.info(f"V96: 日内偏度因子计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _compute_rolling_skewness(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算滚动偏度
        
        使用三阶中心矩近似偏度：
        skew = E[(X - μ)³] / σ³
        """
        result = df.clone()
        
        # 计算滚动均值和标准差
        result = result.with_columns([
            pl.col('intraday_return')
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('rolling_mean')
        ])
        
        result = result.with_columns([
            pl.col('intraday_return')
            .rolling_std(window_size=self.window)
            .over('symbol')
            .alias('rolling_std')
        ])
        
        # 计算三阶中心矩的滚动平均
        result = result.with_columns([
            ((pl.col('intraday_return') - pl.col('rolling_mean')) ** 3).alias('cubed_deviation')
        ])
        
        result = result.with_columns([
            pl.col('cubed_deviation')
            .rolling_mean(window_size=self.window)
            .over('symbol')
            .alias('rolling_third_moment')
        ])
        
        # 计算偏度
        result = result.with_columns([
            (pl.col('rolling_third_moment') / 
             (pl.col('rolling_std') ** 3 + EPSILON)).alias('rolling_skew')
        ])
        
        # 清理临时列
        result = result.drop(['cubed_deviation', 'rolling_third_moment'])
        
        return result


# ===========================================
# V96 NonLinear Interaction - 非线性交互（V96 新增）
# ===========================================

class V96NonLinearInteractionEngine:
    """
    V96 非线性交互引擎
    
    【核心逻辑】
    将日内偏度因子与 V90 残差动量进行非线性叠加
    
    【交互方式】
    1. 乘法交互：skewness * residual
    2. 条件加权：根据偏度方向调整残差动量权重
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.interaction_strength = self.config.get('interaction_strength', 0.3)
    
    def compute_nonlinear_interaction(self, df: pl.DataFrame,
                                       residual_col: str = 'refined_residual_score',
                                       skewness_col: str = 'intraday_skewness_score') -> pl.DataFrame:
        """计算非线性交互信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 归一化因子到 0-1 范围
        result = result.with_columns([
            (pl.col(residual_col) / 100.0).clip(0, 1).alias('residual_norm'),
            (pl.col(skewness_col) / 100.0).clip(0, 1).alias('skewness_norm')
        ])
        
        # 2. 乘法交互
        result = result.with_columns([
            (pl.col('residual_norm') * pl.col('skewness_norm')).alias('interaction_product')
        ])
        
        # 3. 条件加权交互
        # 当偏度为负（吸筹）时，增强残差动量信号
        # 当偏度为正（派发）时，减弱残差动量信号
        result = result.with_columns([
            (pl.col('residual_norm') * (1.0 + self.interaction_strength * (pl.col('skewness_norm') - 0.5))).alias('interaction_weighted')
        ])
        
        # 4. 融合交互信号
        result = result.with_columns([
            (
                pl.col('interaction_product') * 0.5 +
                pl.col('interaction_weighted') * 0.5
            ).alias('nonlinear_interaction_score')
        ])
        
        # 5. 转换为百分位
        result = result.with_columns([
            pl.col('nonlinear_interaction_score').rank('ordinal', descending=True).over('trade_date').alias('interaction_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('interaction_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('nonlinear_interaction_percentile')
        ])
        
        # 清理临时列
        result = result.drop(['residual_norm', 'skewness_norm', 'interaction_product', 
                              'interaction_weighted', 'interaction_rank', 'n_stocks',
                              'nonlinear_interaction_score'])
        
        logger.info(f"V96: 非线性交互计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V96 AlphaFusion - 半衰期融合引擎
# ===========================================

class V96AlphaFusion:
    """V96 AlphaFusion - 半衰期衰减融合引擎"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.fusion_lags = self.config.get('fusion_lags', V96_HALF_LIFE_LAGS)
        
        self.half_life_weights = calculate_half_life_decay_weights(self.fusion_lags)
        
        self.fusion_signals: List[V96FusionSignal] = []
        
        logger.info("V96 AlphaFusion 初始化完成")
        logger.info(f"V96: 融合 Lags={self.fusion_lags}")
        logger.info(f"V96: 半衰期权重={dict(zip(self.fusion_lags, self.half_life_weights))}")
    
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
        
        logger.info(f"V96: 融合信号计算完成，处理 {result.height} 条记录")
        
        return result
    
    def get_fusion_summary(self) -> Dict[str, Any]:
        """获取融合信号摘要"""
        return {
            'weights': dict(zip(self.fusion_lags, self.half_life_weights)),
            'lag1_weight': self.half_life_weights[0],
        }


# ===========================================
# V96 AlphaWeight - Alpha 权重引擎
# ===========================================

class V96AlphaWeightEngine:
    """V96 Alpha 权重引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_score = self.config.get('min_score', V96_MIN_SCORE_THRESHOLD)
        self.min_weight = self.config.get('min_weight', V96_MIN_SINGLE_WEIGHT)
        self.max_weight = self.config.get('max_weight', V96_MAX_SINGLE_WEIGHT)
    
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
        
        logger.info(f"V96: Alpha 权重计算完成，处理 {result.height} 条记录")
        
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
# V96 ICAudit - IC 审计
# ===========================================

class V96ICAudit:
    """V96 IC 审计"""
    
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
            't1_ic_passed': ic_t1 >= V96_T1_IC_TARGET,
            'ic_by_year': ic_by_year_avg,
        }


# ===========================================
# V96 TurnoverTracker - 换手率追踪
# ===========================================

class V96TurnoverTracker:
    """V96 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[V96TurnoverRecord] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> V96TurnoverRecord:
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
        
        record = V96TurnoverRecord(
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
        
        is_active = V96_TURNOVER_MIN <= annualized_turnover <= V96_TURNOVER_MAX
        daily_ok = max_daily <= V96_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean([r.turnover_rate for r in self.turnover_records])),
            'std_turnover': float(np.std([r.turnover_rate for r in self.turnover_records])),
            'max_turnover': float(np.max([r.turnover_rate for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V96_TURNOVER_MIN,
            'turnover_max': V96_TURNOVER_MAX,
        }


# ===========================================
# V96 PortfolioTracker - 组合追踪
# ===========================================

class V96PortfolioTracker:
    """V96 组合追踪器"""
    
    def __init__(self, initial_capital: float = V96_INITIAL_CAPITAL,
                 config: Dict[str, Any] = None):
        self.initial_capital = initial_capital
        self.config = config or {}
        
        self.cash = initial_capital
        self.positions: Dict[str, V96Position] = {}
        self.portfolio_snapshots: List[V96DailyPortfolio] = []
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
                        turnover_rate: float) -> V96DailyPortfolio:
        """记录组合快照"""
        position_value = sum(
            p.current_price * p.quantity for p in self.positions.values()
        )
        
        cumulative_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        snapshot = V96DailyPortfolio(
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
# V96 DynamicRebalance - 动态调仓
# ===========================================

class V96DynamicRebalanceEngine:
    """V96 动态调仓引擎"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.correlation_threshold = self.config.get('threshold', V96_RANK_CORRELATION_THRESHOLD)
        self.min_interval = self.config.get('min_interval', V96_MIN_REBALANCE_INTERVAL)
        self.max_interval = self.config.get('max_interval', V96_MAX_REBALANCE_INTERVAL)
        
        self.last_rebalance_date = None
        self.last_rebalance_signals = {}
        self.rebalance_count = 0
    
    def should_rebalance(self, trade_date: str, current_signals: Dict[str, float]) -> bool:
        """判断是否应该调仓"""
        if self.last_rebalance_date is None:
            self._update_rebalance_state(trade_date, current_signals)
            logger.info(f"V96: 首次调仓 ({trade_date})")
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
            logger.info(f"V96: 达到最大调仓间隔 ({actual_days}天)，强制调仓")
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
                    logger.info(f"V96: Rank Correlation ({rank_corr:.3f}) < 阈值，触发调仓")
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
# V96 导出列表
# ===========================================

__all__ = [
    # 常量
    'V96_INITIAL_CAPITAL',
    'V96_MAX_POSITIONS',
    'V96_WARMUP_PERIOD',
    'V96_MIN_SCORE_THRESHOLD',
    'V96_MIN_SINGLE_WEIGHT',
    'V96_MAX_SINGLE_WEIGHT',
    'V96_TURNOVER_MIN',
    'V96_TURNOVER_MAX',
    'V96_DAILY_TURNOVER_MAX',
    'V96_T1_IC_TARGET',
    'V96_IC_IR_TARGET',
    'V96_COMMISSION_RATE',
    'V96_MIN_COMMISSION',
    'V96_STAMP_DUTY',
    'V96_TRANSFER_FEE',
    'V96_HALF_LIFE_LAGS',
    'V96_LAG1_WEIGHT',
    'V96_LAG3_WEIGHT',
    'V96_LAG5_WEIGHT',
    'V96_SIZE_NEUTRALIZATION',
    'V96_INDUSTRY_NEUTRALIZATION',
    'V96_NEUTRALIZATION_WINDOW',
    'V96_LIQUIDITY_FILTER',
    'V96_LIQUIDITY_PERCENTILE',
    'V96_FILTER_ST',
    'V96_MIN_REBALANCE_INTERVAL',
    'V96_MAX_REBALANCE_INTERVAL',
    'V96_RANK_CORRELATION_THRESHOLD',
    'V96_INTRADAY_WINDOW',
    'V96_SKEWNESS_THRESHOLD',
    'V96_RESIDUAL_WEIGHT',
    'V96_FLOW_WEIGHT',
    'V96_INTERACTION_WEIGHT',
    'V96_SKEWNESS_WEIGHT',
    # 数据类
    'V96SingleFactorIC',
    'V96FusionSignal',
    'V96TurnoverRecord',
    'V96Position',
    'V96DailyPortfolio',
    # 工具函数
    'normalize_rank',
    'zscore_normalize',
    'calculate_skewness',
    'calculate_half_life_decay_weights',
    'fill_with_market_median',
    'calculate_rank_correlation',
    'EPSILON',
    # 核心类
    'V96DataManager',
    'V96IndustryNeutralizationEngine',
    'V96SizeNeutralizationEngine',
    'V96LiquidityFilterEngine',
    'V96IntradaySkewnessEngine',
    'V96NonLinearInteractionEngine',
    'V96AlphaFusion',
    'V96AlphaWeightEngine',
    'V96ICAudit',
    'V96TurnoverTracker',
    'V96PortfolioTracker',
    'V96DynamicRebalanceEngine',
]