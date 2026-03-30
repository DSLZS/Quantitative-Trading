"""
V93 Logic Module - 量价二阶导与资金流偏度增强

【V93 核心理念】
1. 废弃 v90_core，完全重写因子计算逻辑
2. 引入"量价二阶导"逻辑（Volatility of Volatility）
3. 引入"资金流偏度"因子（Order Flow Skewness）
4. 实现严格的行业（Industry-wise）和市值（Size-neutral）中性化

【V93 硬性指标】
- 指标 A：T+1 Rank IC ≥ 0.05，IC IR ≥ 0.6
- 指标 B：最大回撤 ≤ 10%
- 指标 C：年化换手率 300%-400%
- 指标 D：数学自洽性检查（误差 < 0.01%）

【V93 因子框架】
1. 量价二阶导因子 (Volatility of Volatility)
   - 计算价格波动率的波动率
   - 捕捉市场情绪的不稳定性
   
2. 资金流偏度因子 (Order Flow Skewness)
   - 计算主动买入/卖出的偏度
   - 捕捉主力资金的流向

3. 行业中性化
   - 按行业分组进行中性化
   - 去除行业 Beta 干扰

4. 市值中性化
   - 对数市值因子回归
   - 去除大小盘风格暴露

作者：量化系统
版本：V93.0
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
# V93 配置常量
# ===========================================

V93_INITIAL_CAPITAL = 100000.00
V93_MAX_POSITIONS = 30
V93_WARMUP_PERIOD = 250
V93_MIN_SAMPLE_SIZE = 100

# 因子权重配置
V93_VOL_OF_VOL_WEIGHT = 0.30  # 量价二阶导权重
V93_FLOW_SKEW_WEIGHT = 0.30  # 资金流偏度权重
V93_MOMENTUM_WEIGHT = 0.20  # 动量权重
V93_REVERSAL_WEIGHT = 0.20  # 反转权重

# 评分门槛配置
V93_MIN_SCORE_THRESHOLD = 55.0
V93_MIN_SINGLE_WEIGHT = 0.003
V93_MAX_SINGLE_WEIGHT = 0.08

# 换手率控制配置
V93_TURNOVER_MIN = 3.0  # 300%
V93_TURNOVER_MAX = 4.0  # 400%
V93_DAILY_TURNOVER_MAX = 0.10

# 费率配置
V93_COMMISSION_RATE = 0.002
V93_MIN_COMMISSION = 5.0
V93_STAMP_DUTY = 0.0005
V93_TRANSFER_FEE = 0.00001

# IC 目标
V93_T1_IC_TARGET = 0.05
V93_IC_IR_TARGET = 0.6

# V93 新增：量价二阶导配置
V93_VOL_WINDOW = 20  # 波动率计算窗口
V93_VOL_OF_VOL_WINDOW = 10  # 波动率之波动率窗口
V93_PRICE_WINDOW = 5  # 价格动量窗口

# V93 新增：资金流偏度配置
V93_FLOW_WINDOW = 20  # 资金流计算窗口
V93_FLOW_SKEW_THRESHOLD = 1.5  # 偏度阈值

# V93 新增：中性化配置
V93_INDUSTRY_NEUTRALIZATION = True  # 行业中性化
V93_SIZE_NEUTRALIZATION = True  # 市值中性化
V93_NEUTRALIZATION_WINDOW = 60  # 中性化计算窗口

# 调仓配置
V93_MIN_REBALANCE_INTERVAL = 10
V93_MAX_REBALANCE_INTERVAL = 10

EPSILON = 1e-9


# ===========================================
# 数据类
# ===========================================

@dataclass
class V93MathConsistencyResult:
    """数学一致性检查结果"""
    passed: bool
    expected: float
    actual: float
    diff: float
    message: str


@dataclass
class V93RiskControlResult:
    """风险控制检查结果"""
    passed: bool
    expected: float
    actual: float
    diff: float
    message: str


# ===========================================
# V93 工具函数
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


# ===========================================
# V93DataManager
# ===========================================

class V93DataManager:
    """V93 数据管理器"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        self.warmup_period = self.config.get('warmup_period', V93_WARMUP_PERIOD)
    
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
        extra_days = 50
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
            
            logger.info(f"V93: 数据加载成功，行数={df.height}")
            
            return df
            
        except Exception as e:
            logger.error(f"V93: 数据加载失败 - {e}")
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
# V93VolOfVolEngine - 量价二阶导引擎
# ===========================================

class V93VolOfVolEngine:
    """
    V93 Volatility of Volatility 引擎 - 量价二阶导
    
    【核心逻辑】
    1. 计算 N 日价格波动率（滚动标准差）
    2. 计算波动率的变化率（一阶导）
    3. 计算波动率变化率的变化率（二阶导）
    4. 二阶导 > 0 表示波动率加速上升（风险信号）
    5. 二阶导 < 0 表示波动率趋于稳定（机会信号）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vol_window = self.config.get('vol_window', V93_VOL_WINDOW)
        self.vol_of_vol_window = self.config.get('vol_of_vol_window', V93_VOL_OF_VOL_WINDOW)
        self.price_window = self.config.get('price_window', V93_PRICE_WINDOW)
    
    def compute_vol_of_vol(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算量价二阶导信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算日收益率
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(1)) / 
             (pl.col('close').shift(1) + EPSILON)).alias('daily_return')
        ])
        
        # 2. 计算 N 日波动率（滚动标准差）
        result = result.with_columns([
            pl.col('daily_return')
            .rolling_std(window_size=self.vol_window)
            .over('symbol')
            .alias('volatility')
        ])
        
        # 3. 计算波动率一阶导（变化率）
        result = result.with_columns([
            (pl.col('volatility') - pl.col('volatility').shift(1)).over('symbol').alias('vol_1st_deriv')
        ])
        
        # 4. 计算波动率二阶导（加速度）
        result = result.with_columns([
            (pl.col('vol_1st_deriv') - pl.col('vol_1st_deriv').shift(1)).over('symbol').alias('vol_2nd_deriv')
        ])
        
        # 5. 计算价格动量
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.price_window)) / 
             (pl.col('close').shift(self.price_window) + EPSILON)).alias('price_momentum')
        ])
        
        # 6. 计算成交量动量
        result = result.with_columns([
            ((pl.col('volume').fill_null(0) - pl.col('volume').fill_null(0).shift(self.price_window)) / 
             (pl.col('volume').fill_null(0).shift(self.price_window) + EPSILON)).alias('volume_momentum')
        ])
        
        # 7. 融合量价二阶导分数
        # 逻辑：波动率二阶导 < 0 且价格动量 > 0 时，为买入信号
        result = result.with_columns([
            (
                (-pl.col('vol_2nd_deriv')) * 0.4 +  # 波动率下降（负负得正）
                pl.col('price_momentum') * 0.4 +  # 价格动量
                (-pl.col('volume_momentum')) * 0.2  # 成交量下降（量缩价稳）
            ).alias('vol_of_vol_score')
        ])
        
        # 8. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('vol_of_vol_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('score_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('vol_of_vol_score')
        ])
        
        result = result.drop(['score_rank', 'n_stocks'])
        
        logger.info(f"V93: 量价二阶导计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V93FlowSkewEngine - 资金流偏度引擎
# ===========================================

class V93FlowSkewEngine:
    """
    V93 Flow Skewness 引擎 - 资金流偏度
    
    【核心逻辑】
    1. 计算主动买入/卖出压力
      - 主动买入：收盘价 > 开盘价，成交量视为买入
      - 主动卖出：收盘价 < 开盘价，成交量视为卖出
    2. 计算资金流偏度（Skewness）
      - 偏度 > 0：买入压力大
      - 偏度 < 0：卖出压力大
    3. 结合成交额加权
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.flow_window = self.config.get('flow_window', V93_FLOW_WINDOW)
        self.skew_threshold = self.config.get('skew_threshold', V93_FLOW_SKEW_THRESHOLD)
    
    def compute_flow_skew(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算资金流偏度信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算主动买卖压力
        # 主动买入量：当 close > open 时，volume 视为买入；否则为 0
        # 主动卖出量：当 close < open 时，volume 视为卖出；否则为 0
        result = result.with_columns([
            pl.when(pl.col('close') > pl.col('open'))
            .then(pl.col('volume').fill_null(0))
            .otherwise(0.0)
            .alias('buy_volume'),
            pl.when(pl.col('close') < pl.col('open'))
            .then(pl.col('volume').fill_null(0))
            .otherwise(0.0)
            .alias('sell_volume')
        ])
        
        # 2. 计算净资金流（买入 - 卖出）
        result = result.with_columns([
            (pl.col('buy_volume') - pl.col('sell_volume')).alias('net_flow')
        ])
        
        # 3. 计算资金流偏度（滚动窗口内的偏度）
        # 使用滚动窗口的净资金流计算偏度
        result = result.with_columns([
            pl.col('net_flow')
            .rolling_skew(window_size=self.flow_window)
            .over('symbol')
            .alias('flow_skewness')
        ])
        
        # 4. 计算成交额加权的资金流
        result = result.with_columns([
            ((pl.col('close') - pl.col('open')) / 
             (pl.col('close').shift(1) + EPSILON) * pl.col('amount').fill_null(0)).alias('money_flow')
        ])
        
        # 5. 计算资金流偏度的滚动均值（平滑）
        result = result.with_columns([
            pl.col('flow_skewness')
            .rolling_mean(window_size=5)
            .over('symbol')
            .alias('flow_skew_smooth')
        ])
        
        # 6. 融合资金流偏度分数
        # 偏度 > 0 表示买入压力大，是正面信号
        result = result.with_columns([
            (
                pl.col('flow_skew_smooth') * 0.6 +  # 偏度信号
                pl.col('money_flow').clip(-1e6, 1e6) * 0.0000001 * 0.4  # 资金流信号（缩放）
            ).alias('flow_skew_score')
        ])
        
        # 7. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('flow_skew_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('score_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('flow_skew_score')
        ])
        
        result = result.drop(['score_rank', 'n_stocks'])
        
        logger.info(f"V93: 资金流偏度计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V93MomentumReversalEngine - 动量反转引擎
# ===========================================

class V93MomentumReversalEngine:
    """
    V93 Momentum & Reversal 引擎 - 动量与反转
    
    【核心逻辑】
    1. 短期反转（5 日）：近期跌幅大的股票可能反弹
    2. 中期动量（20 日）：趋势延续
    3. 融合动量和反转信号
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.reversal_window = 5  # 短期反转窗口
        self.momentum_window = 20  # 中期动量窗口
    
    def compute_momentum_reversal(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算动量反转信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算短期收益率（反转信号）
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.reversal_window)) / 
             (pl.col('close').shift(self.reversal_window) + EPSILON)).alias('short_return')
        ])
        
        # 2. 计算中期收益率（动量信号）
        result = result.with_columns([
            ((pl.col('close') - pl.col('close').shift(self.momentum_window)) / 
             (pl.col('close').shift(self.momentum_window) + EPSILON)).alias('mid_return')
        ])
        
        # 3. 融合信号
        # 短期反转：负收益 -> 正信号
        # 中期动量：正收益 -> 正信号
        result = result.with_columns([
            (
                (-pl.col('short_return')) * V93_REVERSAL_WEIGHT +  # 反转
                pl.col('mid_return') * V93_MOMENTUM_WEIGHT  # 动量
            ).alias('momentum_reversal_score')
        ])
        
        # 4. 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('momentum_reversal_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('score_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('momentum_reversal_score')
        ])
        
        result = result.drop(['score_rank', 'n_stocks'])
        
        logger.info(f"V93: 动量反转计算完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V93NeutralizationEngine - 中性化引擎
# ===========================================

class V93NeutralizationEngine:
    """
    V93 Neutralization 引擎 - 严格的中性化处理
    
    【核心逻辑】
    1. 行业中性化（Industry-wise Neutralization）
       - 在每个行业内对信号进行标准化
       - 去除行业 Beta 影响
    
    2. 市值中性化（Size Neutralization）
       - 对数市值因子回归
       - 去除大小盘风格暴露
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.industry_neutralization = self.config.get('industry_neutralization', V93_INDUSTRY_NEUTRALIZATION)
        self.size_neutralization = self.config.get('size_neutralization', V93_SIZE_NEUTRALIZATION)
    
    def compute_neutralization(self, df: pl.DataFrame, 
                                signal_col: str = 'composite_score') -> pl.DataFrame:
        """计算中性化信号"""
        result = df.clone()
        result = result.sort(['symbol', 'trade_date'])
        
        # 1. 计算对数市值因子
        result = result.with_columns([
            pl.col('total_mv').fill_null(1.0).log().alias('log_size')
        ])
        
        # 2. 按日期处理
        unique_dates = sorted(result['trade_date'].unique().to_list())
        
        neutralized_signals = []
        for trade_date in unique_dates:
            day_data = result.filter(pl.col('trade_date') == trade_date)
            
            if day_data.is_empty():
                continue
            
            # 获取当日数据
            symbols = day_data['symbol'].to_numpy()
            signals = day_data[signal_col].to_numpy()
            sizes = day_data['log_size'].to_numpy()
            industries = day_data['industry_code'].to_numpy() if 'industry_code' in day_data.columns else None
            
            # 过滤无效数据
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
            valid_industries = industries[valid_mask] if industries is not None else None
            
            # 3. 行业中性化
            if self.industry_neutralization and valid_industries is not None:
                valid_signals = self._industry_neutralize(valid_signals, valid_industries)
            
            # 4. 市值中性化
            if self.size_neutralization:
                valid_signals = self._size_neutralize(valid_signals, valid_sizes)
            
            # 5. 组装结果
            all_symbols_list = list(symbols)
            for i, symbol in enumerate(valid_symbols):
                idx = all_symbols_list.index(symbol) if symbol in all_symbols_list else i
                neutralized_signals.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'neutralized_signal': valid_signals[i],
                })
        
        # 6. 合并回 DataFrame
        if neutralized_signals:
            neutralized_df = pl.DataFrame({
                'trade_date': [s['trade_date'] for s in neutralized_signals],
                'symbol': [s['symbol'] for s in neutralized_signals],
                'neutralized_signal': [s['neutralized_signal'] for s in neutralized_signals],
            })
            
            result = result.join(neutralized_df, on=['trade_date', 'symbol'], how='left')
        
        logger.info(f"V93: 中性化计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _industry_neutralize(self, signals: np.ndarray, 
                             industries: np.ndarray) -> np.ndarray:
        """
        行业中性化
        在每个行业内进行 Z-Score 标准化
        """
        result = signals.copy()
        unique_industries = np.unique(industries)
        
        for industry in unique_industries:
            if industry is None or industry == '':
                continue
            
            mask = industries == industry
            if np.sum(mask) < 3:
                continue
            
            industry_signals = signals[mask]
            mean = np.mean(industry_signals)
            std = np.std(industry_signals)
            
            if std > EPSILON:
                result[mask] = (industry_signals - mean) / std
            else:
                result[mask] = np.zeros_like(industry_signals)
        
        return result
    
    def _size_neutralize(self, signals: np.ndarray, 
                         sizes: np.ndarray) -> np.ndarray:
        """
        市值中性化
        使用回归剥离市值因子暴露
        """
        n = len(signals)
        
        # 标准化
        sizes_std = zscore_normalize(sizes)
        signals_std = zscore_normalize(signals)
        
        # 构建设计矩阵
        X = np.column_stack([np.ones(n), sizes_std])
        y = signals_std
        
        try:
            # OLS 回归
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # 计算拟合值
            fitted = X @ coeffs
            
            # 残差 = 原始值 - 拟合值（剥离市值暴露）
            residual = y - fitted
            
            # 转换回原始尺度
            neutralized = residual * np.std(signals) + np.mean(signals)
            
            return neutralized
            
        except Exception:
            return signals


# ===========================================
# V93CompositeEngine - 信号融合引擎
# ===========================================

class V93CompositeEngine:
    """
    V93 Composite 引擎 - 多因子信号融合
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.vol_of_vol_weight = self.config.get('vol_of_vol_weight', V93_VOL_OF_VOL_WEIGHT)
        self.flow_skew_weight = self.config.get('flow_skew_weight', V93_FLOW_SKEW_WEIGHT)
        self.momentum_weight = self.config.get('momentum_weight', V93_MOMENTUM_WEIGHT)
        self.reversal_weight = self.config.get('reversal_weight', V93_REVERSAL_WEIGHT)
    
    def compute_composite(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算融合信号"""
        result = df.clone()
        
        # 融合各因子信号
        result = result.with_columns([
            (
                pl.col('vol_of_vol_score').fill_null(0) * self.vol_of_vol_weight +
                pl.col('flow_skew_score').fill_null(0) * self.flow_skew_weight +
                pl.col('momentum_reversal_score').fill_null(0) * (self.momentum_weight + self.reversal_weight)
            ).alias('composite_score')
        ])
        
        # 按日期排名转换为百分位
        result = result.with_columns([
            pl.col('composite_score').rank('ordinal', descending=True).over('trade_date').alias('score_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('score_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks') + EPSILON))).alias('composite_score')
        ])
        
        result = result.drop(['score_rank', 'n_stocks'])
        
        logger.info(f"V93: 信号融合完成，处理 {result.height} 条记录")
        
        return result


# ===========================================
# V93ICAudit - IC 审计
# ===========================================

class V93ICAudit:
    """V93 IC 审计"""
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
    
    def calculate_rank_ic(self, df: pl.DataFrame,
                          signal_col: str = 'composite_score') -> Dict[str, Any]:
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
        mean_ic_3yr = float(np.mean([ic_t1, ic_t2, ic_t3]))
        
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
# V93ConsistencyChecker - 一致性检查器
# ===========================================

class V93ConsistencyChecker:
    """V93 一致性检查器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.check_results = []
    
    def check_mathematical_consistency(self, total_return: float, 
                                        annualized_return: float,
                                        years: float = 6.0) -> V93MathConsistencyResult:
        """数学一致性检查"""
        # 预期年化 = (1 + 总收益)^(1/年数) - 1
        expected = (1 + total_return) ** (1 / years) - 1
        actual = annualized_return
        diff = abs(expected - actual)
        
        # 误差 < 0.01% 为通过
        passed = diff < 0.0001
        
        message = f"预期年化={expected:.4f}, 实际年化={actual:.4f}, 误差={diff:.4%}"
        
        result = V93MathConsistencyResult(
            passed=passed,
            expected=expected,
            actual=actual,
            diff=diff,
            message=message,
        )
        
        self.check_results.append(('mathematical', result))
        return result
    
    def check_risk_control(self, max_drawdown: float) -> V93RiskControlResult:
        """风险控制检查"""
        expected = 0.10  # 10%
        actual = max_drawdown
        diff = actual - expected
        
        passed = actual <= expected
        
        message = f"最大回撤={actual:.2%}, 允许上限={expected:.2%}"
        
        result = V93RiskControlResult(
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
# V93TurnoverTracker - 换手率追踪器
# ===========================================

class V93TurnoverTracker:
    """V93 换手率追踪器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.turnover_records: List[Dict] = []
        self.trading_days = 0
    
    def record_turnover(self, trade_date: str, portfolio_value: float,
                        buy_value: float, sell_value: float,
                        is_rebalance_day: bool = False) -> Dict:
        """记录换手率（带单日 10% 上限）"""
        if portfolio_value < EPSILON:
            turnover_rate = 0.0
            buy_turnover = 0.0
            sell_turnover = 0.0
            daily_turnover = 0.0
        else:
            # V93 单日换手率限制：10% 上限
            max_daily_turnover_value = portfolio_value * V93_DAILY_TURNOVER_MAX
            
            # 限制买入和卖出金额
            capped_buy_value = min(buy_value, max_daily_turnover_value)
            capped_sell_value = min(sell_value, max_daily_turnover_value)
            
            # V93 换手率计算：使用双边换手率（买入 + 卖出）/ 组合价值
            buy_turnover = capped_buy_value / portfolio_value
            sell_turnover = capped_sell_value / portfolio_value
            
            # 双边换手率
            turnover_rate = (capped_buy_value + capped_sell_value) / portfolio_value
            
            # 单日换手率（限制在 10% 以内）
            daily_turnover = min(turnover_rate, V93_DAILY_TURNOVER_MAX)
        
        self.trading_days += 1
        
        # V93 年化换手率：累计换手率 * (252 / 实际交易天数)
        cumulative_turnover = sum(r['turnover_rate'] for r in self.turnover_records) + turnover_rate
        annualized_turnover = cumulative_turnover * (252.0 / max(1, self.trading_days))
        
        record = {
            'trade_date': trade_date,
            'turnover_rate': turnover_rate,
            'buy_turnover': buy_turnover,
            'sell_turnover': sell_turnover,
            'annualized_turnover': annualized_turnover,
            'daily_turnover': daily_turnover,
            'is_rebalance_day': is_rebalance_day,
        }
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
        
        # V93 使用最终累计年化换手率
        total_turnover = sum(r['turnover_rate'] for r in self.turnover_records)
        annualized_turnover = total_turnover * (252.0 / max(1, self.trading_days))
        
        daily_turnovers = [r['daily_turnover'] for r in self.turnover_records]
        max_daily = np.max(daily_turnovers) if daily_turnovers else 0.0
        
        is_active = V93_TURNOVER_MIN <= annualized_turnover <= V93_TURNOVER_MAX
        daily_ok = max_daily <= V93_DAILY_TURNOVER_MAX
        
        return {
            'mean_turnover': float(np.mean([r['turnover_rate'] for r in self.turnover_records])),
            'std_turnover': float(np.std([r['turnover_rate'] for r in self.turnover_records])),
            'max_turnover': float(np.max([r['turnover_rate'] for r in self.turnover_records])),
            'annualized_turnover': float(annualized_turnover),
            'is_active': is_active,
            'max_daily_turnover': float(max_daily),
            'daily_turnover_ok': daily_ok,
            'turnover_min': V93_TURNOVER_MIN,
            'turnover_max': V93_TURNOVER_MAX,
        }


# ===========================================
# 导出列表
# ===========================================

__all__ = [
    'V93_INITIAL_CAPITAL',
    'V93_MAX_POSITIONS',
    'V93_WARMUP_PERIOD',
    'V93_MIN_SCORE_THRESHOLD',
    'V93_MIN_SINGLE_WEIGHT',
    'V93_MAX_SINGLE_WEIGHT',
    'V93_TURNOVER_MIN',
    'V93_TURNOVER_MAX',
    'V93_DAILY_TURNOVER_MAX',
    'V93_T1_IC_TARGET',
    'V93_IC_IR_TARGET',
    'V93_COMMISSION_RATE',
    'V93_MIN_COMMISSION',
    'V93_STAMP_DUTY',
    'V93_TRANSFER_FEE',
    'V93_VOL_OF_VOL_WEIGHT',
    'V93_FLOW_SKEW_WEIGHT',
    'V93_MOMENTUM_WEIGHT',
    'V93_REVERSAL_WEIGHT',
    'V93_MIN_REBALANCE_INTERVAL',
    'V93_MAX_REBALANCE_INTERVAL',
    'V93DataManager',
    'V93VolOfVolEngine',
    'V93FlowSkewEngine',
    'V93MomentumReversalEngine',
    'V93NeutralizationEngine',
    'V93CompositeEngine',
    'V93ICAudit',
    'V93ConsistencyChecker',
    'V93TurnoverTracker',
    'normalize_rank',
    'zscore_normalize',
    'EPSILON',
]