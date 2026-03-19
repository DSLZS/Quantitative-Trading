"""
V41 Factor Library Module - 因子库模块

【核心功能】
1. 基础因子计算（ATR, RSRS, 趋势强度等）
2. V41 新增：二阶导动量因子（Momentum of Momentum）
3. V41 新增：板块中性化逻辑
4. 市场波动率指数（VIX 模拟）
5. 综合信号计算

作者：量化系统
版本：V41.0
日期：2026-03-19
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger

ATR_PERIOD = 20
RSRS_WINDOW = 18
MOMENTUM_WINDOW = 20
VOLATILITY_WINDOW = 20
MOM_ACCEL_WINDOW = 10
INDUSTRY_NEUTRALIZATION_ENABLED = True
INDUSTRY_ZSCORE_WINDOW = 20

# V41 优化后的因子权重（降低动量加速度权重，增加趋势强度）
DEFAULT_FACTOR_WEIGHTS = {
    'trend_strength_20': 0.25,
    'trend_strength_60': 0.20,
    'rsrs_factor': 0.20,
    'volatility_adjusted_momentum': 0.20,
    'momentum_acceleration': 0.15,  # 降低权重
}

VOLATILITY_FILTER_WINDOW = 20
VOLATILITY_FILTER_THRESHOLD = 1.5


@dataclass
class FactorConfig:
    """因子库配置"""
    atr_period: int = ATR_PERIOD
    rsrs_window: int = RSRS_WINDOW
    momentum_window: int = MOMENTUM_WINDOW
    volatility_window: int = VOLATILITY_WINDOW
    mom_accel_window: int = MOM_ACCEL_WINDOW
    industry_neutralization_enabled: bool = INDUSTRY_NEUTRALIZATION_ENABLED
    industry_zscore_window: int = INDUSTRY_ZSCORE_WINDOW
    factor_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.factor_weights is None:
            self.factor_weights = DEFAULT_FACTOR_WEIGHTS.copy()


class FactorLibrary:
    """
    V41 因子库 - 模块化设计
    
    【核心功能】
    1. 基础因子计算（ATR, RSRS, 趋势强度等）
    2. V41 新增：二阶导动量因子（Momentum of Momentum）
    3. V41 新增：板块中性化逻辑
    4. 市场波动率指数（VIX 模拟）
    5. 综合信号计算
    """
    
    EPSILON = 1e-9
    
    def __init__(self, config: FactorConfig = None):
        self.config = config or FactorConfig()
        self.factor_weights = self.config.factor_weights
        self._industry_cache: Dict[str, Set[str]] = {}
    
    def compute_all_factors(self, df: pl.DataFrame, industry_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """V41 全因子计算"""
        try:
            required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
            self._validate_columns(df, required_cols)
            
            result = df.clone().with_columns([
                pl.col('open').cast(pl.Float64, strict=False).alias('open'),
                pl.col('high').cast(pl.Float64, strict=False).alias('high'),
                pl.col('low').cast(pl.Float64, strict=False).alias('low'),
                pl.col('close').cast(pl.Float64, strict=False).alias('close'),
                pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            ])
            
            logger.info("[Step 1] Computing ATR(20)...")
            result = self._compute_atr(result, period=self.config.atr_period)
            
            logger.info("[Step 2] Computing RSRS factor...")
            result = self._compute_rsrs_factor(result)
            
            logger.info("[Step 3] Computing trend factors...")
            result = self._compute_trend_factors(result)
            
            logger.info("[Step 4] Computing volatility adjusted momentum...")
            result = self._compute_volatility_adjusted_momentum(result)
            
            logger.info("[Step 5] Computing momentum acceleration (V41 NEW)...")
            result = self._compute_momentum_acceleration(result)
            
            logger.info("[Step 6] Computing market volatility index...")
            result = self._compute_market_volatility_index(result)
            
            if self.config.industry_neutralization_enabled and industry_data is not None and not industry_data.is_empty():
                logger.info("[Step 7] Applying industry neutralization (V41 NEW)...")
                result = self._apply_industry_neutralization(result, industry_data)
            else:
                logger.info("[Step 7] Skipping industry neutralization (no data)")
                result = result.with_columns([
                    pl.lit(0.0).alias('industry_neutralized_signal'),
                    pl.lit(0.0).alias('base_signal'),
                ])
            
            logger.info("[Step 8] Computing composite signal...")
            result = self._compute_composite_signal(result)
            
            logger.info("All V41 factors computed successfully")
            return result
            
        except Exception as e:
            logger.error(f"compute_all_factors FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_columns(self, df: pl.DataFrame, required_columns: List[str]) -> bool:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def _compute_atr(self, df: pl.DataFrame, period: int = ATR_PERIOD) -> pl.DataFrame:
        """ATR (Average True Range) 计算"""
        try:
            result = df.clone()
            prev_close = pl.col('close').shift(1).over('symbol')
            tr1 = pl.col('high') - pl.col('low')
            tr2 = (pl.col('high') - prev_close).abs()
            tr3 = (pl.col('low') - prev_close).abs()
            tr = pl.max_horizontal([tr1, tr2, tr3])
            atr = tr.rolling_mean(window_size=period).over('symbol')
            
            result = result.with_columns([
                tr.alias('true_range'),
                atr.alias('atr_20'),
                prev_close.alias('prev_close')
            ])
            return result
        except Exception as e:
            logger.error(f"_compute_atr FAILED: {e}")
            raise
    
    def _compute_rsrs_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """RSRS 因子 (Resistance Support Relative Strength)"""
        try:
            result = df.clone()
            rsrs_window = self.config.rsrs_window
            high_low_spread = pl.col('high') - pl.col('low')
            spread_mean = high_low_spread.rolling_mean(window_size=rsrs_window).over('symbol')
            spread_std = high_low_spread.rolling_std(window_size=rsrs_window).over('symbol')
            rsrs_raw = (high_low_spread - spread_mean) / (spread_std + self.EPSILON)
            r_squared = 1.0 / (1.0 + spread_std)
            rsrs = rsrs_raw * r_squared * 0.5
            
            result = result.with_columns([
                high_low_spread.alias('high_low_spread'),
                spread_mean.alias('spread_mean'),
                spread_std.alias('spread_std'),
                rsrs.alias('rsrs_factor')
            ])
            return result
        except Exception as e:
            logger.error(f"_compute_rsrs_factor FAILED: {e}")
            raise
    
    def _compute_trend_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """趋势因子计算"""
        try:
            result = df.clone()
            close_20_ago = pl.col('close').shift(20).over('symbol')
            trend_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
            close_60_ago = pl.col('close').shift(60).over('symbol')
            trend_60 = (pl.col('close') - close_60_ago) / (close_60_ago + self.EPSILON)
            ma60 = pl.col('close').rolling_mean(window_size=60).over('symbol')
            
            result = result.with_columns([
                trend_20.alias('trend_strength_20'),
                trend_60.alias('trend_strength_60'),
                ma60.alias('ma60')
            ])
            return result
        except Exception as e:
            logger.error(f"_compute_trend_factors FAILED: {e}")
            raise
    
    def _compute_volatility_adjusted_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """波动率调整动量"""
        try:
            result = df.clone()
            close_20_ago = pl.col('close').shift(20).over('symbol')
            momentum_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
            returns = pl.col('close').pct_change().over('symbol')
            vol_20 = returns.rolling_std(window_size=20).over('symbol')
            vol_adj_momentum = momentum_20 / (vol_20 + self.EPSILON) * 0.5
            
            result = result.with_columns([
                momentum_20.alias('momentum_20'),
                vol_20.alias('volatility_20'),
                vol_adj_momentum.alias('volatility_adjusted_momentum')
            ])
            return result
        except Exception as e:
            logger.error(f"_compute_volatility_adjusted_momentum FAILED: {e}")
            raise
    
    def _compute_momentum_acceleration(self, df: pl.DataFrame) -> pl.DataFrame:
        """V41 新增：二阶导动量因子（Momentum of Momentum）"""
        try:
            result = df.clone()
            close_20_ago = pl.col('close').shift(20).over('symbol')
            momentum = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
            momentum_prev = momentum.shift(self.config.mom_accel_window).over('symbol')
            momentum_accel = momentum - momentum_prev
            accel_std = momentum_accel.rolling_std(window_size=20).over('symbol')
            momentum_accel_normalized = momentum_accel / (accel_std + self.EPSILON) * 0.5
            
            result = result.with_columns([
                momentum.alias('momentum_raw'),
                momentum_accel.alias('momentum_acceleration_raw'),
                momentum_accel_normalized.alias('momentum_acceleration')
            ])
            return result
        except Exception as e:
            logger.error(f"_compute_momentum_acceleration FAILED: {e}")
            raise
    
    def _compute_market_volatility_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """市场波动率指数 (VIX 模拟)"""
        try:
            result = df.clone()
            returns = pl.col('close').pct_change().over('symbol')
            stock_vol = returns.rolling_std(window_size=20, ddof=1).over('symbol')
            market_vol = stock_vol
            market_vol_mean = market_vol.rolling_mean(window_size=VOLATILITY_FILTER_WINDOW).over('symbol')
            vol_ratio = market_vol / (market_vol_mean + self.EPSILON)
            vix_sim = market_vol * 100
            
            result = result.with_columns([
                returns.alias('returns'),
                stock_vol.alias('stock_volatility'),
                market_vol.alias('market_volatility'),
                market_vol_mean.alias('market_volatility_mean'),
                vol_ratio.alias('volatility_ratio'),
                vix_sim.alias('vix_sim')
            ])
            return result
        except Exception as e:
            logger.error(f"_compute_market_volatility_index FAILED: {e}")
            raise
    
    def _apply_industry_neutralization(self, df: pl.DataFrame, industry_data: pl.DataFrame) -> pl.DataFrame:
        """V41 新增：板块中性化逻辑"""
        try:
            result = df.clone()
            
            base_signal = pl.lit(0.0)
            for factor, weight in self.factor_weights.items():
                if factor in result.columns and factor != 'momentum_acceleration':
                    factor_mean = result[factor].mean() or 0
                    factor_std = result[factor].std() or 1
                    z_factor = (pl.col(factor) - factor_mean) / (factor_std + self.EPSILON)
                    base_signal = base_signal + z_factor * weight
            
            if 'momentum_acceleration' in result.columns:
                factor_mean = result['momentum_acceleration'].mean() or 0
                factor_std = result['momentum_acceleration'].std() or 1
                z_factor = (pl.col('momentum_acceleration') - factor_mean) / (factor_std + self.EPSILON)
                base_signal = base_signal + z_factor * self.factor_weights.get('momentum_acceleration', 0.3)
            
            result = result.with_columns([base_signal.alias('base_signal')])
            
            if 'industry_name' in industry_data.columns:
                result = result.join(
                    industry_data.select(['symbol', 'trade_date', 'industry_name']),
                    on=['symbol', 'trade_date'],
                    how='left'
                )
                result = result.with_columns([
                    pl.when(pl.col('industry_name').is_null())
                    .then(pl.lit('Unknown'))
                    .otherwise(pl.col('industry_name'))
                    .alias('industry_name')
                ])
                
                industry_mean = pl.col('base_signal').mean().over(['trade_date', 'industry_name'])
                industry_std = pl.col('base_signal').std().over(['trade_date', 'industry_name'])
                industry_zscore = (pl.col('base_signal') - industry_mean) / (industry_std + self.EPSILON)
                industry_momentum = industry_mean - industry_mean.shift(20).over(['trade_date', 'industry_name'])
                
                industry_adjustment = pl.when(industry_momentum >= 0).then(pl.lit(1.0)).otherwise(1.0 + industry_momentum)
                neutralized_signal = industry_zscore * industry_adjustment
                
                result = result.with_columns([
                    industry_mean.alias('industry_mean_signal'),
                    industry_momentum.alias('industry_momentum'),
                    industry_zscore.alias('industry_zscore'),
                    neutralized_signal.alias('industry_neutralized_signal')
                ])
            else:
                result = result.with_columns([
                    pl.lit('Unknown').alias('industry_name'),
                    pl.lit(0.0).alias('industry_mean_signal'),
                    pl.lit(0.0).alias('industry_momentum'),
                    pl.lit(0.0).alias('industry_zscore'),
                    pl.col('base_signal').alias('industry_neutralized_signal')
                ])
            
            return result
        except Exception as e:
            logger.error(f"_apply_industry_neutralization FAILED: {e}")
            raise
    
    def _compute_composite_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算加权综合信号"""
        try:
            result = df.clone()
            
            if 'industry_neutralized_signal' in result.columns:
                signal_for_ranking = pl.col('industry_neutralized_signal')
            else:
                signal = pl.lit(0.0)
                for factor, weight in self.factor_weights.items():
                    if factor in result.columns:
                        factor_mean = result[factor].mean() or 0
                        factor_std = result[factor].std() or 1
                        z_factor = (pl.col(factor) - factor_mean) / (factor_std + self.EPSILON)
                        signal = signal + z_factor * weight
                signal_for_ranking = signal
            
            vol_filter = pl.col('volatility_ratio') <= VOLATILITY_FILTER_THRESHOLD
            ma_filter = pl.col('close') >= pl.col('ma60')
            entry_allowed = vol_filter & ma_filter
            
            result = result.with_columns([
                signal_for_ranking.alias('signal'),
                vol_filter.alias('vol_filter_pass'),
                ma_filter.alias('ma_filter_pass'),
                entry_allowed.alias('entry_allowed')
            ])
            
            result = result.with_columns([
                pl.when(pl.col('entry_allowed'))
                .then(pl.col('signal').rank('ordinal', descending=True).over('trade_date'))
                .otherwise(9999)
                .cast(pl.Int64)
                .alias('rank')
            ])
            
            return result
        except Exception as e:
            logger.error(f"_compute_composite_signal FAILED: {e}")
            raise
    
    def get_factor_weights(self) -> Dict[str, float]:
        """获取因子权重"""
        return self.factor_weights.copy()
    
    def update_factor_weights(self, weights: Dict[str, float]):
        """更新因子权重"""
        self.factor_weights.update(weights)
        logger.info(f"Factor weights updated: {self.factor_weights}")