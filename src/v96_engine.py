"""
V96 Engine - 强制中性化下的 Alpha 深度挖掘

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

作者：量化系统
版本：V96.0
日期：2026-03-31
"""

import traceback
import time
import math
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger

# 导入 V96 核心模块
from src.core.v96_core import (
    V96DataManager,
    V96IndustryNeutralizationEngine,
    V96SizeNeutralizationEngine,
    V96LiquidityFilterEngine,
    V96IntradaySkewnessEngine,
    V96NonLinearInteractionEngine,
    V96AlphaFusion,
    V96AlphaWeightEngine,
    V96ICAudit,
    V96TurnoverTracker,
    V96PortfolioTracker,
    V96DynamicRebalanceEngine,
    V96_INITIAL_CAPITAL,
    V96_MAX_POSITIONS,
    V96_WARMUP_PERIOD,
    V96_MIN_SCORE_THRESHOLD,
    V96_MIN_SINGLE_WEIGHT,
    V96_MAX_SINGLE_WEIGHT,
    V96_TURNOVER_MIN,
    V96_TURNOVER_MAX,
    V96_DAILY_TURNOVER_MAX,
    V96_T1_IC_TARGET,
    V96_COMMISSION_RATE,
    V96_MIN_COMMISSION,
    V96_STAMP_DUTY,
    V96_TRANSFER_FEE,
    V96_HALF_LIFE_LAGS,
    V96_RESIDUAL_WEIGHT,
    V96_FLOW_WEIGHT,
    V96_INTERACTION_WEIGHT,
    V96_SKEWNESS_WEIGHT,
    V96_SIZE_NEUTRALIZATION,
    V96_INDUSTRY_NEUTRALIZATION,
    V96_LIQUIDITY_FILTER,
    V96_FILTER_ST,
    V96_NEUTRALIZATION_WINDOW,
    EPSILON,
)

# 导入数据库管理器
try:
    from src.db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    try:
        from db_manager import DatabaseManager, get_db
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False


# ===========================================
# V96 引擎配置
# ===========================================

class V96EngineConfig:
    """V96 引擎配置"""
    
    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = V96_INITIAL_CAPITAL,
        max_positions: int = V96_MAX_POSITIONS,
        warmup_period: int = V96_WARMUP_PERIOD,
        commission_rate: float = V96_COMMISSION_RATE,
        min_commission: float = V96_MIN_COMMISSION,
        stamp_duty: float = V96_STAMP_DUTY,
        transfer_fee: float = V96_TRANSFER_FEE,
        oos_years: List[str] = None,
        min_score_threshold: float = V96_MIN_SCORE_THRESHOLD,
        min_single_weight: float = V96_MIN_SINGLE_WEIGHT,
        max_single_weight: float = V96_MAX_SINGLE_WEIGHT,
        # V96 强制配置（严禁禁用）
        enable_industry_neutralization: bool = True,  # 强制开启
        enable_size_neutralization: bool = True,      # 强制开启
        enable_liquidity_filter: bool = True,         # 强制开启
        filter_st: bool = True,                       # 强制过滤 ST
        enable_intraday_skewness: bool = True,        # 新增因子
        enable_nonlinear_interaction: bool = True,    # 非线性交互
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.warmup_period = warmup_period
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty = stamp_duty
        self.transfer_fee = transfer_fee
        self.oos_years = oos_years or ["2019", "2021", "2024"]
        self.min_score_threshold = min_score_threshold
        self.min_single_weight = min_single_weight
        self.max_single_weight = max_single_weight
        
        # V96 强制配置
        self.enable_industry_neutralization = enable_industry_neutralization
        self.enable_size_neutralization = enable_size_neutralization
        self.enable_liquidity_filter = enable_liquidity_filter
        self.filter_st = filter_st
        self.enable_intraday_skewness = enable_intraday_skewness
        self.enable_nonlinear_interaction = enable_nonlinear_interaction
        
        # 验证强制配置
        if not self.enable_industry_neutralization:
            logger.error("V96: 行业中性化必须开启，否则回测无效！")
        if not self.enable_size_neutralization:
            logger.error("V96: 市值中性化必须开启，否则回测无效！")
        if not self.enable_liquidity_filter:
            logger.error("V96: 流动性过滤必须开启，否则回测无效！")


# ===========================================
# V96 引擎
# ===========================================

class V96Engine:
    """V96 回测引擎"""
    
    def __init__(self, config: V96EngineConfig = None, db=None):
        self.config = config or V96EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V96: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        self.data_manager = V96DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
        })
        
        # V96 新增：行业中性化（强制）
        self.industry_neutralization = V96IndustryNeutralizationEngine(
            db=self.db,
            config={
                'industry_neutralization': self.config.enable_industry_neutralization,
                'window': V96_NEUTRALIZATION_WINDOW,
            }
        ) if self.config.enable_industry_neutralization else None
        
        # V96 新增：市值中性化（强制）
        self.size_neutralization = V96SizeNeutralizationEngine(
            config={
                'size_neutralization': self.config.enable_size_neutralization,
            }
        ) if self.config.enable_size_neutralization else None
        
        # V96 新增：流动性过滤（强制）
        self.liquidity_filter = V96LiquidityFilterEngine(
            config={
                'liquidity_filter': self.config.enable_liquidity_filter,
                'liquidity_percentile': 10,
                'filter_st': self.config.filter_st,
            }
        ) if self.config.enable_liquidity_filter else None
        
        # V96 新增：日内偏度因子
        self.intraday_skewness = V96IntradaySkewnessEngine() if self.config.enable_intraday_skewness else None
        
        # V96 新增：非线性交互
        self.nonlinear_interaction = V96NonLinearInteractionEngine() if self.config.enable_nonlinear_interaction else None
        
        self.alpha_fusion = V96AlphaFusion(db=self.db, config={
            'fusion_lags': V96_HALF_LIFE_LAGS,
        })
        self.alpha_weight = V96AlphaWeightEngine(config={
            'min_score': self.config.min_score_threshold,
            'min_weight': self.config.min_single_weight,
            'max_weight': self.config.max_single_weight,
        })
        self.ic_audit = V96ICAudit(db=self.db)
        self.turnover_tracker = V96TurnoverTracker()
        self.portfolio_tracker = V96PortfolioTracker(
            initial_capital=self.config.initial_capital
        )
        self.dynamic_rebalance = V96DynamicRebalanceEngine()
        
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        self.trade_records: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        self.rebalance_dates: List[str] = []
        
        logger.info("=" * 70)
        logger.info("V96 Engine 初始化完成")
        logger.info("=" * 70)
        logger.info(f"V96: 初始资金={self.config.initial_capital:,.2f} (严禁修改)")
        logger.info(f"V96: 最大持仓数={self.config.max_positions}")
        logger.info(f"V96: 评分门槛={self.config.min_score_threshold}")
        logger.info("=" * 70)
        logger.info("V96 强制配置（红线）:")
        logger.info(f"  行业中性化：{'✓ ENABLED' if self.config.enable_industry_neutralization else '✗ DISABLED (INVALID!)'}")
        logger.info(f"  市值中性化：{'✓ ENABLED' if self.config.enable_size_neutralization else '✗ DISABLED (INVALID!)'}")
        logger.info(f"  流动性过滤：{'✓ ENABLED' if self.config.enable_liquidity_filter else '✗ DISABLED (INVALID!)'}")
        logger.info(f"  ST 股过滤：{'✓ ENABLED' if self.config.filter_st else '✗ DISABLED'}")
        logger.info("=" * 70)
        logger.info("V96 新增因子:")
        logger.info(f"  日内偏度因子：{'✓ ENABLED' if self.config.enable_intraday_skewness else '✗ DISABLED'}")
        logger.info(f"  非线性交互：{'✓ ENABLED' if self.config.enable_nonlinear_interaction else '✗ DISABLED'}")
        logger.info("=" * 70)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V96 强制中性化下的 Alpha 深度挖掘引擎启动")
        logger.info("=" * 70)
        
        # 验证强制配置
        if not self.config.enable_industry_neutralization:
            logger.error("V96: 行业中性化未开启，回测无效！")
            return self._empty_result()
        if not self.config.enable_size_neutralization:
            logger.error("V96: 市值中性化未开启，回测无效！")
            return self._empty_result()
        if not self.config.enable_liquidity_filter:
            logger.error("V96: 流动性过滤未开启，回测无效！")
            return self._empty_result()
        
        if self.db is None:
            logger.error("V96: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查
            logger.info("V96: [1/8] 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V96: [2/8] 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V96: 未加载到任何数据")
                return self._empty_result()
            
            for year in self.config.oos_years:
                df_year = df.filter(
                    pl.col('trade_date').cast(pl.Utf8).str.starts_with(year)
                )
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V96: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算 V90 因子信号
            logger.info("V96: [3/8] 开始计算 V90 因子信号...")
            df_with_signals = self._compute_v90_signals(df)
            
            # 4. 计算日内偏度因子（V96 新增）
            if self.config.enable_intraday_skewness:
                logger.info("V96: [4/8] 开始计算日内偏度因子...")
                df_with_skewness = self._compute_intraday_skewness(df_with_signals)
                df_with_signals = df_with_skewness
            else:
                logger.info("V96: [4/8] 跳过日内偏度因子计算")
            
            # 5. 非线性交互（V96 新增）
            if self.config.enable_nonlinear_interaction and 'intraday_skewness_score' in df_with_signals.columns:
                logger.info("V96: [5/8] 开始非线性交互...")
                df_with_interaction = self._compute_nonlinear_interaction(df_with_signals)
                df_with_signals = df_with_interaction
            
            # 6. 计算综合评分
            logger.info("V96: [6/8] 开始计算综合评分...")
            df_with_signals = self._compute_composite_score_v96(df_with_signals)
            
            # 7. 行业中性化（V96 强制）
            if self.industry_neutralization:
                logger.info("V96: [7/8] 开始行业中性化 (强制)...")
                df_with_neutralization = self.industry_neutralization.compute_industry_neutralization(
                    df_with_signals, signal_col='composite_score'
                )
                df_with_signals = df_with_neutralization.with_columns([
                    pl.when(pl.col('neutralized_signal').is_not_null())
                    .then(pl.col('neutralized_signal'))
                    .otherwise(pl.col('composite_score'))
                    .alias('composite_score')
                ])
            else:
                logger.error("V96: 行业中性化未执行，回测无效！")
            
            # 8. 市值中性化（V96 强制）
            if self.size_neutralization:
                logger.info("V96: [8/8] 开始市值中性化 (强制)...")
                df_with_size_neut = self.size_neutralization.compute_size_neutralization(
                    df_with_signals, signal_col='composite_score'
                )
                df_with_signals = df_with_size_neut.with_columns([
                    pl.when(pl.col('neutralized_signal').is_not_null())
                    .then(pl.col('neutralized_signal'))
                    .otherwise(pl.col('composite_score'))
                    .alias('composite_score')
                ])
            else:
                logger.error("V96: 市值中性化未执行，回测无效！")
            
            # 流动性过滤（V96 强制）
            if self.liquidity_filter:
                logger.info("V96: 开始流动性过滤 (强制)...")
                df_with_filter = self.liquidity_filter.apply_liquidity_filter(df_with_signals)
                df_with_signals = df_with_filter
            
            # 半衰期融合
            logger.info("V96: 开始半衰期融合...")
            df_with_fusion = self.alpha_fusion.compute_fusion_signal(
                df_with_signals, signal_col='composite_score'
            )
            
            # Alpha 权重与 IC 审计
            logger.info("V96: 开始 Alpha 权重计算与 IC 审计...")
            df_with_weights = self.alpha_weight.compute_alpha_weights(
                df_with_fusion, score_col='fused_signal'
            )
            
            ic_audit_results = self.ic_audit.calculate_rank_ic(
                df_with_weights, signal_col='fused_signal'
            )
            
            logger.info(f"V96: T+1 Rank IC = {ic_audit_results['ic_t1']:.4f} (目标 > {V96_T1_IC_TARGET})")
            logger.info(f"V96: T+2 Rank IC = {ic_audit_results['ic_t2']:.4f}")
            logger.info(f"V96: T+3 Rank IC = {ic_audit_results['ic_t3']:.4f}")
            
            # 执行回测交易
            logger.info("V96: 开始执行回测交易...")
            trade_results = self._execute_backtest(df_with_weights)
            
            # 生成报告
            logger.info("V96: 生成审计报告...")
            audit_report = self._generate_audit_report(
                data_integrity_results,
                ic_audit_results,
                trade_results,
            )
            
            result = {
                'data_integrity': data_integrity_results,
                'ic_audit': ic_audit_results,
                'trade_results': trade_results,
                'audit_report': audit_report,
                'trade_records': self.trade_records,
                'daily_snapshots': self.daily_snapshots,
                'rebalance_dates': self.rebalance_dates,
                'neutralization_status': {
                    'industry_neutralization': self.config.enable_industry_neutralization,
                    'size_neutralization': self.config.enable_size_neutralization,
                    'liquidity_filter': self.config.enable_liquidity_filter,
                    'filter_st': self.config.filter_st,
                }
            }
            
            logger.info("=" * 70)
            logger.info("V96 回测完成")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"V96 回测失败 - {e}")
            logger.error(traceback.format_exc())
            return self._empty_result()
    
    def _check_data_integrity(self) -> Dict[str, Dict[str, Any]]:
        """检查数据完整性"""
        results = {}
        for year in self.config.oos_years:
            passed, message, stats = self.data_manager.check_data_integrity(year)
            results[year] = {
                'passed': passed,
                'message': message,
                'stats': stats,
            }
            if passed:
                logger.info(f"V96: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V96: {year}年数据检查失败 - {message}")
        return results
    
    def _load_data(self) -> pl.DataFrame:
        """加载数据"""
        all_dfs = []
        for year in self.config.oos_years:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            try:
                df = self.data_manager.load_data(start_date, end_date)
                if not df.is_empty():
                    all_dfs.append(df)
                    logger.info(f"V96: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V96: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V96: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _compute_v90_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 V90 因子信号"""
        result = df.clone()
        
        result = result.with_columns([
            pl.col('open').cast(pl.Float64, strict=False).alias('open'),
            pl.col('high').cast(pl.Float64, strict=False).alias('high'),
            pl.col('low').cast(pl.Float64, strict=False).alias('low'),
            pl.col('close').cast(pl.Float64, strict=False).alias('close'),
            pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            pl.col('amount').cast(pl.Float64, strict=False).alias('amount'),
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
        ])
        
        result = self._compute_refined_residual(result)
        result = self._compute_smart_flow(result)
        result = self._compute_vol_price_interaction(result)
        
        return result
    
    def _compute_refined_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Refined Residual 因子"""
        result = df.clone()
        window = 10
        
        result = result.with_columns([
            pl.col('volume').fill_null(0).alias('volume_filled'),
            pl.col('close').fill_null(0).alias('close_filled'),
            pl.col('industry_code').cast(pl.Utf8, strict=False).fill_null('Unknown').alias('industry_code'),
        ])
        
        result = result.with_columns([
            ((pl.col('close_filled').shift(1) - pl.col('close_filled').shift(window + 1)) / 
             (pl.col('close_filled').shift(window + 1) + EPSILON)).alias('stock_return_10d')
        ])
        
        market_median = result.group_by('trade_date').agg([
            pl.col('pct_chg').median().alias('market_return_10d')
        ])
        
        result = result.join(
            market_median.select(['trade_date', 'market_return_10d']),
            on='trade_date',
            how='left'
        )
        
        result = result.with_columns([
            (pl.col('stock_return_10d') - pl.col('market_return_10d')).alias('residual_return')
        ])
        
        result = result.with_columns([
            pl.col('volume_filled').rolling_sum(window_size=20).over('symbol').alias('volume_ma_20')
        ])
        
        result = result.with_columns([
            (pl.col('volume_filled') / (pl.col('volume_ma_20') / 20 + EPSILON)).alias('volume_ratio')
        ])
        
        result = result.with_columns([
            (pl.col('residual_return') * pl.col('volume_ratio')).alias('residual_volume_confirmed')
        ])
        
        result = result.with_columns([
            pl.col('residual_volume_confirmed').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + EPSILON))).alias('refined_residual_score')
        ])
        
        result = result.drop(['volume_filled', 'close_filled', 'volume_ma_20'])
        
        return result
    
    def _compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Smart Flow 因子"""
        result = df.clone()
        window = 10
        
        result = result.with_columns([
            (pl.col('close') - pl.col('open')).alias('price_change'),
            pl.col('volume').fill_null(0.0).alias('volume_filled')
        ])
        
        result = result.with_columns([
            (pl.col('volume_filled') * pl.col('price_change')).alias('volume_weighted_change')
        ])
        
        result = result.sort(['symbol', 'trade_date'])
        
        result = result.with_columns([
            pl.col('volume_weighted_change').rolling_sum(window_size=window).over('symbol').alias('flow_sum'),
            pl.col('volume_filled').rolling_sum(window_size=window).over('symbol').alias('volume_sum')
        ])
        
        result = result.with_columns([
            (pl.col('flow_sum') / (pl.col('volume_sum') + EPSILON)).alias('smart_flow_raw')
        ])
        
        result = result.with_columns([
            pl.col('smart_flow_raw').rank('ordinal', descending=True).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_flow')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('flow_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON))).alias('smart_flow_score')
        ])
        
        return result
    
    def _compute_vol_price_interaction(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Vol_Price_Interaction 因子"""
        result = df.clone()
        
        if 'residual_rank' not in result.columns:
            result = result.with_columns([
                pl.col('refined_residual_score').rank('ordinal', descending=True).over('trade_date').alias('residual_rank')
            ])
        
        if 'flow_rank' not in result.columns:
            result = result.with_columns([
                pl.col('smart_flow_score').rank('ordinal', descending=True).over('trade_date').alias('flow_rank')
            ])
        
        if 'n_stocks' not in result.columns:
            result = result.with_columns([
                pl.col('symbol').count().over('trade_date').alias('n_stocks')
            ])
        
        if 'n_stocks_flow' not in result.columns:
            result = result.with_columns([
                pl.col('symbol').count().over('trade_date').alias('n_stocks_flow')
            ])
        
        result = result.with_columns([
            (pl.col('residual_rank') / (pl.col('n_stocks').cast(pl.Float64) + EPSILON)).alias('residual_rank_norm'),
            (pl.col('flow_rank') / (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON)).alias('flow_rank_norm'),
        ])
        
        result = result.with_columns([
            (pl.col('residual_rank_norm') * pl.col('flow_rank_norm')).alias('interaction_raw')
        ])
        
        result = result.with_columns([
            pl.col('interaction_raw').rank('ordinal', descending=True).over('trade_date').alias('interaction_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_interaction')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('interaction_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_interaction').cast(pl.Float64) + EPSILON))).alias('interaction_score')
        ])
        
        result = result.with_columns([
            (0.7 * pl.col('interaction_score') + 0.3 * 50.0).alias('vol_price_interaction_score')
        ])
        
        return result
    
    def _compute_intraday_skewness(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算日内偏度因子（V96 新增）"""
        if self.intraday_skewness is None:
            return df
        
        result = self.intraday_skewness.compute_intraday_skewness(df)
        return result
    
    def _compute_nonlinear_interaction(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算非线性交互（V96 新增）"""
        if self.nonlinear_interaction is None:
            return df
        
        result = self.nonlinear_interaction.compute_nonlinear_interaction(
            df,
            residual_col='refined_residual_score',
            skewness_col='intraday_skewness_score'
        )
        return result
    
    def _compute_composite_score_v96(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 V96 综合评分"""
        result = df.clone()
        
        for col, default in [
            ('refined_residual_score', 50.0),
            ('smart_flow_score', 50.0),
            ('vol_price_interaction_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # V96: 使用新的权重配置，丢弃量价背离因子
        # V96_RESIDUAL_WEIGHT=0.30, V96_FLOW_WEIGHT=0.20, V96_INTERACTION_WEIGHT=0.40
        # V96_SKEWNESS_WEIGHT=0.10（新增）
        
        if 'intraday_skewness_score' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('intraday_skewness_score')])
        
        if 'nonlinear_interaction_percentile' in result.columns:
            # 如果进行了非线性交互，使用交互后的分数代替残差动量
            result = result.with_columns([
                (V96_RESIDUAL_WEIGHT * pl.col('nonlinear_interaction_percentile') + 
                 V96_FLOW_WEIGHT * pl.col('smart_flow_score') +
                 V96_INTERACTION_WEIGHT * pl.col('vol_price_interaction_score') +
                 V96_SKEWNESS_WEIGHT * pl.col('intraday_skewness_score')).alias('composite_score')
            ])
        else:
            result = result.with_columns([
                (V96_RESIDUAL_WEIGHT * pl.col('refined_residual_score') + 
                 V96_FLOW_WEIGHT * pl.col('smart_flow_score') +
                 V96_INTERACTION_WEIGHT * pl.col('vol_price_interaction_score') +
                 V96_SKEWNESS_WEIGHT * pl.col('intraday_skewness_score')).alias('composite_score')
            ])
        
        return result
    
    def _execute_backtest(self, df: pl.DataFrame) -> Dict[str, Any]:
        """执行回测交易"""
        logger.info("V96: 开始执行回测交易...")
        
        df = df.sort(['trade_date', 'symbol'])
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        warmup_cutoff = unique_dates[:min(V96_WARMUP_PERIOD, len(unique_dates))]
        trade_dates = [d for d in unique_dates if d not in warmup_cutoff]
        
        logger.info(f"V96: 热身期 {len(warmup_cutoff)} 天，交易期 {len(trade_dates)} 天")
        
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.positions = {}
        self.trade_records = []
        self.daily_snapshots = []
        self.rebalance_dates = []
        
        prev_date = None
        total_position_count = 0
        snapshot_count = 0
        
        for i, trade_date in enumerate(trade_dates):
            is_rebalance_day = self._should_rebalance_today(trade_date, df)
            
            day_df = df.filter(pl.col('trade_date') == trade_date)
            
            if day_df.is_empty():
                continue
            
            # V96: 应用流动性过滤
            if self.config.enable_liquidity_filter and 'is_filtered' in day_df.columns:
                day_df = day_df.filter(
                    (pl.col('is_filtered') == False) | pl.col('is_filtered').is_null()
                )
            
            price_map = dict(zip(
                day_df['symbol'].to_list(),
                day_df['close'].to_list()
            ))
            
            # 更新持仓价格
            for symbol, position in self.positions.items():
                if symbol in price_map:
                    position['current_price'] = price_map[symbol]
                    position['pnl'] = (price_map[symbol] - position['entry_price']) * position['quantity']
            
            # 计算组合价值
            if self.positions:
                position_value = sum(
                    p.get('current_price', 0) * p.get('quantity', 0) 
                    for p in self.positions.values()
                )
            else:
                position_value = 0.0
            self.portfolio_value = self.cash + position_value
            
            buy_value = 0.0
            sell_value = 0.0
            
            if is_rebalance_day:
                self.rebalance_dates.append(trade_date)
                logger.debug(f"V96: {trade_date} 调仓")
                
                valid_stocks = day_df.filter(
                    (pl.col('alpha_weight').is_not_null()) &
                    (pl.col('alpha_weight') > EPSILON)
                ).sort('alpha_weight', descending=True)
                
                if valid_stocks.is_empty():
                    valid_stocks = day_df.filter(
                        (pl.col('fused_signal').is_not_null()) &
                        (pl.col('fused_signal').is_finite())
                    ).sort('fused_signal', descending=True)
                
                target_positions = self._calculate_target_positions(
                    valid_stocks, self.portfolio_value
                )
                
                buy_value, sell_value = self._execute_trades(
                    trade_date, target_positions, price_map
                )
            
            self.turnover_tracker.record_turnover(
                trade_date, self.portfolio_value, buy_value, sell_value,
                is_rebalance_day=is_rebalance_day
            )
            
            # 计算日收益率
            if prev_date and self.daily_snapshots:
                prev_value = self.daily_snapshots[-1]['total_value']
                daily_return = (self.portfolio_value - prev_value) / prev_value if prev_value > EPSILON else 0.0
            else:
                daily_return = 0.0
            
            total_position_count += len(self.positions)
            snapshot_count += 1
            
            snapshot = {
                'trade_date': trade_date,
                'total_value': self.portfolio_value,
                'cash': self.cash,
                'position_value': sum(
                    p.get('current_price', 0) * p.get('quantity', 0) 
                    for p in self.positions.values()
                ) if self.positions else 0.0,
                'position_count': len(self.positions),
                'daily_return': daily_return,
                'turnover_rate': buy_value / self.portfolio_value if self.portfolio_value > EPSILON else 0.0,
            }
            self.daily_snapshots.append(snapshot)
            
            prev_date = trade_date
            
            if (i + 1) % 50 == 0:
                logger.info(f"V96: 处理 {i + 1}/{len(trade_dates)} 天，组合价值={self.portfolio_value:,.2f}, 持仓数={len(self.positions)}")
        
        avg_position_count = total_position_count / max(1, snapshot_count)
        
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        turnover_summary = self.turnover_tracker.get_turnover_summary()
        max_drawdown = self._calculate_max_drawdown()
        
        annual_returns = self._calculate_annual_returns()
        
        result = {
            'total_return': total_return,
            'final_value': self.portfolio_value,
            'max_drawdown': max_drawdown,
            'annualized_turnover': turnover_summary['annualized_turnover'],
            'total_trading_days': len(trade_dates),
            'total_trades': len(self.trade_records),
            'rebalance_count': len(self.rebalance_dates),
            'avg_position_count': avg_position_count,
            'annual_returns': annual_returns,
            'avg_annual_return': np.mean(list(annual_returns.values())) if annual_returns else 0.0,
        }
        
        logger.info(f"V96: 回测完成 - 总收益={total_return:.2%}, 年化换手={turnover_summary['annualized_turnover']:.2%}")
        logger.info(f"V96: 调仓次数={len(self.rebalance_dates)}, 最大回撤={max_drawdown:.2%}")
        logger.info(f"V96: 平均持仓数={avg_position_count:.1f}")
        
        return result
    
    def _should_rebalance_today(self, trade_date: str, df: pl.DataFrame) -> bool:
        """判断是否应该调仓"""
        if not self.rebalance_dates:
            return True
        
        try:
            last_date_str = self.rebalance_dates[-1]
            if hasattr(last_date_str, 'strftime'):
                last_date_str = last_date_str.strftime('%Y-%m-%d')
            
            last_date = datetime.strptime(last_date_str, "%Y-%m-%d")
            
            if hasattr(trade_date, 'strftime'):
                trade_date_str = trade_date.strftime('%Y-%m-%d')
            else:
                trade_date_str = str(trade_date)
            
            curr_date = datetime.strptime(trade_date_str, "%Y-%m-%d")
            days_diff = (curr_date - last_date).days
        except Exception:
            days_diff = 30
        
        # 每 5 天调仓一次
        if days_diff >= 5:
            logger.debug(f"V96: {trade_date} 达到调仓间隔 ({days_diff}天)")
            return True
        
        return False
    
    def _calculate_target_positions(self, valid_stocks: pl.DataFrame, 
                                     portfolio_value: float) -> Dict[str, float]:
        """计算目标持仓"""
        if valid_stocks.is_empty():
            return {}
        
        max_stocks = min(40, self.config.max_positions)
        top_stocks = valid_stocks.head(max_stocks)
        
        target_positions = {}
        
        sorted_stocks = top_stocks.sort('fused_signal', descending=True)
        stock_list = sorted_stocks.to_dicts()
        
        n_stocks = len(stock_list)
        base_weight = 1.0 / n_stocks if n_stocks > 0 else 0.025
        
        for i, row in enumerate(stock_list):
            symbol = row['symbol']
            rank_weight = base_weight * (1.0 + 0.1 - (i / max(1, n_stocks - 1)) * 0.2)
            target_positions[symbol] = max(0.015, min(0.04, rank_weight))
        
        total_weight = sum(target_positions.values())
        if total_weight > EPSILON:
            target_positions = {k: v / total_weight for k, v in target_positions.items()}
        
        return target_positions
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        if not self.daily_snapshots:
            return 0.0
        
        peak = self.config.initial_capital
        for snapshot in self.daily_snapshots:
            if snapshot['total_value'] > peak:
                peak = snapshot['total_value']
        
        if peak < EPSILON:
            return 0.0
        
        return (peak - self.portfolio_value) / peak
    
    def _execute_trades(self, trade_date: str, target_positions: Dict[str, float],
                        price_map: Dict[str, float]) -> Tuple[float, float]:
        """执行交易"""
        buy_value = 0.0
        sell_value = 0.0
        
        # 卖出不在目标持仓中的股票
        symbols_to_sell = set(self.positions.keys()) - set(target_positions.keys())
        
        for symbol in symbols_to_sell:
            position = self.positions[symbol]
            if symbol in price_map:
                sell_price = price_map[symbol]
                sell_amount = sell_price * position['quantity']
                
                commission = max(sell_amount * self.config.commission_rate, self.config.min_commission)
                stamp_duty = sell_amount * self.config.stamp_duty
                transfer_fee = sell_amount * self.config.transfer_fee
                total_fees = commission + stamp_duty + transfer_fee
                
                self.cash += sell_amount - total_fees
                sell_value += sell_amount
                
                del self.positions[symbol]
                
                self.trade_records.append({
                    'trade_date': trade_date,
                    'symbol': symbol,
                    'action': 'sell',
                    'price': sell_price,
                    'quantity': position['quantity'],
                    'amount': sell_amount,
                    'fees': total_fees,
                })
        
        # 买入或调整目标持仓
        for symbol, target_weight in target_positions.items():
            if symbol not in price_map:
                continue
            
            buy_price = price_map[symbol]
            target_value = self.portfolio_value * target_weight
            
            if symbol in self.positions:
                position = self.positions[symbol]
                current_value = position['current_price'] * position['quantity']
                
                diff_value = target_value - current_value
                
                if abs(diff_value) > buy_price:
                    if diff_value > 0:
                        buy_quantity = int(diff_value / buy_price)
                        if buy_quantity > 0:
                            buy_amount = buy_quantity * buy_price
                            commission = max(buy_amount * self.config.commission_rate, self.config.min_commission)
                            transfer_fee = buy_amount * self.config.transfer_fee
                            total_fees = commission + transfer_fee
                            
                            if self.cash >= buy_amount + total_fees:
                                self.cash -= buy_amount + total_fees
                                buy_value += buy_amount
                                
                                position['quantity'] += buy_quantity
                                position['weight'] = target_weight
                                
                                self.trade_records.append({
                                    'trade_date': trade_date,
                                    'symbol': symbol,
                                    'action': 'buy',
                                    'price': buy_price,
                                    'quantity': buy_quantity,
                                    'amount': buy_amount,
                                    'fees': total_fees,
                                })
                    else:
                        sell_quantity = int(abs(diff_value) / buy_price)
                        if sell_quantity > 0 and sell_quantity <= position['quantity']:
                            sell_amount = sell_quantity * buy_price
                            commission = max(sell_amount * self.config.commission_rate, self.config.min_commission)
                            stamp_duty = sell_amount * self.config.stamp_duty
                            transfer_fee = sell_amount * self.config.transfer_fee
                            total_fees = commission + stamp_duty + transfer_fee
                            
                            self.cash += sell_amount - total_fees
                            sell_value += sell_amount
                            
                            position['quantity'] -= sell_quantity
                            position['weight'] = target_weight
                            
                            self.trade_records.append({
                                'trade_date': trade_date,
                                'symbol': symbol,
                                'action': 'sell',
                                'price': buy_price,
                                'quantity': sell_quantity,
                                'amount': sell_amount,
                                'fees': total_fees,
                            })
            else:
                buy_quantity = int(target_value / buy_price)
                if buy_quantity > 0:
                    buy_amount = buy_quantity * buy_price
                    commission = max(buy_amount * self.config.commission_rate, self.config.min_commission)
                    transfer_fee = buy_amount * self.config.transfer_fee
                    total_fees = commission + transfer_fee
                    
                    if self.cash >= buy_amount + total_fees:
                        self.cash -= buy_amount + total_fees
                        buy_value += buy_amount
                        
                        self.positions[symbol] = {
                            'symbol': symbol,
                            'entry_date': trade_date,
                            'entry_price': buy_price,
                            'quantity': buy_quantity,
                            'weight': target_weight,
                            'current_price': buy_price,
                            'pnl': 0.0,
                        }
                        
                        self.trade_records.append({
                            'trade_date': trade_date,
                            'symbol': symbol,
                            'action': 'buy',
                            'price': buy_price,
                            'quantity': buy_quantity,
                            'amount': buy_amount,
                            'fees': total_fees,
                        })
        
        return buy_value, sell_value
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.daily_snapshots:
            return 0.0
        
        peak = self.config.initial_capital
        max_dd = 0.0
        
        for snapshot in self.daily_snapshots:
            value = snapshot['total_value']
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > EPSILON else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_annual_returns(self) -> Dict[str, float]:
        """计算各年度收益"""
        annual_returns = {}
        
        for year in self.config.oos_years:
            snapshots_year = []
            for s in self.daily_snapshots:
                trade_date = s['trade_date']
                if hasattr(trade_date, 'strftime'):
                    date_str = trade_date.strftime('%Y')
                else:
                    date_str = str(trade_date)
                if date_str.startswith(year):
                    snapshots_year.append(s)
            
            if len(snapshots_year) >= 2:
                start_value = snapshots_year[0]['total_value']
                end_value = snapshots_year[-1]['total_value']
                year_return = (end_value - start_value) / start_value if start_value > EPSILON else 0.0
                annual_returns[year] = year_return
        
        return annual_returns
    
    def _generate_audit_report(self, data_integrity: Dict, ic_audit: Dict,
                                trade_results: Dict) -> str:
        """生成审计报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("《V96 强制中性化下的 Alpha 深度挖掘审计报告》")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append("1. 数据完整性审计")
        lines.append("   " + "-" * 50)
        for year, result in data_integrity.items():
            status = "✓" if result['passed'] else "✗"
            lines.append(f"   {year}年：{status} {result['message']}")
        lines.append("")
        
        lines.append("2. IC 审计")
        lines.append("   " + "-" * 50)
        lines.append(f"   T+1 Rank IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V96_T1_IC_TARGET})")
        lines.append(f"   T+2 Rank IC: {ic_audit.get('ic_t2', 0.0):.4f}")
        lines.append(f"   T+3 Rank IC: {ic_audit.get('ic_t3', 0.0):.4f}")
        lines.append(f"   IC 衰减正常：{'是' if ic_audit.get('decay_normal') else '否'}")
        lines.append(f"   T+1 IC 达标：{'是' if ic_audit.get('t1_ic_passed') else '否'}")
        
        ic_by_year = ic_audit.get('ic_by_year', {})
        if ic_by_year:
            lines.append("")
            lines.append("   分年度 IC:")
            for year in sorted(ic_by_year.keys()):
                ic_val = ic_by_year[year]
                status = "✓" if ic_val > 0 else "✗"
                lines.append(f"     {year}年：{ic_val:.4f} {status}")
        lines.append("")
        
        lines.append("3. 交易执行审计")
        lines.append("   " + "-" * 50)
        lines.append(f"   总收益：{trade_results.get('total_return', 0.0):.2%}")
        lines.append(f"   最终价值：{trade_results.get('final_value', 0.0):,.2f}")
        lines.append(f"   最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
        lines.append(f"   年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
        lines.append(f"   调仓次数：{trade_results.get('rebalance_count', 0)}")
        lines.append(f"   平均持仓数：{trade_results.get('avg_position_count', 0):.1f}")
        
        annual_returns = trade_results.get('annual_returns', {})
        if annual_returns:
            lines.append("")
            lines.append("   年度收益:")
            for year, ret in annual_returns.items():
                lines.append(f"     {year}年：{ret:.2%}")
            lines.append(f"   平均年化收益：{trade_results.get('avg_annual_return', 0.0):.2%}")
        lines.append("")
        
        lines.append("4. V96 硬性指标验证")
        lines.append("   " + "-" * 50)
        
        ic_t1 = ic_audit.get('ic_t1', 0.0)
        metric_a_pass = ic_t1 >= V96_T1_IC_TARGET
        lines.append(f"   指标 A (T+1 Rank IC >= 0.048): {'✓' if metric_a_pass else '✗'}")
        lines.append(f"     - T+1 IC: {ic_t1:.4f}")
        lines.append("")
        
        max_dd = trade_results.get('max_drawdown', 0.0)
        metric_b_pass = max_dd <= 0.12
        lines.append(f"   指标 B (最大回撤 <= 12%): {'✓' if metric_b_pass else '✗'}")
        lines.append(f"     - 最大回撤：{max_dd:.2%}")
        lines.append("")
        
        # 中性化状态检查
        neutralization_ok = (
            self.config.enable_industry_neutralization and 
            self.config.enable_size_neutralization and
            self.config.enable_liquidity_filter
        )
        lines.append(f"   指标 C (中性化状态 ENABLED): {'✓' if neutralization_ok else '✗ INVALID!'}")
        lines.append(f"     - 行业中性化：{'ENABLED' if self.config.enable_industry_neutralization else 'DISABLED'}")
        lines.append(f"     - 市值中性化：{'ENABLED' if self.config.enable_size_neutralization else 'DISABLED'}")
        lines.append(f"     - 流动性过滤：{'ENABLED' if self.config.enable_liquidity_filter else 'DISABLED'}")
        lines.append("")
        
        lines.append("5. V96 新特性")
        lines.append("   " + "-" * 50)
        lines.append(f"   日内偏度因子：{'✓ ENABLED' if self.config.enable_intraday_skewness else '✗ DISABLED'}")
        lines.append(f"   非线性交互：{'✓ ENABLED' if self.config.enable_nonlinear_interaction else '✗ DISABLED'}")
        lines.append("")
        
        lines.append("=" * 70)
        
        all_passed = metric_a_pass and metric_b_pass and neutralization_ok
        lines.append(f"总体评估：{'所有指标通过 ✓' if all_passed else '部分指标未通过 ✗'}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'data_integrity': {},
            'ic_audit': {},
            'trade_results': {},
            'audit_report': '',
            'trade_records': [],
            'daily_snapshots': [],
            'rebalance_dates': [],
            'neutralization_status': {},
        }


# ===========================================
# 主程序
# ===========================================

def run_v96_backtest(config: V96EngineConfig = None) -> Dict[str, Any]:
    """运行 V96 回测"""
    engine = V96Engine(config=config)
    return engine.run_backtest()


def print_v96_report(result: Dict[str, Any]) -> None:
    """打印 V96 报告"""
    logger.info("=" * 70)
    logger.info("V96 最终报告")
    logger.info("=" * 70)
    
    trade_results = result.get('trade_results', {})
    logger.info("【交易执行】")
    logger.info(f"  总收益：{trade_results.get('total_return', 0.0):.2%}")
    logger.info(f"  最终价值：{trade_results.get('final_value', 0.0):,.2f}")
    logger.info(f"  最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
    logger.info(f"  年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
    logger.info(f"  平均持仓数：{trade_results.get('avg_position_count', 0):.1f}")
    logger.info(f"  调仓次数：{trade_results.get('rebalance_count', 0)}")
    logger.info(f"  平均年化收益：{trade_results.get('avg_annual_return', 0.0):.2%}")
    
    ic_audit = result.get('ic_audit', {})
    logger.info("")
    logger.info("【IC 审计】")
    logger.info(f"  T+1 IC: {ic_audit.get('ic_t1', 0.0):.4f}")
    logger.info(f"  T+2 IC: {ic_audit.get('ic_t2', 0.0):.4f}")
    logger.info(f"  T+3 IC: {ic_audit.get('ic_t3', 0.0):.4f}")
    logger.info(f"  衰减正常：{'是' if ic_audit.get('decay_normal') else '否'}")
    
    neutralization = result.get('neutralization_status', {})
    logger.info("")
    logger.info("【中性化状态】")
    logger.info(f"  行业中性化：{'✓ ENABLED' if neutralization.get('industry_neutralization') else '✗ DISABLED'}")
    logger.info(f"  市值中性化：{'✓ ENABLED' if neutralization.get('size_neutralization') else '✗ DISABLED'}")
    logger.info(f"  流动性过滤：{'✓ ENABLED' if neutralization.get('liquidity_filter') else '✗ DISABLED'}")
    
    audit_report = result.get('audit_report', '')
    if audit_report:
        logger.info("")
        logger.info(audit_report)
    
    logger.info("=" * 70)


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # V96 配置（强制中性化，严禁禁用）
    config = V96EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        enable_industry_neutralization=True,  # 强制开启
        enable_size_neutralization=True,      # 强制开启
        enable_liquidity_filter=True,         # 强制开启
        filter_st=True,                       # 强制过滤 ST
        enable_intraday_skewness=True,        # 新增因子
        enable_nonlinear_interaction=True,    # 非线性交互
    )
    
    result = run_v96_backtest(config)
    print_v96_report(result)
    
    # 保存结果
    output_path = "reports/v96_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_result = {
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': result.get('ic_audit', {}),
        'trade_results': result.get('trade_results', {}),
        'audit_report': result.get('audit_report', ''),
        'neutralization_status': result.get('neutralization_status', {}),
        'trade_count': len(result.get('trade_records', [])),
        'snapshot_count': len(result.get('daily_snapshots', [])),
        'rebalance_count': len(result.get('rebalance_dates', [])),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V96: 结果已保存至 {output_path}")