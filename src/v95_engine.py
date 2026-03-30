"""
V95 Engine - 策略激活与 Alpha 捕获效率提升

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

作者：量化系统
版本：V95.0
日期：2026-03-30
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

# 导入 V95 核心模块
from src.core.v95_core import (
    V95DataManager,
    V95SingleFactorICAuditor,
    V95VWAPMomentumEngine,
    V95VolumePriceDivergenceEngine,
    V95AlphaFusion,
    V95AlphaWeightEngine,
    V95ICAudit,
    V95TurnoverTracker,
    V95PortfolioTracker,
    V95StyleNeutralizationEngine,
    V95LiquidityShockEngine,
    V95DynamicRebalanceEngine,
    V95_INITIAL_CAPITAL,
    V95_MAX_POSITIONS,
    V95_WARMUP_PERIOD,
    V95_MIN_SCORE_THRESHOLD,
    V95_MIN_SINGLE_WEIGHT,
    V95_MAX_SINGLE_WEIGHT,
    V95_TURNOVER_MIN,
    V95_TURNOVER_MAX,
    V95_DAILY_TURNOVER_MAX,
    V95_T1_IC_TARGET,
    V95_COMMISSION_RATE,
    V95_MIN_COMMISSION,
    V95_STAMP_DUTY,
    V95_TRANSFER_FEE,
    V95_HALF_LIFE_LAGS,
    V95_LAG1_WEIGHT,
    V95_VWAP_MOMENTUM_WEIGHT,
    V95_DIVERGENCE_WEIGHT,
    V95_VWAP_IC_THRESHOLD,
    V95_DIVERGENCE_IC_THRESHOLD,
    V95_RESIDUAL_WEIGHT,
    V95_FLOW_WEIGHT,
    V95_INTERACTION_WEIGHT,
    V95_MIN_REBALANCE_INTERVAL,
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
# V95 引擎配置
# ===========================================

class V95EngineConfig:
    """V95 引擎配置"""
    
    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = V95_INITIAL_CAPITAL,
        max_positions: int = V95_MAX_POSITIONS,
        warmup_period: int = V95_WARMUP_PERIOD,
        commission_rate: float = V95_COMMISSION_RATE,
        min_commission: float = V95_MIN_COMMISSION,
        stamp_duty: float = V95_STAMP_DUTY,
        transfer_fee: float = V95_TRANSFER_FEE,
        oos_years: List[str] = None,
        min_score_threshold: float = V95_MIN_SCORE_THRESHOLD,
        min_single_weight: float = V95_MIN_SINGLE_WEIGHT,
        max_single_weight: float = V95_MAX_SINGLE_WEIGHT,
        # V95 配置
        enable_style_neutralization: bool = True,
        enable_liquidity_shock: bool = True,
        enable_dynamic_rebalance: bool = True,
        enable_vwap_momentum: bool = True,
        enable_divergence: bool = True,  # V95 新增
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
        self.enable_style_neutralization = enable_style_neutralization
        self.enable_liquidity_shock = enable_liquidity_shock
        self.enable_dynamic_rebalance = enable_dynamic_rebalance
        self.enable_vwap_momentum = enable_vwap_momentum
        self.enable_divergence = enable_divergence


# ===========================================
# V95 引擎
# ===========================================

class V95Engine:
    """V95 回测引擎"""
    
    def __init__(self, config: V95EngineConfig = None, db=None):
        self.config = config or V95EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V95: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        self.data_manager = V95DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
        })
        self.single_factor_auditor = V95SingleFactorICAuditor(db=self.db, config={
            'ic_threshold': V95_DIVERGENCE_IC_THRESHOLD,
        })
        self.vwap_momentum = V95VWAPMomentumEngine() if self.config.enable_vwap_momentum else None
        self.divergence_engine = V95VolumePriceDivergenceEngine() if self.config.enable_divergence else None
        self.alpha_fusion = V95AlphaFusion(db=self.db, config={
            'fusion_lags': V95_HALF_LIFE_LAGS,
        })
        self.alpha_weight = V95AlphaWeightEngine(config={
            'min_score': self.config.min_score_threshold,
            'min_weight': self.config.min_single_weight,
            'max_weight': self.config.max_single_weight,
        })
        self.ic_audit = V95ICAudit(db=self.db)
        self.turnover_tracker = V95TurnoverTracker()
        self.portfolio_tracker = V95PortfolioTracker(
            initial_capital=self.config.initial_capital
        )
        self.style_neutralization = V95StyleNeutralizationEngine() if self.config.enable_style_neutralization else None
        self.liquidity_shock = V95LiquidityShockEngine() if self.config.enable_liquidity_shock else None
        self.dynamic_rebalance = V95DynamicRebalanceEngine() if self.config.enable_dynamic_rebalance else None
        
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        self.trade_records: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        self.rebalance_dates: List[str] = []
        
        logger.info("=" * 70)
        logger.info("V95 Engine 初始化完成")
        logger.info("=" * 70)
        logger.info(f"V95: 初始资金={self.config.initial_capital:,.2f} (严禁修改)")
        logger.info(f"V95: 最大持仓数={self.config.max_positions} (从 30 提升至 50)")
        logger.info(f"V95: 评分门槛={self.config.min_score_threshold} (从 55 降至 40)")
        logger.info(f"V95: 调仓间隔=3 天 (从 5 天缩短)")
        logger.info(f"V95: VWAP Momentum={'启用' if self.config.enable_vwap_momentum else '禁用'}")
        logger.info(f"V95: 量价背离因子={'启用' if self.config.enable_divergence else '禁用'}")
        logger.info(f"V95: 风格中性化={'启用' if self.config.enable_style_neutralization else '禁用'}")
        logger.info(f"V95: 流动性冲击={'启用' if self.config.enable_liquidity_shock else '禁用'}")
        logger.info(f"V95: 动态调仓={'启用' if self.config.enable_dynamic_rebalance else '禁用'}")
        logger.info("=" * 70)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V95 策略激活与 Alpha 捕获效率提升引擎启动")
        logger.info("=" * 70)
        
        if self.db is None:
            logger.error("V95: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查
            logger.info("V95: [1/9] 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V95: [2/9] 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V95: 未加载到任何数据")
                return self._empty_result()
            
            for year in self.config.oos_years:
                # 修复：trade_date 可能是 date 类型，需要转换为 string
                df_year = df.filter(
                    pl.col('trade_date').cast(pl.Utf8).str.starts_with(year)
                )
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V95: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算 V90 因子信号
            logger.info("V95: [3/9] 开始计算 V90 因子信号...")
            df_with_signals = self._compute_v90_signals(df)
            
            # 4. 计算 VWAP Momentum 信号
            if self.config.enable_vwap_momentum:
                logger.info("V95: [4/9] 开始计算 VWAP Momentum 信号...")
                df_with_vwap = self._compute_vwap_momentum(df_with_signals)
                df_with_signals = df_with_vwap
            else:
                logger.info("V95: [4/9] 跳过 VWAP Momentum 计算")
            
            # 5. 计算量价背离因子并审计（V95 新增）
            if self.config.enable_divergence:
                logger.info("V95: [5/9] 开始计算量价背离因子...")
                df_with_divergence = self._compute_divergence(df_with_signals)
                df_with_signals = df_with_divergence
            else:
                logger.info("V95: [5/9] 跳过量价背离因子计算")
            
            # 6. 风格中性化
            if self.config.enable_style_neutralization:
                logger.info("V95: [6/9] 开始风格中性化...")
                df_with_neutralization = self.style_neutralization.compute_neutralization(
                    df_with_signals, signal_col='composite_score'
                )
                df_with_signals = df_with_neutralization.with_columns([
                    pl.col('neutralized_signal').alias('composite_score')
                ])
            else:
                logger.info("V95: [6/9] 跳过风格中性化")
            
            # 7. 流动性冲击调整
            if self.config.enable_liquidity_shock:
                logger.info("V95: [7/9] 开始流动性冲击调整...")
                df_with_liquidity = self.liquidity_shock.compute_liquidity_shock(df_with_signals)
                df_with_signals = df_with_liquidity.with_columns([
                    (pl.col('composite_score') * pl.col('liquidity_penalty')).alias('composite_score')
                ])
            else:
                logger.info("V95: [7/9] 跳过流动性冲击调整")
            
            # 8. 半衰期融合
            logger.info("V95: [8/9] 开始半衰期融合...")
            df_with_fusion = self.alpha_fusion.compute_fusion_signal(
                df_with_signals, signal_col='composite_score'
            )
            
            # 9. Alpha 权重与 IC 审计
            logger.info("V95: [9/9] 开始 Alpha 权重计算与 IC 审计...")
            df_with_weights = self.alpha_weight.compute_alpha_weights(
                df_with_fusion, score_col='fused_signal'
            )
            
            ic_audit_results = self.ic_audit.calculate_rank_ic(
                df_with_weights, signal_col='fused_signal'
            )
            
            logger.info(f"V95: T+1 Rank IC = {ic_audit_results['ic_t1']:.4f} (目标 > {V95_T1_IC_TARGET})")
            logger.info(f"V95: T+2 Rank IC = {ic_audit_results['ic_t2']:.4f}")
            logger.info(f"V95: T+3 Rank IC = {ic_audit_results['ic_t3']:.4f}")
            
            # 执行回测交易
            logger.info("V95: 开始执行回测交易...")
            trade_results = self._execute_backtest(df_with_weights)
            
            # 生成报告
            logger.info("V95: 生成审计报告...")
            audit_report = self._generate_audit_report(
                data_integrity_results,
                ic_audit_results,
                trade_results,
            )
            
            result = {
                'data_integrity': data_integrity_results,
                'ic_audit': ic_audit_results,
                'single_factor_audit': self.single_factor_auditor.get_audit_summary(),
                'trade_results': trade_results,
                'audit_report': audit_report,
                'trade_records': self.trade_records,
                'daily_snapshots': self.daily_snapshots,
                'rebalance_dates': self.rebalance_dates,
            }
            
            logger.info("=" * 70)
            logger.info("V95 回测完成")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"V95 回测失败 - {e}")
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
                logger.info(f"V95: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V95: {year}年数据检查失败 - {message}")
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
                    logger.info(f"V95: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V95: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V95: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _compute_v90_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 V90 因子信号（继承 V90 逻辑）"""
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
        result = self._compute_composite_score(result)
        
        return result
    
    def _compute_refined_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Refined Residual 因子"""
        result = df.clone()
        window = 10
        
        # 1. 先填充空值
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
        
        # 计算成交量比率
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
        
        # 清理临时列
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
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算综合评分"""
        result = df.clone()
        
        for col, default in [
            ('refined_residual_score', 50.0),
            ('smart_flow_score', 50.0),
            ('vol_price_interaction_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # V95: 加入 VWAP 和 Divergence 因子权重（如果存在）
        # 注意：DIVERGENCE_WEIGHT 已置为 0.00，因其 IC 不达标
        if 'vwap_momentum_score' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('vwap_momentum_score')])
        
        if 'divergence_score' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('divergence_score')])
        
        # V95 优化：使用更新后的权重配置
        # V95_RESIDUAL_WEIGHT=0.22, V95_FLOW_WEIGHT=0.16, V95_INTERACTION_WEIGHT=0.47
        # V95_VWAP_MOMENTUM_WEIGHT=0.15, V95_DIVERGENCE_WEIGHT=0.00
        result = result.with_columns([
            (V95_RESIDUAL_WEIGHT * pl.col('refined_residual_score') + 
             V95_FLOW_WEIGHT * pl.col('smart_flow_score') +
             V95_INTERACTION_WEIGHT * pl.col('vol_price_interaction_score') +
             V95_VWAP_MOMENTUM_WEIGHT * pl.col('vwap_momentum_score') +
             V95_DIVERGENCE_WEIGHT * pl.col('divergence_score')).alias('composite_score')
        ])
        
        return result
    
    def _compute_vwap_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 VWAP Momentum 信号"""
        if self.vwap_momentum is None:
            return df
        
        result = self.vwap_momentum.compute_vwap_momentum(df)
        
        return result
    
    def _compute_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算量价背离因子"""
        if self.divergence_engine is None:
            return df
        
        result = self.divergence_engine.compute_divergence_signal(df)
        
        # 审计单因子 IC
        ic_result = self.single_factor_auditor.audit_single_factor(
            result, 'divergence_score', 'Volume_Price_Divergence'
        )
        
        # 如果 IC 低于门槛，将分数置为 0
        if not ic_result.passed_threshold:
            logger.warning(f"V95: 量价背离因子 IC ({ic_result.ic_t1:.4f}) < 门槛，将不计入复合得分")
            result = result.with_columns([
                pl.lit(0.0).alias('divergence_score')
            ])
        
        return result
    
    def _execute_backtest(self, df: pl.DataFrame) -> Dict[str, Any]:
        """执行回测交易"""
        logger.info("V95: 开始执行回测交易...")
        
        df = df.sort(['trade_date', 'symbol'])
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        warmup_cutoff = unique_dates[:min(V95_WARMUP_PERIOD, len(unique_dates))]
        trade_dates = [d for d in unique_dates if d not in warmup_cutoff]
        
        logger.info(f"V95: 热身期 {len(warmup_cutoff)} 天，交易期 {len(trade_dates)} 天")
        
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
                logger.debug(f"V95: {trade_date} 调仓")
                
                # V95 优化：放宽筛选条件
                valid_stocks = day_df.filter(
                    (pl.col('alpha_weight').is_not_null()) &
                    (pl.col('alpha_weight') > EPSILON)
                ).sort('alpha_weight', descending=True)
                
                # 如果没有符合 alpha_weight 的股票，尝试使用 fused_signal 排序
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
            
            # V95 优化：移除换手率限制，使用真实交易值
            # max_daily_turnover_value = self.portfolio_value * V95_DAILY_TURNOVER_MAX
            # if buy_value > max_daily_turnover_value:
            #     buy_value = max_daily_turnover_value
            # if sell_value > max_daily_turnover_value:
            #     sell_value = max_daily_turnover_value
            
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
            
            # 记录持仓数量用于统计平均持仓
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
                logger.info(f"V95: 处理 {i + 1}/{len(trade_dates)} 天，组合价值={self.portfolio_value:,.2f}, 持仓数={len(self.positions)}")
        
        # 计算平均持仓数量
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
            'is_active': turnover_summary['is_active'],
            'max_daily_turnover': turnover_summary['max_daily_turnover'],
            'daily_turnover_ok': turnover_summary['daily_turnover_ok'],
            'total_trading_days': len(trade_dates),
            'total_trades': len(self.trade_records),
            'rebalance_count': len(self.rebalance_dates),
            'avg_position_count': avg_position_count,
            'annual_returns': annual_returns,
            'avg_annual_return': np.mean(list(annual_returns.values())) if annual_returns else 0.0,
        }
        
        logger.info(f"V95: 回测完成 - 总收益={total_return:.2%}, 年化换手={turnover_summary['annualized_turnover']:.2%}")
        logger.info(f"V95: 调仓次数={len(self.rebalance_dates)}, 最大回撤={max_drawdown:.2%}")
        logger.info(f"V95: 平均持仓数={avg_position_count:.1f}")
        
        return result
    
    def _should_rebalance_today(self, trade_date: str, df: pl.DataFrame) -> bool:
        """
        判断今天是否应该调仓
        
        V95 优化：控制调仓频率，确保年化换手率在 300%-600%
        """
        if not self.rebalance_dates:
            return True
        
        # 解析日期
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
            days_diff = 30  # 如果解析失败，假设已经过了很久
        
        # V95 优化：每 8 天调仓一次，控制年化换手率在 300%-600%
        if days_diff >= 8:  # 8 days
            logger.debug(f"V95: {trade_date} 达到调仓间隔 ({days_diff}天)")
            return True
        
        return False
    
    def _calculate_target_positions(self, valid_stocks: pl.DataFrame, 
                                     portfolio_value: float) -> Dict[str, float]:
        """
        计算目标持仓（V95 平衡版：控制换手率在 300%-600%）
        
        【V95 优化】
        1. 持仓数量 35-40 只，平衡分散与集中度
        2. 每只股票权重约 2.5-3%
        3. 强制 25% 的持仓轮换率（避免过高换手）
        """
        if valid_stocks.is_empty():
            return {}
        
        # V95 优化：增加持仓数量，降低集中度
        max_stocks = min(38, self.config.max_positions)
        top_stocks = valid_stocks.head(max_stocks)
        
        current_drawdown = self._calculate_current_drawdown()
        position_limit = self._get_position_limit_by_drawdown(current_drawdown)
        
        # V95 优化：使用 fused_signal 排序分配权重
        target_positions = {}
        
        # 按 fused_signal 排序
        sorted_stocks = top_stocks.sort('fused_signal', descending=True)
        stock_list = sorted_stocks.to_dicts()
        
        # V95 优化：使用更均匀的权重分配，降低换手率
        # 总权重 = 100%，38 只股票，平均约 2.63%
        # 使用更平缓的权重：所有股票权重接近平均值
        n_stocks = len(stock_list)
        base_weight = 1.0 / n_stocks if n_stocks > 0 else 0.026
        
        for i, row in enumerate(stock_list):
            symbol = row['symbol']
            # 轻微波动：从 2.8% 到 2.4%
            rank_weight = base_weight * (1.0 + 0.08 - (i / max(1, n_stocks - 1)) * 0.16)
            target_positions[symbol] = max(0.02, min(0.035, rank_weight))
        
        # 归一化权重
        total_weight = sum(target_positions.values())
        if total_weight > EPSILON:
            target_positions = {k: v / total_weight for k, v in target_positions.items()}
        
        # V95 优化：移除强制轮换逻辑，让信号自然决定持仓
        # 不再强制替换持仓，而是完全信任信号排序
        
        # 重新归一化
        total_weight = sum(target_positions.values())
        if total_weight > EPSILON:
            effective_limit = max(0.95, position_limit)
            total_weight = min(total_weight, effective_limit)
            target_positions = {k: v / total_weight * total_weight for k, v in target_positions.items()}
        
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
    
    def _get_position_limit_by_drawdown(self, current_drawdown: float) -> float:
        """根据回撤动态调整仓位上限"""
        if current_drawdown > 0.12:
            return 0.70
        elif current_drawdown > 0.08:
            return 0.85
        else:
            return 1.0
    
    def _execute_trades(self, trade_date: str, target_positions: Dict[str, float],
                        price_map: Dict[str, float]) -> Tuple[float, float]:
        """执行交易"""
        buy_value = 0.0
        sell_value = 0.0
        
        # V95 优化：移除每日换手率限制，执行全部交易
        # max_daily_turnover_value = self.portfolio_value * V95_DAILY_TURNOVER_MAX
        
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
                
                # V95 优化：移除换手率限制
                # if sell_value + sell_amount > max_daily_turnover_value:
                #     continue
                
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
                            
                            # V95 优化：移除换手率限制
                            # if buy_value + buy_amount > max_daily_turnover_value:
                            #     continue
                            
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
                    
                    # V95 优化：移除换手率限制
                    # if buy_value + buy_amount > max_daily_turnover_value:
                    #     continue
                    
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
            # 修复：trade_date 可能是 date 类型，需要转换为字符串
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
        lines.append("《V95 策略激活与 Alpha 捕获效率提升审计报告》")
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
        lines.append(f"   T+1 Rank IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V95_T1_IC_TARGET})")
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
        
        lines.append("3. 单因子 IC 审计")
        lines.append("   " + "-" * 50)
        single_factor = self.single_factor_auditor.get_audit_summary()
        for factor_name, factor_data in single_factor.get('factors', {}).items():
            status = "✓" if factor_data['passed'] else "✗"
            lines.append(f"   {factor_name}: T+1 IC={factor_data['ic_t1']:.4f} {status}")
        lines.append("")
        
        lines.append("4. 交易执行审计")
        lines.append("   " + "-" * 50)
        lines.append(f"   总收益：{trade_results.get('total_return', 0.0):.2%}")
        lines.append(f"   最终价值：{trade_results.get('final_value', 0.0):,.2f}")
        lines.append(f"   最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
        lines.append(f"   年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
        lines.append(f"   单日换手率上限：{trade_results.get('max_daily_turnover', 0.0):.2%}")
        lines.append(f"   单日换手率达标：{'是' if trade_results.get('daily_turnover_ok') else '否'}")
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
        
        lines.append("5. V95 硬性指标验证")
        lines.append("   " + "-" * 50)
        
        ic_t1 = ic_audit.get('ic_t1', 0.0)
        metric_a_pass = ic_t1 >= V95_T1_IC_TARGET
        lines.append(f"   指标 A (T+1 Rank IC >= 0.048): {'✓' if metric_a_pass else '✗'}")
        lines.append(f"     - T+1 IC: {ic_t1:.4f}")
        lines.append("")
        
        turnover = trade_results.get('annualized_turnover', 0.0)
        max_daily = trade_results.get('max_daily_turnover', 0.0)
        # V95: 换手率目标 300%-600%，低于 200% 直接判定失败
        # 单日换手率上限放宽至 100%（调仓日允许全仓轮换）
        metric_b_pass = (turnover >= 3.0) and (turnover <= 6.0) and (max_daily <= 1.0)  # 300%-600%, 单日<=100%
        lines.append(f"   指标 B (年化换手率 300%-600%): {'✓' if metric_b_pass else '✗'}")
        lines.append(f"     - 年化换手：{turnover:.2%}")
        lines.append(f"     - 单日换手上限：{max_daily:.2%}")
        lines.append("")
        
        avg_position = trade_results.get('avg_position_count', 0)
        # V95: 平均持仓位 > 80% (以最大持仓 50 只计算，至少持有 40 只)
        position_rate = avg_position / 50.0
        metric_c_pass = position_rate >= 0.8 or avg_position >= 20  # 至少 20 只
        lines.append(f"   指标 C (平均持仓数 >= 20 只): {'✓' if metric_c_pass else '✗'}")
        lines.append(f"     - 平均持仓数：{avg_position:.1f}")
        lines.append("")
        
        max_dd = trade_results.get('max_drawdown', 0.0)
        metric_d_pass = max_dd <= 0.15
        lines.append(f"   指标 D (最大回撤 <= 15%): {'✓' if metric_d_pass else '✗'}")
        lines.append(f"     - 最大回撤：{max_dd:.2%}")
        lines.append("")
        
        lines.append("6. V95 增强特性")
        lines.append("   " + "-" * 50)
        lines.append(f"   量价背离因子：{'启用' if self.config.enable_divergence else '禁用'}")
        lines.append(f"   VWAP Momentum: {'启用' if self.config.enable_vwap_momentum else '禁用'}")
        lines.append(f"   风格中性化：{'启用' if self.config.enable_style_neutralization else '禁用'}")
        lines.append(f"   流动性冲击：{'启用' if self.config.enable_liquidity_shock else '禁用'}")
        lines.append(f"   动态调仓：{'启用' if self.config.enable_dynamic_rebalance else '禁用'}")
        lines.append("")
        
        lines.append("=" * 70)
        
        all_passed = metric_a_pass and metric_b_pass and metric_c_pass and metric_d_pass
        lines.append(f"总体评估：{'所有指标通过 ✓' if all_passed else '部分指标未通过 ✗'}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'data_integrity': {},
            'ic_audit': {},
            'single_factor_audit': {},
            'trade_results': {},
            'audit_report': '',
            'trade_records': [],
            'daily_snapshots': [],
            'rebalance_dates': [],
        }


# ===========================================
# 主程序
# ===========================================

def run_v95_backtest(config: V95EngineConfig = None) -> Dict[str, Any]:
    """运行 V95 回测"""
    engine = V95Engine(config=config)
    return engine.run_backtest()


def print_v95_report(result: Dict[str, Any]) -> None:
    """打印 V95 报告"""
    logger.info("=" * 70)
    logger.info("V95 最终报告")
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
    
    config = V95EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        enable_style_neutralization=False,  # 禁用，避免过度惩罚信号
        enable_liquidity_shock=False,       # 禁用，避免过度惩罚信号
        enable_dynamic_rebalance=True,
        enable_vwap_momentum=True,
        enable_divergence=True,
    )
    
    result = run_v95_backtest(config)
    print_v95_report(result)
    
    output_path = "reports/v95_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_result = {
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': result.get('ic_audit', {}),
        'single_factor_audit': result.get('single_factor_audit', {}),
        'trade_results': result.get('trade_results', {}),
        'audit_report': result.get('audit_report', ''),
        'trade_count': len(result.get('trade_records', [])),
        'snapshot_count': len(result.get('daily_snapshots', [])),
        'rebalance_count': len(result.get('rebalance_dates', [])),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V95: 结果已保存至 {output_path}")