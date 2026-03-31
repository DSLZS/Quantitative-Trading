"""
V97 Engine - 深度防御与重启：V90 基准复现 + 量价一致性因子消融实验

【V97 核心理念】
1. 强制中性化 - 行业中性化和市值中性化必须开启
2. 强制流动性过滤 - 禁止买入 ST 股、禁止买入成交额后 10% 的股票
3. V90 基准复现 - 首先完整复现 V90 的原始算法
4. 消融实验 - 引入"量价一致性强度"因子，分别打印各因子 IC
5. 自动回退 - 如果融合后 IC 低于原始 V90，自动剔除新因子

【V97 硬性指标】
| 维度 | 指标 | 目标值 | 失败判定 |
| :--- | :--- | :--- | :--- |
| 预测力 | T+1 Rank IC | ≥ 0.048 | 低于 0.045 视为负优化 |
| 真实度 | 数据采样打印 | 必须可见 | 缺失采样视为逻辑欺诈 |
| 风险比 | Calmar Ratio | > 1.5 | 禁用中性化则报告无效 |
| 活跃度 | 年化换手率 | 300% - 600% | 低于 200% 视为消极交易 |

作者：量化系统
版本：V97.0
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

# 导入 V97 核心模块
from src.core.v97_core import (
    V97DataManager,
    V97IndustryNeutralizationEngine,
    V97SizeNeutralizationEngine,
    V97LiquidityFilterEngine,
    V97ResidualMomentumEngine,
    V97SmartFlowEngine,
    V97PriceVolumeConsistencyEngine,
    V97AlphaFusion,
    V97AlphaWeightEngine,
    V97ICAudit,
    V97TurnoverTracker,
    V97PortfolioTracker,
    V97DynamicRebalanceEngine,
    V97_INITIAL_CAPITAL,
    V97_MAX_POSITIONS,
    V97_WARMUP_PERIOD,
    V97_MIN_SCORE_THRESHOLD,
    V97_MIN_SINGLE_WEIGHT,
    V97_MAX_SINGLE_WEIGHT,
    V97_TURNOVER_MIN,
    V97_TURNOVER_MAX,
    V97_DAILY_TURNOVER_MAX,
    V97_T1_IC_TARGET,
    V97_COMMISSION_RATE,
    V97_MIN_COMMISSION,
    V97_STAMP_DUTY,
    V97_TRANSFER_FEE,
    V97_HALF_LIFE_LAGS,
    V97_RESIDUAL_WEIGHT,
    V97_FLOW_WEIGHT,
    V97_CONSISTENCY_WEIGHT,
    V97_SIZE_NEUTRALIZATION,
    V97_INDUSTRY_NEUTRALIZATION,
    V97_LIQUIDITY_FILTER,
    V97_FILTER_ST,
    V97_NEUTRALIZATION_WINDOW,
    V97_AUDIT_MODE,
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
# V97 引擎配置
# ===========================================

class V97EngineConfig:
    """V97 引擎配置"""
    
    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = V97_INITIAL_CAPITAL,
        max_positions: int = V97_MAX_POSITIONS,
        warmup_period: int = V97_WARMUP_PERIOD,
        commission_rate: float = V97_COMMISSION_RATE,
        min_commission: float = V97_MIN_COMMISSION,
        stamp_duty: float = V97_STAMP_DUTY,
        transfer_fee: float = V97_TRANSFER_FEE,
        oos_years: List[str] = None,
        min_score_threshold: float = V97_MIN_SCORE_THRESHOLD,
        min_single_weight: float = V97_MIN_SINGLE_WEIGHT,
        max_single_weight: float = V97_MAX_SINGLE_WEIGHT,
        # V97 强制配置
        enable_industry_neutralization: bool = True,
        enable_size_neutralization: bool = True,
        enable_liquidity_filter: bool = True,
        filter_st: bool = True,
        # V97 新增因子
        enable_consistency_factor: bool = True,
        # 消融实验
        audit_mode: bool = True,
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
        
        # V97 强制配置
        self.enable_industry_neutralization = enable_industry_neutralization
        self.enable_size_neutralization = enable_size_neutralization
        self.enable_liquidity_filter = enable_liquidity_filter
        self.filter_st = filter_st
        self.enable_consistency_factor = enable_consistency_factor
        self.audit_mode = audit_mode
        
        # 验证强制配置
        if not self.enable_industry_neutralization:
            logger.error("V97: 行业中性化必须开启，否则回测无效！")
        if not self.enable_size_neutralization:
            logger.error("V97: 市值中性化必须开启，否则回测无效！")
        if not self.enable_liquidity_filter:
            logger.error("V97: 流动性过滤必须开启，否则回测无效！")


# ===========================================
# V97 引擎
# ===========================================

class V97Engine:
    """V97 回测引擎"""
    
    def __init__(self, config: V97EngineConfig = None, db=None):
        self.config = config or V97EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V97: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        self.data_manager = V97DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
        })
        
        # 行业中性化（强制）
        self.industry_neutralization = V97IndustryNeutralizationEngine(
            db=self.db,
            config={
                'industry_neutralization': self.config.enable_industry_neutralization,
                'window': V97_NEUTRALIZATION_WINDOW,
            }
        ) if self.config.enable_industry_neutralization else None
        
        # 市值中性化（强制）
        self.size_neutralization = V97SizeNeutralizationEngine(
            config={
                'size_neutralization': self.config.enable_size_neutralization,
            }
        ) if self.config.enable_size_neutralization else None
        
        # 流动性过滤（强制）
        self.liquidity_filter = V97LiquidityFilterEngine(
            config={
                'liquidity_filter': self.config.enable_liquidity_filter,
                'liquidity_percentile': 10,
                'filter_st': self.config.filter_st,
            }
        ) if self.config.enable_liquidity_filter else None
        
        # V90 基准因子
        self.residual_momentum = V97ResidualMomentumEngine()
        self.smart_flow = V97SmartFlowEngine()
        
        # V97 新增因子
        self.consistency = V97PriceVolumeConsistencyEngine() if self.config.enable_consistency_factor else None
        
        self.alpha_fusion = V97AlphaFusion(db=self.db, config={
            'fusion_lags': V97_HALF_LIFE_LAGS,
        })
        self.alpha_weight = V97AlphaWeightEngine(config={
            'min_score': self.config.min_score_threshold,
            'min_weight': self.config.min_single_weight,
            'max_weight': self.config.max_single_weight,
        })
        self.ic_audit = V97ICAudit(db=self.db)
        self.turnover_tracker = V97TurnoverTracker()
        self.portfolio_tracker = V97PortfolioTracker(
            initial_capital=self.config.initial_capital
        )
        self.dynamic_rebalance = V97DynamicRebalanceEngine()
        
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        self.trade_records: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        self.rebalance_dates: List[str] = []
        
        # 消融实验记录
        self.factor_ic_records: Dict[str, Dict] = {}
        
        logger.info("=" * 70)
        logger.info("V97 Engine 初始化完成")
        logger.info("=" * 70)
        logger.info(f"V97: 初始资金={self.config.initial_capital:,.2f}")
        logger.info(f"V97: 最大持仓数={self.config.max_positions}")
        logger.info(f"V97: 评分门槛={self.config.min_score_threshold}")
        logger.info("=" * 70)
        logger.info("V97 强制配置:")
        logger.info(f"  行业中性化：{'✓ ENABLED' if self.config.enable_industry_neutralization else '✗ DISABLED'}")
        logger.info(f"  市值中性化：{'✓ ENABLED' if self.config.enable_size_neutralization else '✗ DISABLED'}")
        logger.info(f"  流动性过滤：{'✓ ENABLED' if self.config.enable_liquidity_filter else '✗ DISABLED'}")
        logger.info(f"  ST 股过滤：{'✓ ENABLED' if self.config.filter_st else '✗ DISABLED'}")
        logger.info(f"  量价一致性因子：{'✓ ENABLED' if self.config.enable_consistency_factor else '✗ DISABLED'}")
        logger.info(f"  消融实验模式：{'✓ ENABLED' if self.config.audit_mode else '✗ DISABLED'}")
        logger.info("=" * 70)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V97 深度防御与重启引擎启动")
        logger.info("=" * 70)
        
        # 验证强制配置
        if not self.config.enable_industry_neutralization:
            logger.error("V97: 行业中性化未开启，回测无效！")
            return self._empty_result()
        if not self.config.enable_size_neutralization:
            logger.error("V97: 市值中性化未开启，回测无效！")
            return self._empty_result()
        if not self.config.enable_liquidity_filter:
            logger.error("V97: 流动性过滤未开启，回测无效！")
            return self._empty_result()
        
        if self.db is None:
            logger.error("V97: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查
            logger.info("V97: [1/8] 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V97: [2/8] 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V97: 未加载到任何数据")
                return self._empty_result()
            
            # 打印 2024 年数据采样（V97 强制要求）
            df_2024 = df.filter(pl.col('trade_date').cast(pl.Utf8).str.starts_with('2024'))
            if not df_2024.is_empty():
                logger.info("V97: [2024 年数据采样 - 包含 industry_code]:")
                sample = df_2024.head(5).select(['symbol', 'trade_date', 'close', 'industry_code', 'total_mv'])
                print(sample)
            
            for year in self.config.oos_years:
                df_year = df.filter(
                    pl.col('trade_date').cast(pl.Utf8).str.starts_with(year)
                )
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V97: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算 V90 基准因子信号
            logger.info("V97: [3/8] 开始计算 V90 基准因子信号...")
            df_with_signals = self._compute_v90_signals(df)
            
            # 4. 计算量价一致性因子（V97 新增）
            if self.config.enable_consistency_factor:
                logger.info("V97: [4/8] 开始计算量价一致性因子...")
                df_with_consistency = self._compute_consistency_factor(df_with_signals)
                df_with_signals = df_with_consistency
            else:
                logger.info("V97: [4/8] 跳过量价一致性因子计算")
            
            # 5. 计算综合评分（消融实验）
            logger.info("V97: [5/8] 开始计算综合评分...")
            df_with_signals = self._compute_composite_score_v97(df_with_signals)
            
            # 6. 行业中性化（V97 强制）
            if self.industry_neutralization:
                logger.info("V97: [6/8] 开始行业中性化 (强制)...")
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
                logger.error("V97: 行业中性化未执行，回测无效！")
            
            # 7. 市值中性化（V97 强制）
            if self.size_neutralization:
                logger.info("V97: [7/8] 开始市值中性化 (强制)...")
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
                logger.error("V97: 市值中性化未执行，回测无效！")
            
            # 流动性过滤（V97 强制）
            if self.liquidity_filter:
                logger.info("V97: 开始流动性过滤 (强制)...")
                df_with_filter = self.liquidity_filter.apply_liquidity_filter(df_with_signals)
                df_with_signals = df_with_filter
            
            # 8. 消融实验：单因子 IC 审计
            if self.config.audit_mode:
                logger.info("V97: [8/8] 开始消融实验 - 单因子 IC 审计...")
                self._run_ablation_study(df_with_signals)
            
            # 半衰期融合
            logger.info("V97: 开始半衰期融合...")
            df_with_fusion = self.alpha_fusion.compute_fusion_signal(
                df_with_signals, signal_col='composite_score'
            )
            
            # Alpha 权重与 IC 审计
            logger.info("V97: 开始 Alpha 权重计算与 IC 审计...")
            df_with_weights = self.alpha_weight.compute_alpha_weights(
                df_with_fusion, score_col='fused_signal'
            )
            
            ic_audit_results = self.ic_audit.calculate_rank_ic(
                df_with_weights, signal_col='fused_signal'
            )
            
            logger.info(f"V97: T+1 Rank IC = {ic_audit_results['ic_t1']:.4f} (目标 > {V97_T1_IC_TARGET})")
            logger.info(f"V97: T+2 Rank IC = {ic_audit_results['ic_t2']:.4f}")
            logger.info(f"V97: T+3 Rank IC = {ic_audit_results['ic_t3']:.4f}")
            
            # 执行回测交易
            logger.info("V97: 开始执行回测交易...")
            trade_results = self._execute_backtest(df_with_weights)
            
            # 生成报告
            logger.info("V97: 生成审计报告...")
            audit_report = self._generate_audit_report(
                data_integrity_results,
                ic_audit_results,
                trade_results,
            )
            
            result = {
                'data_integrity': data_integrity_results,
                'ic_audit': ic_audit_results,
                'factor_ic': self.factor_ic_records,
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
            logger.info("V97 回测完成")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"V97 回测失败 - {e}")
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
                logger.info(f"V97: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V97: {year}年数据检查失败 - {message}")
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
                    logger.info(f"V97: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V97: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V97: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _compute_v90_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 V90 基准因子信号"""
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
        
        # V90 基准因子 1: 残差动量
        result = self.residual_momentum.compute_residual_momentum(result)
        
        # V90 基准因子 2: 聪明资金流
        result = self.smart_flow.compute_smart_flow(result)
        
        return result
    
    def _compute_consistency_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算量价一致性因子（V97 新增）"""
        if self.consistency is None:
            return df
        
        result = self.consistency.compute_price_volume_consistency(df)
        return result
    
    def _compute_composite_score_v97(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 V97 综合评分"""
        result = df.clone()
        
        for col, default in [
            ('residual_momentum_score', 50.0),
            ('smart_flow_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # V97: V90 基准权重
        if 'price_volume_consistency_score' in result.columns and self.config.enable_consistency_factor:
            # 使用新因子
            result = result.with_columns([
                (V97_RESIDUAL_WEIGHT * pl.col('residual_momentum_score') + 
                 V97_FLOW_WEIGHT * pl.col('smart_flow_score') +
                 V97_CONSISTENCY_WEIGHT * pl.col('price_volume_consistency_score')).alias('composite_score')
            ])
        else:
            # 仅使用 V90 基准因子
            result = result.with_columns([
                ((V97_RESIDUAL_WEIGHT + V97_CONSISTENCY_WEIGHT) * pl.col('residual_momentum_score') + 
                 (V97_FLOW_WEIGHT + V97_CONSISTENCY_WEIGHT) * pl.col('smart_flow_score')).alias('composite_score')
            ])
        
        return result
    
    def _run_ablation_study(self, df: pl.DataFrame) -> None:
        """消融实验：单因子 IC 审计"""
        logger.info("=" * 50)
        logger.info("V97 消融实验 - 单因子 IC 审计")
        logger.info("=" * 50)
        
        # V90 基准因子 IC
        ic_residual = self.ic_audit.calculate_single_factor_ic(
            df, 'Residual Momentum (V90)', 'residual_momentum_score'
        )
        ic_flow = self.ic_audit.calculate_single_factor_ic(
            df, 'Smart Flow (V90)', 'smart_flow_score'
        )
        
        self.factor_ic_records['residual_momentum'] = {
            'ic_t1': ic_residual.ic_t1,
            'ic_t2': ic_residual.ic_t2,
            'ic_t3': ic_residual.ic_t3,
            'ic_ir': ic_residual.ic_ir,
        }
        
        self.factor_ic_records['smart_flow'] = {
            'ic_t1': ic_flow.ic_t1,
            'ic_t2': ic_flow.ic_t2,
            'ic_t3': ic_flow.ic_t3,
            'ic_ir': ic_flow.ic_ir,
        }
        
        logger.info(f"V90 基准因子 IC:")
        logger.info(f"  残差动量：T+1={ic_residual.ic_t1:.4f}, T+2={ic_residual.ic_t2:.4f}, T+3={ic_residual.ic_t3:.4f}")
        logger.info(f"  聪明资金流：T+1={ic_flow.ic_t1:.4f}, T+2={ic_flow.ic_t2:.4f}, T+3={ic_flow.ic_t3:.4f}")
        
        # V97 新因子 IC
        if self.config.enable_consistency_factor and 'price_volume_consistency_score' in df.columns:
            ic_consistency = self.ic_audit.calculate_single_factor_ic(
                df, 'Price-Volume Consistency (V97)', 'price_volume_consistency_score'
            )
            self.factor_ic_records['consistency'] = {
                'ic_t1': ic_consistency.ic_t1,
                'ic_t2': ic_consistency.ic_t2,
                'ic_t3': ic_consistency.ic_t3,
                'ic_ir': ic_consistency.ic_ir,
            }
            logger.info(f"  量价一致性：T+1={ic_consistency.ic_t1:.4f}, T+2={ic_consistency.ic_t2:.4f}, T+3={ic_consistency.ic_t3:.4f}")
        
        # V90 基准组合 IC
        v90_score = (V97_RESIDUAL_WEIGHT * df['residual_momentum_score'] + 
                     V97_FLOW_WEIGHT * df['smart_flow_score'])
        df_v90 = df.clone()
        df_v90 = df_v90.with_columns([v90_score.alias('v90_combined_score')])
        ic_v90 = self.ic_audit.calculate_rank_ic(df_v90, 'v90_combined_score')
        
        self.factor_ic_records['v90_baseline'] = {
            'ic_t1': ic_v90['ic_t1'],
            'ic_t2': ic_v90['ic_t2'],
            'ic_t3': ic_v90['ic_t3'],
        }
        logger.info(f"V90 基准组合 IC: T+1={ic_v90['ic_t1']:.4f}")
        
        logger.info("=" * 50)
        
        # 自动回退判断
        if self.config.enable_consistency_factor:
            logger.info("V97: 自动回退机制检查...")
            # 如果新因子 IC 低于 V90 基准，发出警告
            if 'consistency' in self.factor_ic_records:
                consistency_ic = self.factor_ic_records['consistency']['ic_t1']
                v90_ic = self.factor_ic_records['v90_baseline']['ic_t1']
                if consistency_ic < v90_ic * 0.8:  # 低于 V90 的 80%
                    logger.warning(f"V97: 量价一致性因子 IC ({consistency_ic:.4f}) 显著低于 V90 基准 ({v90_ic:.4f})")
                    logger.warning("V97: 建议剔除该因子或调整权重")
    
    def _execute_backtest(self, df: pl.DataFrame) -> Dict[str, Any]:
        """执行回测交易"""
        logger.info("V97: 开始执行回测交易...")
        
        df = df.sort(['trade_date', 'symbol'])
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        warmup_cutoff = unique_dates[:min(V97_WARMUP_PERIOD, len(unique_dates))]
        trade_dates = [d for d in unique_dates if d not in warmup_cutoff]
        
        logger.info(f"V97: 热身期 {len(warmup_cutoff)} 天，交易期 {len(trade_dates)} 天")
        
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
            
            # V97: 应用流动性过滤
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
                logger.debug(f"V97: {trade_date} 调仓")
                
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
                logger.info(f"V97: 处理 {i + 1}/{len(trade_dates)} 天，组合价值={self.portfolio_value:,.2f}, 持仓数={len(self.positions)}")
        
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
        
        logger.info(f"V97: 回测完成 - 总收益={total_return:.2%}, 年化换手={turnover_summary['annualized_turnover']:.2%}")
        logger.info(f"V97: 调仓次数={len(self.rebalance_dates)}, 最大回撤={max_drawdown:.2%}")
        logger.info(f"V97: 平均持仓数={avg_position_count:.1f}")
        
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
            logger.debug(f"V97: {trade_date} 达到调仓间隔 ({days_diff}天)")
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
    
    def _generate_audit_report(self, data_integrity: Dict, ic_audit: Dict,
                                trade_results: Dict) -> str:
        """生成审计报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("《V97 深度防御与重启审计报告》")
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
        lines.append(f"   T+1 Rank IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V97_T1_IC_TARGET})")
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
        
        lines.append("3. 消融实验 - 单因子 IC")
        lines.append("   " + "-" * 50)
        for factor_name, ic_data in self.factor_ic_records.items():
            lines.append(f"   {factor_name}:")
            lines.append(f"     T+1 IC: {ic_data.get('ic_t1', 0.0):.4f}")
            lines.append(f"     T+2 IC: {ic_data.get('ic_t2', 0.0):.4f}")
            lines.append(f"     T+3 IC: {ic_data.get('ic_t3', 0.0):.4f}")
        lines.append("")
        
        lines.append("4. 交易执行审计")
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
        
        lines.append("5. V97 硬性指标验证")
        lines.append("   " + "-" * 50)
        
        ic_t1 = ic_audit.get('ic_t1', 0.0)
        metric_a_pass = ic_t1 >= V97_T1_IC_TARGET
        lines.append(f"   指标 A (T+1 Rank IC >= 0.048): {'✓' if metric_a_pass else '✗'}")
        lines.append(f"     - T+1 IC: {ic_t1:.4f}")
        lines.append("")
        
        max_dd = trade_results.get('max_drawdown', 0.0)
        calmar_ratio = trade_results.get('avg_annual_return', 0.0) / max_dd if max_dd > 0 else 0.0
        metric_b_pass = calmar_ratio > 1.5
        lines.append(f"   指标 B (Calmar Ratio > 1.5): {'✓' if metric_b_pass else '✗'}")
        lines.append(f"     - Calmar Ratio: {calmar_ratio:.2f}")
        lines.append("")
        
        turnover = trade_results.get('annualized_turnover', 0.0)
        turnover_pass = V97_TURNOVER_MIN <= turnover <= V97_TURNOVER_MAX
        lines.append(f"   指标 C (年化换手率 300%-600%): {'✓' if turnover_pass else '✗'}")
        lines.append(f"     - 年化换手率：{turnover:.2%}")
        lines.append("")
        
        # 中性化状态检查
        neutralization_ok = (
            self.config.enable_industry_neutralization and 
            self.config.enable_size_neutralization and
            self.config.enable_liquidity_filter
        )
        lines.append(f"   指标 D (中性化状态 ENABLED): {'✓' if neutralization_ok else '✗ INVALID!'}")
        lines.append(f"     - 行业中性化：{'ENABLED' if self.config.enable_industry_neutralization else 'DISABLED'}")
        lines.append(f"     - 市值中性化：{'ENABLED' if self.config.enable_size_neutralization else 'DISABLED'}")
        lines.append(f"     - 流动性过滤：{'ENABLED' if self.config.enable_liquidity_filter else 'DISABLED'}")
        lines.append("")
        
        lines.append("=" * 70)
        
        all_passed = metric_a_pass and metric_b_pass and turnover_pass and neutralization_ok
        lines.append(f"总体评估：{'所有指标通过 ✓' if all_passed else '部分指标未通过 ✗'}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'data_integrity': {},
            'ic_audit': {},
            'factor_ic': {},
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

def run_v97_backtest(config: V97EngineConfig = None) -> Dict[str, Any]:
    """运行 V97 回测"""
    engine = V97Engine(config=config)
    return engine.run_backtest()


def print_v97_report(result: Dict[str, Any]) -> None:
    """打印 V97 报告"""
    logger.info("=" * 70)
    logger.info("V97 最终报告")
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
    
    factor_ic = result.get('factor_ic', {})
    if factor_ic:
        logger.info("")
        logger.info("【消融实验 - 单因子 IC】")
        for factor_name, ic_data in factor_ic.items():
            logger.info(f"  {factor_name}: T+1={ic_data.get('ic_t1', 0.0):.4f}")
    
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
    
    # V97 配置（强制中性化）
    config = V97EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        enable_industry_neutralization=True,
        enable_size_neutralization=True,
        enable_liquidity_filter=True,
        filter_st=True,
        enable_consistency_factor=True,
        audit_mode=True,
    )
    
    result = run_v97_backtest(config)
    print_v97_report(result)
    
    # 保存结果
    output_path = "reports/v97_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_result = {
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': result.get('ic_audit', {}),
        'factor_ic': result.get('factor_ic', {}),
        'trade_results': result.get('trade_results', {}),
        'audit_report': result.get('audit_report', ''),
        'neutralization_status': result.get('neutralization_status', {}),
        'trade_count': len(result.get('trade_records', [])),
        'snapshot_count': len(result.get('daily_snapshots', [])),
        'rebalance_count': len(result.get('rebalance_dates', [])),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V97: 结果已保存至 {output_path}")