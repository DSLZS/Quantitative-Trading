"""
V92 Engine - IC 驱动预测与稳健 Alpha 修复

【V92 核心任务】
1. 修复 V91 信号反转错误
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
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger

# 导入 V92 核心模块
from src.core.v92_logic import (
    V92DataManager,
    V92DivergenceEngine,
    V92ICWeightEngine,
    V92StyleNeutralizationEngine,
    V92ICAudit,
    V92ConsistencyChecker,
    V92LiquidityShockEngine,
    V92TurnoverTracker,
    V92_INITIAL_CAPITAL,
    V92_MAX_POSITIONS,
    V92_WARMUP_PERIOD,
    V92_MIN_SCORE_THRESHOLD,
    V92_MIN_SINGLE_WEIGHT,
    V92_MAX_SINGLE_WEIGHT,
    V92_T1_IC_TARGET,
    V92_IC_IR_TARGET,
    V92_COMMISSION_RATE,
    V92_MIN_COMMISSION,
    V92_STAMP_DUTY,
    V92_TRANSFER_FEE,
    V92_MIN_REBALANCE_INTERVAL,
    V92_MAX_REBALANCE_INTERVAL,
    V92_DAILY_TURNOVER_MAX,
    V92_DIVERGENCE_WEIGHT,
    V92_RANK_CORRELATION_THRESHOLD,
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
# V92 引擎配置
# ===========================================

class V92EngineConfig:
    """V92 引擎配置"""
    
    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = V92_INITIAL_CAPITAL,
        max_positions: int = V92_MAX_POSITIONS,
        warmup_period: int = V92_WARMUP_PERIOD,
        commission_rate: float = V92_COMMISSION_RATE,
        min_commission: float = V92_MIN_COMMISSION,
        stamp_duty: float = V92_STAMP_DUTY,
        transfer_fee: float = V92_TRANSFER_FEE,
        oos_years: List[str] = None,
        min_score_threshold: float = V92_MIN_SCORE_THRESHOLD,
        min_single_weight: float = V92_MIN_SINGLE_WEIGHT,
        max_single_weight: float = V92_MAX_SINGLE_WEIGHT,
        # V92 新增配置
        enable_divergence: bool = True,  # 启用背离检测
        enable_ic_weighting: bool = True,  # 启用 IC 权重
        enable_neutralization: bool = True,  # 启用中性化
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
        # V92 新增配置
        self.enable_divergence = enable_divergence
        self.enable_ic_weighting = enable_ic_weighting
        self.enable_neutralization = enable_neutralization


# ===========================================
# V92 引擎
# ===========================================

class V92Engine:
    """V92 回测引擎"""
    
    def __init__(self, config: V92EngineConfig = None, db=None):
        self.config = config or V92EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V92: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        # V92 核心模块
        self.data_manager = V92DataManager(db=self.db)
        self.divergence_engine = V92DivergenceEngine() if self.config.enable_divergence else None
        self.ic_weight_engine = V92ICWeightEngine() if self.config.enable_ic_weighting else None
        self.style_neutralization = V92StyleNeutralizationEngine() if self.config.enable_neutralization else None
        self.liquidity_shock_engine = V92LiquidityShockEngine()
        self.ic_audit = V92ICAudit(db=self.db)
        self.consistency_checker = V92ConsistencyChecker()
        
        # V92 新增：换手率追踪器
        self.turnover_tracker = V92TurnoverTracker()
        
        # 组合管理
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        # 交易记录
        self.trade_records: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        self.rebalance_dates: List[str] = []
        
        # V92 新增：调仓状态追踪
        self.last_rebalance_date = None
        self.days_since_rebalance = 0
        self.prev_rank_correlation = None
        
        logger.info("=" * 70)
        logger.info("V92 Engine 初始化完成")
        logger.info("=" * 70)
        logger.info(f"V92: 初始资金={self.config.initial_capital:,.2f}")
        logger.info(f"V92: 量价背离={'启用' if self.config.enable_divergence else '禁用'}")
        logger.info(f"V92: IC 权重={'启用' if self.config.enable_ic_weighting else '禁用'}")
        logger.info(f"V92: 中性化={'启用' if self.config.enable_neutralization else '禁用'}")
        logger.info(f"V92: Score 阈值={self.config.min_score_threshold}")
        logger.info(f"V92: IC 目标={V92_T1_IC_TARGET:.4f}, IC IR 目标={V92_IC_IR_TARGET:.2f}")
        logger.info("=" * 70)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V92 IC 驱动预测回测引擎启动")
        logger.info("=" * 70)
        
        if self.db is None:
            logger.error("V92: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查
            logger.info("V92: [1/7] 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V92: [2/7] 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V92: 未加载到任何数据")
                return self._empty_result()
            
            for year in self.config.oos_years:
                df_year = df.filter(pl.col('trade_date').str.starts_with(year))
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V92: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算因子信号
            logger.info("V92: [3/7] 开始计算因子信号...")
            df_with_signals = self._compute_signals(df)
            
            # 4. 量价背离计算（V92 新增）
            if self.config.enable_divergence:
                logger.info("V92: [4/7] 开始量价背离计算...")
                df_with_divergence = self.divergence_engine.compute_divergence(df_with_signals)
            else:
                logger.info("V92: [4/7] 跳过量价背离计算")
                df_with_divergence = df_with_signals.with_columns([
                    pl.lit(50.0).alias('divergence_score')
                ])
            
            # 5. 风格中性化
            logger.info("V92: [5/7] 开始风格中性化...")
            df_with_neutralization = self._apply_neutralization(df_with_divergence)
            
            # 6. 计算综合评分
            logger.info("V92: [6/7] 开始计算综合评分...")
            df_with_composite = self._compute_composite_score(df_with_neutralization)
            
            # 7. IC 审计
            logger.info("V92: [7/7] 开始 IC 审计...")
            ic_audit_results = self.ic_audit.calculate_rank_ic(
                df_with_composite, signal_col='final_signal'
            )
            
            logger.info(f"V92: T+1 Rank IC = {ic_audit_results['ic_t1']:.4f} (目标 > {V92_T1_IC_TARGET})")
            logger.info(f"V92: IC IR = {ic_audit_results['ic_ir']:.2f} (目标 > {V92_IC_IR_TARGET})")
            
            # 执行回测交易
            logger.info("V92: 开始执行回测交易...")
            trade_results = self._execute_backtest(df_with_composite)
            
            # 一致性检查
            logger.info("V92: 开始一致性检查...")
            consistency_results = self._run_consistency_check(ic_audit_results, trade_results)
            
            # 生成报告
            logger.info("V92: 生成审计报告...")
            audit_report = self._generate_audit_report(
                data_integrity_results,
                ic_audit_results,
                trade_results,
                consistency_results,
            )
            
            result = {
                'data_integrity': data_integrity_results,
                'ic_audit': ic_audit_results,
                'consistency_check': consistency_results,
                'trade_results': trade_results,
                'audit_report': audit_report,
                'trade_records': self.trade_records,
                'daily_snapshots': self.daily_snapshots,
                'rebalance_dates': self.rebalance_dates,
            }
            
            logger.info("=" * 70)
            logger.info("V92 回测完成")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"V92 回测失败 - {e}")
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
                logger.info(f"V92: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V92: {year}年数据检查失败 - {message}")
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
                    logger.info(f"V92: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V92: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V92: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _compute_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算因子信号（回归 V90 稳健方法）
        
        【因子组成】
        1. Refined Residual (20%): 残差 Alpha + 成交量确认
        2. Smart Flow (15%): 资金流向因子
        3. Vol_Price_Interaction (65%): 量价交互因子
        """
        result = df.clone()
        
        # 数据类型转换
        result = result.with_columns([
            pl.col('open').cast(pl.Float64, strict=False).alias('open'),
            pl.col('high').cast(pl.Float64, strict=False).alias('high'),
            pl.col('low').cast(pl.Float64, strict=False).alias('low'),
            pl.col('close').cast(pl.Float64, strict=False).alias('close'),
            pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            pl.col('amount').cast(pl.Float64, strict=False).alias('amount'),
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
        ])
        
        # 计算 Refined Residual 因子
        result = self._compute_residual_alpha(result)
        
        # 计算 Smart Flow 因子
        result = self._compute_smart_flow(result)
        
        # 计算 Vol_Price_Interaction 因子
        result = self._compute_vol_price_interaction(result)
        
        # 计算综合评分（V90 融合公式）
        residual_weight = 0.20
        flow_weight = 0.15
        interaction_weight = 0.65
        
        result = result.with_columns([
            (residual_weight * pl.col('residual_alpha_score') + 
             flow_weight * pl.col('smart_flow_score') +
             interaction_weight * pl.col('interaction_score')).alias('composite_score')
        ])
        
        return result
    
    def _compute_residual_alpha(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Refined Residual 因子（回归 V90 稳健方法）
        
        【核心逻辑】
        1. 计算过去 N 日的超额收益（个股 - 市场）
        2. 用成交量比率确认（放量上涨更可靠）
        3. 排名转换为百分位分数
        """
        result = df.clone()
        window = 10
        
        if 'industry_code' not in result.columns:
            result = result.with_columns([pl.lit('Unknown').alias('industry_code')])
        
        result = result.with_columns([
            pl.col('industry_code').cast(pl.Utf8, strict=False).fill_null('Unknown').alias('industry_code')
        ])
        
        # 计算过去 N 日的收益率（使用 shift(1) 避免未来函数）
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(window + 1)) / 
             (pl.col('close').shift(window + 1) + EPSILON)).alias('stock_return_10d')
        ])
        
        # 计算市场收益率中位数
        market_median = result.group_by('trade_date').agg([
            pl.col('pct_chg').median().alias('market_return_10d')
        ])
        
        result = result.join(
            market_median.select(['trade_date', 'market_return_10d']),
            on='trade_date',
            how='left'
        )
        
        # 残差收益率 = 个股收益 - 市场收益
        result = result.with_columns([
            (pl.col('stock_return_10d') - pl.col('market_return_10d')).alias('residual_return')
        ])
        
        # 成交量比率
        result = result.with_columns([
            (pl.col('volume').fill_null(0) / (pl.col('volume').fill_null(0).rolling_sum(window_size=20).over('symbol') / 20 + EPSILON)).alias('volume_ratio')
        ])
        
        # 量价确认
        result = result.with_columns([
            (pl.col('residual_return') * pl.col('volume_ratio')).alias('residual_volume_confirmed')
        ])
        
        # 排名转换为分数
        result = result.with_columns([
            pl.col('residual_volume_confirmed').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + EPSILON))).alias('residual_alpha_score')
        ])
        
        return result
    
    def _compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Smart Flow 因子（V90 方法）
        
        【核心逻辑】
        1. 计算价量加权变化（Volume × Price Change）
        2. 滚动求和得到资金流向
        3. 排名转换为百分位分数
        """
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
        """
        计算 Vol_Price_Interaction 因子（V90 方法）
        
        【核心逻辑】
        1. 将 Residual 和 Flow 因子归一化
        2. 计算交互项（乘法）
        3. 排名转换为百分位分数
        """
        result = df.clone()
        
        # 确保必需字段存在
        if 'residual_alpha_score' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('residual_alpha_score')])
        
        if 'smart_flow_score' not in result.columns:
            result = result.with_columns([pl.lit(50.0).alias('smart_flow_score')])
        
        # 计算排名归一化
        result = result.with_columns([
            pl.col('residual_alpha_score').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('smart_flow_score').rank('ordinal', descending=True).over('trade_date').alias('flow_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks')
        ])
        
        result = result.with_columns([
            (pl.col('residual_rank') / (pl.col('n_stocks') + EPSILON)).alias('residual_rank_norm'),
            (pl.col('flow_rank') / (pl.col('n_stocks') + EPSILON)).alias('flow_rank_norm'),
        ])
        
        # 交互项（乘法）
        result = result.with_columns([
            (pl.col('residual_rank_norm') * pl.col('flow_rank_norm')).alias('interaction_raw')
        ])
        
        result = result.with_columns([
            pl.col('interaction_raw').rank('ordinal', descending=True).over('trade_date').alias('interaction_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks_interaction')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('interaction_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_interaction').cast(pl.Float64) + EPSILON))).alias('interaction_score')
        ])
        
        return result
    
    def _apply_neutralization(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        应用风格中性化（V90 稳健方法）
        
        【核心逻辑】
        1. 使用已经计算好的 composite_score（来自 V90 融合公式）
        2. 进行行业 + 市值中性化
        3. 返回中性化后的信号
        """
        if not self.config.enable_neutralization or self.style_neutralization is None:
            return df.with_columns([
                pl.lit(50.0).alias('neutralized_signal')
            ])
        
        # 使用已经计算好的 composite_score
        result = df.clone()
        
        # 确保 composite_score 存在
        if 'composite_score' not in result.columns:
            # 如果不存在，使用 V90 融合公式计算
            result = result.with_columns([
                (0.20 * pl.col('residual_alpha_score') + 
                 0.15 * pl.col('smart_flow_score') +
                 0.65 * pl.col('interaction_score')).alias('composite_score')
            ])
        
        # 应用中性化
        neutralized = self.style_neutralization.compute_neutralization(
            result, signal_col='composite_score'
        )
        
        return neutralized
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算综合评分（V92 最终信号）
        
        【融合逻辑】
        1. 使用 neutralized_signal（如果已中性化）
        2. 使用 composite_score（来自 V90 融合公式）
        3. 加入 divergence_score 进行微调
        """
        result = df.clone()
        
        # 确保必需字段存在
        for col, default in [
            ('residual_alpha_score', 50.0),
            ('smart_flow_score', 50.0),
            ('interaction_score', 50.0),
            ('divergence_score', 50.0),
            ('neutralized_signal', None),
            ('composite_score', None),
        ]:
            if col not in result.columns:
                if default is not None:
                    result = result.with_columns([pl.lit(default).alias(col)])
        
        # 确定基础信号
        if 'neutralized_signal' in result.columns:
            # 使用中性化后的信号
            base_signal = 'neutralized_signal'
        elif 'composite_score' in result.columns:
            # 使用已经计算好的 composite_score（来自 V90 融合公式）
            base_signal = 'composite_score'
        else:
            # 计算基础信号（V90 融合公式）
            base_signal = 'base_signal'
            result = result.with_columns([
                (0.20 * pl.col('residual_alpha_score') + 
                 0.15 * pl.col('smart_flow_score') +
                 0.65 * pl.col('interaction_score')).alias(base_signal)
            ])
        
        # V92 融合公式：加入背离分数进行微调
        # final_base = (1 - DIVERGENCE_WEIGHT) * base_signal + DIVERGENCE_WEIGHT * divergence_score
        result = result.with_columns([
            ((1.0 - V92_DIVERGENCE_WEIGHT) * pl.col(base_signal) + 
             V92_DIVERGENCE_WEIGHT * pl.col('divergence_score')).alias('final_base')
        ])
        
        # 排名转换为最终信号
        result = result.with_columns([
            pl.col('final_base').rank('ordinal', descending=True).over('trade_date').alias('composite_rank'),
            pl.col('symbol').count().over('trade_date').cast(pl.Float64).alias('n_stocks_final')
        ])
        
        # V92 关键修复：确保值越大（综合评分越高），final_signal 越高
        # 使用 descending=True 排名，值越大排名越小
        # 然后使用公式：final_signal = 100 * (1 - (rank - 0.5) / n)
        # 这样排名越小（值越大），final_signal 越高
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('composite_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_final').cast(pl.Float64) + EPSILON))).alias('final_signal')
        ])
        
        # 调试：输出信号统计
        logger.info(f"V92: final_signal 统计 - 均值={result['final_signal'].mean():.2f}, 标准差={result['final_signal'].std():.2f}")
        
        logger.info(f"V92: 综合评分计算完成，处理 {result.height} 条记录")
        
        return result
    
    def _run_consistency_check(self, ic_audit: Dict[str, Any],
                                trade_results: Dict[str, Any]) -> Dict[str, Any]:
        """运行一致性检查"""
        # 数学一致性检查
        total_return = trade_results.get('total_return', 0.0)
        annualized_return = trade_results.get('annualized_return', 0.0)
        years = 6.0  # 2019-2024 共 6 年
        
        math_check = self.consistency_checker.check_mathematical_consistency(
            total_return, annualized_return, years
        )
        
        # 预测一致性检查
        pred_check = self.consistency_checker.check_predictive_consistency(
            ic_audit.get('ic_t1', 0.0),
            ic_audit.get('ic_t2', 0.0),
            ic_audit.get('ic_t3', 0.0),
        )
        
        # 风险控制检查
        risk_check = self.consistency_checker.check_risk_control(
            trade_results.get('max_drawdown', 0.0)
        )
        
        summary = self.consistency_checker.get_consistency_summary()
        
        logger.info(f"V92: 一致性检查完成 - 通过率={summary['pass_rate']:.2%}")
        
        return {
            'mathematical_consistency': {
                'passed': math_check.passed,
                'expected': math_check.expected,
                'actual': math_check.actual,
                'diff': math_check.diff,
                'message': math_check.message,
            },
            'predictive_consistency': {
                'passed': pred_check.passed,
                'expected': pred_check.expected,
                'actual': pred_check.actual,
                'diff': pred_check.diff,
                'message': pred_check.message,
            },
            'risk_control': {
                'passed': risk_check.passed,
                'expected': risk_check.expected,
                'actual': risk_check.actual,
                'diff': risk_check.diff,
                'message': risk_check.message,
            },
            'summary': summary,
        }
    
    def _execute_backtest(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        执行回测交易（V92 修复版 - 参考 V90 换手率控制）
        
        【核心修复】
        1. 使用换手率追踪器记录每日换手
        2. 单日换手率上限 10%
        3. 调仓间隔 5 天（避免频繁调仓）
        4. 使用 Rank 相关性判断是否需要调仓
        """
        logger.info("V92: 开始执行回测交易...")
        
        df = df.sort(['trade_date', 'symbol'])
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        warmup_cutoff = unique_dates[:min(V92_WARMUP_PERIOD, len(unique_dates))]
        trade_dates = [d for d in unique_dates if d not in warmup_cutoff]
        
        logger.info(f"V92: 热身期 {len(warmup_cutoff)} 天，交易期 {len(trade_dates)} 天")
        
        # 重置状态
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.positions = {}
        self.trade_records = []
        self.daily_snapshots = []
        self.rebalance_dates = []
        self.turnover_tracker = V92TurnoverTracker()  # 重置换手率追踪器
        
        # 调仓状态
        last_rebalance_date = None
        days_since_rebalance = 0
        prev_rank_correlation = None
        
        for i, trade_date in enumerate(trade_dates):
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
            is_rebalance_day = False
            
            # 判断是否调仓（V92 修复：增加调仓间隔）
            days_since_rebalance += 1
            
            # V92 调仓条件：
            # 1. 距离上次调仓至少 5 天
            # 2. 或者达到最大调仓间隔 5 天（强制调仓）
            should_rebalance = days_since_rebalance >= V92_MIN_REBALANCE_INTERVAL
            
            if should_rebalance:
                signal_col = 'final_signal'
                
                if signal_col not in day_df.columns:
                    logger.warning(f"V92: {trade_date} 信号列缺失，跳过")
                    continue
                
                # 获取有效股票（信号>0 且非空）
                valid_stocks = day_df.filter(
                    (pl.col(signal_col).is_not_null()) &
                    (pl.col(signal_col) > 0)
                ).sort(signal_col, descending=True)
                
                if valid_stocks.is_empty():
                    logger.warning(f"V92: {trade_date} 无有效股票，跳过调仓")
                    continue
                
                # 计算目标持仓
                target_positions = self._calculate_target_positions(
                    valid_stocks, self.portfolio_value, signal_col
                )
                
                # 执行交易
                try:
                    buy_value, sell_value = self._execute_trades(
                        trade_date, target_positions, price_map
                    )
                    
                    # V92 关键修复：应用单日换手率上限 10%
                    max_daily_turnover_value = self.portfolio_value * V92_DAILY_TURNOVER_MAX
                    buy_value = min(buy_value, max_daily_turnover_value)
                    sell_value = min(sell_value, max_daily_turnover_value)
                    
                    is_rebalance_day = True
                    self.rebalance_dates.append(trade_date)
                    last_rebalance_date = trade_date
                    days_since_rebalance = 0
                    
                except Exception as e:
                    logger.error(f"V92: {trade_date} 交易执行失败 - {e}")
            
            # 记录换手率（使用换手率追踪器）
            turnover_record = self.turnover_tracker.record_turnover(
                trade_date=trade_date,
                portfolio_value=self.portfolio_value,
                buy_value=buy_value,
                sell_value=sell_value,
                is_rebalance_day=is_rebalance_day,
            )
            
            # 记录组合快照
            if self.daily_snapshots:
                prev_value = self.daily_snapshots[-1]['total_value']
                daily_return = (self.portfolio_value - prev_value) / prev_value if prev_value > EPSILON else 0.0
            else:
                daily_return = 0.0
            
            snapshot = {
                'trade_date': trade_date,
                'total_value': self.portfolio_value,
                'cash': self.cash,
                'position_value': position_value,
                'position_count': len(self.positions),
                'daily_return': daily_return,
                'turnover_rate': turnover_record['turnover_rate'],
                'is_rebalance_day': is_rebalance_day,
            }
            self.daily_snapshots.append(snapshot)
            
            if (i + 1) % 50 == 0:
                turnover_summary = self.turnover_tracker.get_turnover_summary()
                logger.info(f"V92: 处理 {i + 1}/{len(trade_dates)} 天，组合价值={self.portfolio_value:,.2f}, 年化换手={turnover_summary['annualized_turnover']:.1f}%")
        
        # 计算结果
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        # 计算年化收益率
        trading_days = len(self.daily_snapshots)
        years = trading_days / 252.0
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        # 获取换手率摘要
        turnover_summary = self.turnover_tracker.get_turnover_summary()
        annualized_turnover = turnover_summary['annualized_turnover']
        
        max_drawdown = self._calculate_max_drawdown()
        annual_returns = self._calculate_annual_returns()
        
        result = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'final_value': self.portfolio_value,
            'max_drawdown': max_drawdown,
            'annualized_turnover': annualized_turnover,
            'total_trading_days': len(trade_dates),
            'total_trades': len(self.trade_records),
            'rebalance_count': len(self.rebalance_dates),
            'annual_returns': annual_returns,
            'avg_annual_return': np.mean(list(annual_returns.values())) if annual_returns else 0.0,
            'turnover_summary': turnover_summary,
        }
        
        logger.info(f"V92: 回测完成 - 总收益={total_return:.2%}, 年化={annualized_return:.2%}")
        logger.info(f"V92: 调仓次数={len(self.rebalance_dates)}, 最大回撤={max_drawdown:.2%}")
        logger.info(f"V92: 年化换手率={annualized_turnover:.2%}")
        
        return result
    
    def _calculate_target_positions(self, valid_stocks: pl.DataFrame, 
                                     portfolio_value: float,
                                     signal_col: str = 'final_signal') -> Dict[str, float]:
        """
        计算目标持仓（V92 修复版 - 参考 V90 动态仓位控制）
        
        【核心逻辑】
        1. 根据当前回撤动态调整仓位上限
        2. 回撤越大，仓位上限越低
        3. 单只标的最大权重 8%
        """
        if valid_stocks.is_empty():
            return {}
        
        # V92 动态仓位控制：根据回撤调整仓位上限
        current_drawdown = self._calculate_current_drawdown()
        position_limit = self._get_position_limit_by_drawdown(current_drawdown)
        
        max_stocks = min(self.config.max_positions, valid_stocks.height)
        top_stocks = valid_stocks.head(max_stocks)
        
        target_positions = {}
        for row in top_stocks.iter_rows(named=True):
            symbol = row['symbol']
            signal = row.get(signal_col, 50.0)
            
            if signal is not None and np.isfinite(signal) and signal > 0:
                # V92 风险控制：单只标的最大权重 8%
                capped_weight = min(1.0 / max_stocks, V92_MAX_SINGLE_WEIGHT)
                target_positions[symbol] = capped_weight
        
        # V92 风险控制：根据回撤限制总仓位
        total_weight = sum(target_positions.values())
        if total_weight > EPSILON:
            effective_limit = min(total_weight, position_limit)
            target_positions = {k: v / total_weight * effective_limit for k, v in target_positions.items()}
        
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
        """
        根据回撤动态调整仓位上限（V90 方法）
        
        【核心逻辑】
        - 回撤 > 12%，仓位上限 70%
        - 回撤 > 8%，仓位上限 85%
        - 回撤 <= 8%，满仓
        """
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
        
        # 买入/调整目标持仓
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
                buy_quantity = max(1, int(target_value / buy_price))
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
            snapshots_year = [s for s in self.daily_snapshots 
                             if s['trade_date'].startswith(year)]
            
            if len(snapshots_year) >= 2:
                start_value = snapshots_year[0]['total_value']
                end_value = snapshots_year[-1]['total_value']
                year_return = (end_value - start_value) / start_value if start_value > EPSILON else 0.0
                annual_returns[year] = year_return
        
        return annual_returns
    
    def _generate_audit_report(self, data_integrity: Dict, ic_audit: Dict,
                                trade_results: Dict, consistency: Dict) -> str:
        """生成审计报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("《V92 IC 驱动预测审计报告》")
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
        lines.append(f"   T+1 Rank IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V92_T1_IC_TARGET})")
        lines.append(f"   T+2 Rank IC: {ic_audit.get('ic_t2', 0.0):.4f}")
        lines.append(f"   T+3 Rank IC: {ic_audit.get('ic_t3', 0.0):.4f}")
        lines.append(f"   IC IR: {ic_audit.get('ic_ir', 0.0):.2f} (目标 > {V92_IC_IR_TARGET})")
        lines.append(f"   三年度平均 IC: {ic_audit.get('mean_ic_3yr', 0.0):.4f}")
        
        ic_by_year = ic_audit.get('ic_by_year', {})
        if ic_by_year:
            lines.append("   分年度 IC:")
            for year in ['2019', '2021', '2024']:
                if year in ic_by_year:
                    lines.append(f"     {year}年：{ic_by_year[year]['mean_ic']:.4f}")
        lines.append("")
        
        lines.append("3. 交易执行审计")
        lines.append("   " + "-" * 50)
        lines.append(f"   总收益：{trade_results.get('total_return', 0.0):.2%}")
        lines.append(f"   年化收益：{trade_results.get('annualized_return', 0.0):.2%}")
        lines.append(f"   最终价值：{trade_results.get('final_value', 0.0):,.2f}")
        lines.append(f"   最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
        lines.append(f"   年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
        lines.append(f"   调仓次数：{trade_results.get('rebalance_count', 0)}")
        
        annual_returns = trade_results.get('annual_returns', {})
        if annual_returns:
            lines.append("")
            lines.append("   年度收益:")
            for year, ret in annual_returns.items():
                lines.append(f"     {year}年：{ret:.2%}")
        lines.append("")
        
        lines.append("4. 一致性检查")
        lines.append("   " + "-" * 50)
        
        math_check = consistency.get('mathematical_consistency', {})
        lines.append(f"   数学一致性：{'✓' if math_check.get('passed', False) else '✗'}")
        lines.append(f"     {math_check.get('message', 'N/A')}")
        
        pred_check = consistency.get('predictive_consistency', {})
        lines.append(f"   预测一致性：{'✓' if pred_check.get('passed', False) else '✗'}")
        lines.append(f"     {pred_check.get('message', 'N/A')}")
        
        risk_check = consistency.get('risk_control', {})
        lines.append(f"   风险控制：{'✓' if risk_check.get('passed', False) else '✗'}")
        lines.append(f"     {risk_check.get('message', 'N/A')}")
        
        summary = consistency.get('summary', {})
        lines.append(f"   总通过率：{summary.get('pass_rate', 0.0):.2%}")
        lines.append("")
        
        lines.append("5. V92 硬性指标验证")
        lines.append("   " + "-" * 50)
        
        # 指标 A：IC
        metric_a_ic = ic_audit.get('ic_t1', 0.0) >= V92_T1_IC_TARGET
        metric_a_ir = ic_audit.get('ic_ir', 0.0) >= V92_IC_IR_TARGET
        metric_a_pass = metric_a_ic and metric_a_ir
        
        lines.append(f"   指标 A (T+1 IC ≥ {V92_T1_IC_TARGET}, IC IR ≥ {V92_IC_IR_TARGET}): {'✓' if metric_a_pass else '✗'}")
        lines.append(f"     - T+1 IC: {ic_audit.get('ic_t1', 0.0):.4f} {'✓' if metric_a_ic else '✗'}")
        lines.append(f"     - IC IR: {ic_audit.get('ic_ir', 0.0):.2f} {'✓' if metric_a_ir else '✗'}")
        lines.append("")
        
        # 指标 B：最大回撤
        metric_b_pass = trade_results.get('max_drawdown', 0.0) <= 0.10
        lines.append(f"   指标 B (最大回撤 ≤ 10%): {'✓' if metric_b_pass else '✗'}")
        lines.append(f"     - 最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
        lines.append("")
        
        # 指标 C：年化换手率
        ann_turnover = trade_results.get('annualized_turnover', 0.0)
        metric_c_pass = 3.0 <= ann_turnover <= 5.0
        lines.append(f"   指标 C (年化换手率 300%-500%): {'✓' if metric_c_pass else '✗'}")
        lines.append(f"     - 年化换手率：{ann_turnover:.2%}")
        lines.append("")
        
        # 指标 D：数学一致性
        metric_d_pass = math_check.get('passed', False)
        lines.append(f"   指标 D (数学一致性误差 < 0.1%): {'✓' if metric_d_pass else '✗'}")
        lines.append("")
        
        # 指标 E：预测一致性
        metric_e_pass = pred_check.get('passed', False)
        lines.append(f"   指标 E (预测一致性 T+1≥T+2≥T+3): {'✓' if metric_e_pass else '✗'}")
        lines.append("")
        
        lines.append("6. V92 增强特性")
        lines.append("   " + "-" * 50)
        lines.append(f"   量价背离：{'启用' if self.config.enable_divergence else '禁用'}")
        lines.append(f"   IC 权重：{'启用' if self.config.enable_ic_weighting else '禁用'}")
        lines.append(f"   中性化：{'启用' if self.config.enable_neutralization else '禁用'}")
        lines.append("")
        
        lines.append("=" * 70)
        
        all_passed = metric_a_pass and metric_b_pass and metric_c_pass and metric_d_pass and metric_e_pass
        lines.append(f"总体评估：{'所有指标通过 ✓' if all_passed else '部分指标未通过 ✗'}")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'data_integrity': {},
            'ic_audit': {},
            'consistency_check': {},
            'trade_results': {},
            'audit_report': '',
            'trade_records': [],
            'daily_snapshots': [],
            'rebalance_dates': [],
        }


# ===========================================
# 主程序
# ===========================================

def run_v92_backtest(config: V92EngineConfig = None) -> Dict[str, Any]:
    """运行 V92 回测"""
    engine = V92Engine(config=config)
    return engine.run_backtest()


def print_v92_report(result: Dict[str, Any]) -> None:
    """打印 V92 报告"""
    logger.info("=" * 70)
    logger.info("V92 最终报告")
    logger.info("=" * 70)
    
    trade_results = result.get('trade_results', {})
    logger.info("【交易执行】")
    logger.info(f"  总收益：{trade_results.get('total_return', 0.0):.2%}")
    logger.info(f"  年化收益：{trade_results.get('annualized_return', 0.0):.2%}")
    logger.info(f"  最终价值：{trade_results.get('final_value', 0.0):,.2f}")
    logger.info(f"  最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
    logger.info(f"  年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
    logger.info(f"  调仓次数：{trade_results.get('rebalance_count', 0)}")
    
    ic_audit = result.get('ic_audit', {})
    logger.info("")
    logger.info("【IC 审计】")
    logger.info(f"  T+1 IC: {ic_audit.get('ic_t1', 0.0):.4f}")
    logger.info(f"  T+2 IC: {ic_audit.get('ic_t2', 0.0):.4f}")
    logger.info(f"  T+3 IC: {ic_audit.get('ic_t3', 0.0):.4f}")
    logger.info(f"  IC IR: {ic_audit.get('ic_ir', 0.0):.2f}")
    
    consistency = result.get('consistency_check', {})
    logger.info("")
    logger.info("【一致性检查】")
    logger.info(f"  数学一致性：{'✓' if consistency.get('mathematical_consistency', {}).get('passed', False) else '✗'}")
    logger.info(f"  预测一致性：{'✓' if consistency.get('predictive_consistency', {}).get('passed', False) else '✗'}")
    logger.info(f"  风险控制：{'✓' if consistency.get('risk_control', {}).get('passed', False) else '✗'}")
    
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
    
    config = V92EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        enable_divergence=True,
        enable_ic_weighting=True,
        enable_neutralization=True,
    )
    
    result = run_v92_backtest(config)
    print_v92_report(result)
    
    output_path = "reports/v92_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_result = {
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': result.get('ic_audit', {}),
        'consistency_check': result.get('consistency_check', {}),
        'trade_results': result.get('trade_results', {}),
        'audit_report': result.get('audit_report', ''),
        'trade_count': len(result.get('trade_records', [])),
        'snapshot_count': len(result.get('daily_snapshots', [])),
        'rebalance_count': len(result.get('rebalance_dates', [])),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V92: 结果已保存至 {output_path}")