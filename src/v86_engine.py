"""
V86 Engine - 因子稳定性增强与时空一致性审计

【V86 核心理念】
1. IC 衰减审计 (IC Decay Audit)
   - 计算 T+1, T+2, T+3 的 Rank IC
   - 检测高频噪声并输出衰减率

2. 动态 Regime 分类器 (Dynamic Regime Classifier)
   - 基于行业离散度和波动率偏度
   - 自动切换动量/反转策略

3. 行业中性化 2.0 (Industry Neutralization 2.0)
   - 横截面行业调整
   - 确保选股是个股超额

4. 指数退避重试 (Exponential Backoff Retry)
   - Database connection timeout 自动重试
   - Data missing 自动重试

【硬性指标】
- 指标 A (稳定性): 三年度 Mean Rank IC 均值 >= 0.045，且每个年度的 IC IR >= 0.5
- 指标 B (可交易性): T+2 延迟 Rank IC > 0.02
- 指标 C (纯净度): 最大单一行业持仓占比不得连续 3 天超过 30%

作者：量化系统
版本：V86.0
日期：2026-03-29
"""

import traceback
import time
import math
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import polars as pl
from loguru import logger

# 导入 V86 核心模块
try:
    from src.core.v86_core import (
        V86DataManager,
        V86AlphaCenter,
        V86RankICCalculator,
        V86ICDecayMetrics,
        V86RegimeState,
        V86_INITIAL_CAPITAL,
        V86_MAX_POSITIONS,
        V86_WARMUP_PERIOD,
        V86_COMMISSION_RATE,
        V86_MIN_COMMISSION,
        V86_STAMP_DUTY,
        V86_TRANSFER_FEE,
        V86_RANK_IC_TARGET_MIN,
        V86_RANK_IC_IR_TARGET,
        V86_RANK_IC_OOS_YEARS,
        V86_MIN_STOCK_DAILY_ROWS,
        V86_IC_DECAY_THRESHOLD,
        V86_MAX_SINGLE_INDUSTRY_PCT,
        V86_MAX_CONSECUTIVE_DAYS_OVER_LIMIT,
        EPSILON,
    )
except ImportError:
    from core.v86_core import (
        V86DataManager,
        V86AlphaCenter,
        V86RankICCalculator,
        V86ICDecayMetrics,
        V86RegimeState,
        V86_INITIAL_CAPITAL,
        V86_MAX_POSITIONS,
        V86_WARMUP_PERIOD,
        V86_COMMISSION_RATE,
        V86_MIN_COMMISSION,
        V86_STAMP_DUTY,
        V86_TRANSFER_FEE,
        V86_RANK_IC_TARGET_MIN,
        V86_RANK_IC_IR_TARGET,
        V86_RANK_IC_OOS_YEARS,
        V86_MIN_STOCK_DAILY_ROWS,
        V86_IC_DECAY_THRESHOLD,
        V86_MAX_SINGLE_INDUSTRY_PCT,
        V86_MAX_CONSECUTIVE_DAYS_OVER_LIMIT,
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
# V86 引擎配置
# ===========================================

@dataclass
class V86EngineConfig:
    """V86 引擎配置"""
    # 回测配置
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = V86_INITIAL_CAPITAL  # 严禁修改
    max_positions: int = V86_MAX_POSITIONS
    warmup_period: int = V86_WARMUP_PERIOD
    
    # 费率配置（严禁修改）
    commission_rate: float = V86_COMMISSION_RATE  # 0.2%
    min_commission: float = V86_MIN_COMMISSION
    stamp_duty: float = V86_STAMP_DUTY
    transfer_fee: float = V86_TRANSFER_FEE
    
    # OOS 测试年份
    oos_years: List[str] = None
    
    # Regime 分类器配置
    regime_lookback: int = 20
    regime_dispersion_threshold: float = 0.6
    
    # 行业中性化配置
    industry_neutral_window: int = 20
    max_single_industry_pct: float = V86_MAX_SINGLE_INDUSTRY_PCT
    
    def __post_init__(self):
        if self.oos_years is None:
            self.oos_years = V86_RANK_IC_OOS_YEARS


# ===========================================
# V86 引擎
# ===========================================

class V86Engine:
    """
    V86 回测引擎 - 因子稳定性增强与时空一致性审计
    
    【核心功能】
    1. IC 衰减审计 (T+1, T+2, T+3)
    2. 动态 Regime 分类器
    3. 行业中性化 2.0
    4. 指数退避重试
    """
    
    def __init__(self, config: V86EngineConfig = None, db=None):
        self.config = config or V86EngineConfig()
        
        # 初始化数据库
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V86: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        # 初始化组件
        self.data_manager = V86DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
            'retry_attempts': 5,
            'retry_base_delay': 1.0,
            'retry_max_delay': 30.0,
        })
        self.alpha_center = V86AlphaCenter(config={
            'regime_lookback': self.config.regime_lookback,
            'regime_dispersion_threshold': self.config.regime_dispersion_threshold,
            'industry_neutral_window': self.config.industry_neutral_window,
        })
        self.rank_ic_calculator = V86RankICCalculator(db=self.db, config={
            'ic_decay_max_lag': 3,
            'ic_decay_threshold': V86_IC_DECAY_THRESHOLD,
        })
        
        # 状态变量
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        # 绩效指标
        self.total_return = 0.0
        self.max_drawdown = 0.0
        
        logger.info("V86 Engine 初始化完成")
        logger.info(f"V86: 初始资金={self.config.initial_capital:,.2f} (严禁修改)")
        logger.info(f"V86: 手续费={self.config.commission_rate:.2%} (严禁修改)")
        logger.info(f"V86: OOS 测试年份={self.config.oos_years}")
        logger.info(f"V86: Rank IC 目标 >= {V86_RANK_IC_TARGET_MIN}")
        logger.info(f"V86: IC IR 目标 >= {V86_RANK_IC_IR_TARGET}")
        logger.info(f"V86: T+2 IC 目标 > {V86_IC_DECAY_THRESHOLD}")
        logger.info("V86: IC 衰减审计已启用 (T+1, T+2, T+3)")
        logger.info("V86: 动态 Regime 分类器已启用")
        logger.info("V86: 行业中性化 2.0 已启用")
        logger.info("V86: 指数退避重试已启用")
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 60)
        logger.info("V86 回测引擎启动")
        logger.info("=" * 60)
        
        if self.db is None:
            logger.error("V86: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查（带重试）
            logger.info("V86: 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据（带重试）
            logger.info("V86: 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V86: 未加载到任何数据")
                return self._empty_result()
            
            # 记录参与计算的股票数量
            for year in self.config.oos_years:
                df_year = df.filter(pl.col('trade_date').str.starts_with(year))
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V86: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算因子信号
            logger.info("V86: 开始计算因子信号...")
            df_with_signals = self._compute_signals(df)
            
            # 4. 行业中性化 2.0
            logger.info("V86: 开始行业中性化 2.0...")
            df_with_signals = self.alpha_center.compute_industry_neutralization(
                df_with_signals, score_col='composite_score'
            )
            
            # 5. IC 衰减审计
            logger.info("V86: 开始 IC 衰减审计...")
            ic_decay_results = self.rank_ic_calculator.calculate_ic_decay_series(
                df_with_signals, signal_col='composite_score'
            )
            
            # 6. 动态 Regime 分类
            logger.info("V86: 开始动态 Regime 分类...")
            regime_summary = self._compute_regime_classification(df)
            
            # 7. 打印 IC 衰减报告
            ic_decay_report = self.rank_ic_calculator.generate_ic_decay_report()
            logger.info("")
            logger.info(ic_decay_report)
            
            # 8. 验证硬性指标
            hard_metrics = self._verify_hard_metrics()
            
            # 9. 生成重试报告
            retry_report = self.data_manager.get_retry_report()
            
            # 10. 生成 V86 特征稳定性与衰减分析报告
            stability_report = self._generate_stability_report(
                data_integrity_results, regime_summary, retry_report
            )
            
            # 11. 生成结果
            result = {
                'ic_decay_summary': self.rank_ic_calculator.get_ic_decay_summary(),
                'year_ic_stats': self.rank_ic_calculator.get_year_ic_stats(),
                'regime_summary': regime_summary,
                'hard_metrics': hard_metrics,
                'data_integrity': data_integrity_results,
                'retry_report': retry_report,
                'stability_report': stability_report,
                'ic_decay_report': ic_decay_report,
            }
            
            logger.info("=" * 60)
            logger.info("V86 回测完成")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"V86 回测失败 - {e}")
            logger.error(traceback.format_exc())
            
            return self._empty_result()
    
    def _check_data_integrity(self) -> Dict[str, Dict[str, Any]]:
        """检查数据完整性（带重试）"""
        results = {}
        
        for year in self.config.oos_years:
            passed, message = self.data_manager.check_data_integrity(year)
            trading_days = self.data_manager.get_trading_days_count(year)
            
            results[year] = {
                'passed': passed,
                'message': message,
                'trading_days': trading_days,
                'min_required': V86_MIN_STOCK_DAILY_ROWS,
            }
            
            if passed:
                logger.info(f"V86: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V86: {year}年数据检查失败 - {message}")
        
        return results
    
    def _load_data(self) -> pl.DataFrame:
        """加载数据（带重试）"""
        all_dfs = []
        
        for year in self.config.oos_years:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            try:
                df = self.data_manager.load_stock_data(start_date, end_date)
                if not df.is_empty():
                    all_dfs.append(df)
                    logger.info(f"V86: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V86: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V86: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _compute_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算因子信号"""
        result = df.clone()
        
        # 确保必要的数据类型
        result = result.with_columns([
            pl.col('open').cast(pl.Float64, strict=False).alias('open'),
            pl.col('high').cast(pl.Float64, strict=False).alias('high'),
            pl.col('low').cast(pl.Float64, strict=False).alias('low'),
            pl.col('close').cast(pl.Float64, strict=False).alias('close'),
            pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            pl.col('amount').cast(pl.Float64, strict=False).alias('amount'),
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
        ])
        
        # 计算基础因子（使用 V85 的逻辑作为基础）
        # 1. Refined_Residual - 残差收益因子
        result = self._compute_refined_residual(result)
        
        # 2. Smart_Flow - 资金流因子
        result = self._compute_smart_flow(result)
        
        # 3. Vol_Price_Interaction - 交互因子
        result = self._compute_vol_price_interaction(result)
        
        # 4. 计算综合评分
        result = self._compute_composite_score(result)
        
        return result
    
    def _compute_refined_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Refined_Residual 因子"""
        result = df.clone()
        window = 5
        
        # 计算个股 5 日收益率
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(window + 1)) / 
             (pl.col('close').shift(window + 1) + EPSILON)).alias('stock_return_5d')
        ])
        
        # 计算行业中位数收益
        industry_median = result.group_by(['industry_code', 'trade_date']).agg([
            pl.col('pct_chg').median().alias('industry_return_5d')
        ])
        
        result = result.join(
            industry_median.select(['industry_code', 'trade_date', 'industry_return_5d']),
            on=['industry_code', 'trade_date'],
            how='left'
        )
        
        # 计算残差收益
        result = result.with_columns([
            (pl.col('stock_return_5d') - pl.col('industry_return_5d')).alias('residual_return')
        ])
        
        # 横截面排名
        result = result.with_columns([
            pl.col('residual_return').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + EPSILON))).alias('refined_residual_score')
        ])
        
        return result
    
    def _compute_smart_flow(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 Smart_Flow 因子"""
        result = df.clone()
        window = 10
        
        # 计算成交量加权价格变化
        result = result.with_columns([
            (pl.col('close') - pl.col('open')).alias('price_change'),
            pl.col('volume').fill_null(0.0).alias('volume_filled')
        ])
        
        result = result.with_columns([
            (pl.col('volume_filled') * pl.col('price_change')).alias('volume_weighted_change')
        ])
        
        result = result.sort(['symbol', 'trade_date'])
        
        # 滚动和
        result = result.with_columns([
            pl.col('volume_weighted_change')
            .rolling_sum(window_size=window)
            .over('symbol')
            .alias('flow_sum')
        ])
        
        result = result.with_columns([
            pl.col('volume_filled')
            .rolling_sum(window_size=window)
            .over('symbol')
            .alias('volume_sum')
        ])
        
        # Smart_Flow
        result = result.with_columns([
            (pl.col('flow_sum') / (pl.col('volume_sum') + EPSILON)).alias('smart_flow_raw')
        ])
        
        # 横截面排名
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
        
        # 确保排名列存在
        if 'residual_rank' not in result.columns:
            result = result.with_columns([
                pl.col('refined_residual_score').rank('ordinal', descending=True).over('trade_date').alias('residual_rank')
            ])
        
        if 'flow_rank' not in result.columns:
            result = result.with_columns([
                pl.col('smart_flow_score').rank('ordinal', descending=True).over('trade_date').alias('flow_rank')
            ])
        
        # 归一化排名
        result = result.with_columns([
            (pl.col('residual_rank') / (pl.col('n_stocks').cast(pl.Float64) + EPSILON)).alias('residual_rank_norm'),
            (pl.col('flow_rank') / (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON)).alias('flow_rank_norm')
        ])
        
        # 交互项
        result = result.with_columns([
            (pl.col('residual_rank_norm') * pl.col('flow_rank_norm')).alias('interaction_raw')
        ])
        
        # 横截面排名
        result = result.with_columns([
            pl.col('interaction_raw').rank('ordinal', descending=True).over('trade_date').alias('interaction_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_interaction')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('interaction_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_interaction').cast(pl.Float64) + EPSILON))).alias('vol_price_interaction_score')
        ])
        
        return result
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算综合评分"""
        result = df.clone()
        
        # 确保因子列存在
        for col, default in [
            ('refined_residual_score', 50.0),
            ('smart_flow_score', 50.0),
            ('vol_price_interaction_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # 权重配置
        residual_weight = 0.20
        flow_weight = 0.10
        interaction_weight = 0.70
        
        # 综合评分
        result = result.with_columns([
            (residual_weight * pl.col('refined_residual_score') + 
             flow_weight * pl.col('smart_flow_score') +
             interaction_weight * pl.col('vol_price_interaction_score')).alias('composite_score')
        ])
        
        return result
    
    def _compute_regime_classification(self, df: pl.DataFrame) -> Dict[str, Any]:
        """计算动态 Regime 分类"""
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        for trade_date in unique_dates:
            self.alpha_center.compute_regime_classifier(df, trade_date)
        
        return self.alpha_center.get_regime_summary()
    
    def _verify_hard_metrics(self) -> Dict[str, Any]:
        """验证硬性指标"""
        ic_decay_summary = self.rank_ic_calculator.get_ic_decay_summary()
        year_ic_stats = self.rank_ic_calculator.get_year_ic_stats()
        ic_ir_pass, ic_ir_values = self.rank_ic_calculator.check_ic_ir_target()
        
        # 指标 A: 三年度 Mean Rank IC >= 0.045 且 IC IR >= 0.5
        valid_years = [y for y in self.config.oos_years if y in year_ic_stats]
        if valid_years:
            avg_ic_t1 = np.mean([year_ic_stats[y]['mean_ic_t1'] for y in valid_years])
        else:
            avg_ic_t1 = 0.0
        
        metric_a_pass = (avg_ic_t1 >= V86_RANK_IC_TARGET_MIN) and ic_ir_pass
        
        # 指标 B: T+2 IC > 0.02
        metric_b_pass = ic_decay_summary.get('t2_ic_pass', False)
        
        # 指标 C: 行业集中度检查（需要持仓数据）
        is_violated, violation_dates = self.alpha_center.check_concentration_violation()
        metric_c_pass = not is_violated
        
        return {
            'metric_a_pass': metric_a_pass,
            'metric_a_details': {
                'avg_ic_t1': avg_ic_t1,
                'target': V86_RANK_IC_TARGET_MIN,
                'ic_ir_pass': ic_ir_pass,
                'ic_ir_values': ic_ir_values,
            },
            'metric_b_pass': metric_b_pass,
            'metric_b_details': {
                'mean_ic_t2': ic_decay_summary.get('mean_ic_t2', 0.0),
                'target': V86_IC_DECAY_THRESHOLD,
            },
            'metric_c_pass': metric_c_pass,
            'metric_c_details': {
                'is_violated': is_violated,
                'violation_dates': violation_dates[:10] if violation_dates else [],
            },
        }
    
    def _generate_stability_report(self, data_integrity_results: Dict,
                                    regime_summary: Dict,
                                    retry_report: str) -> str:
        """生成《V86 特征稳定性与衰减分析报告》"""
        ic_decay_summary = self.rank_ic_calculator.get_ic_decay_summary()
        year_ic_stats = self.rank_ic_calculator.get_year_ic_stats()
        hard_metrics = self._verify_hard_metrics()
        
        lines = [
            "=" * 70,
            "《V86 特征稳定性与衰减分析报告》",
            "=" * 70,
            "",
            "1. 数据完整性审计",
            "   " + "-" * 50,
        ]
        
        for year, result in data_integrity_results.items():
            status = "✓" if result['passed'] else "✗"
            lines.append(f"   {year}年：{status} {result['message']}")
        
        lines.extend([
            "",
            "2. IC 衰减审计",
            "   " + "-" * 50,
            f"   T+1 Rank IC: {ic_decay_summary['mean_ic_t1']:.4f}",
            f"   T+2 Rank IC: {ic_decay_summary['mean_ic_t2']:.4f} (目标：> {V86_IC_DECAY_THRESHOLD})",
            f"   T+3 Rank IC: {ic_decay_summary['mean_ic_t3']:.4f}",
            f"   平均衰减率：{ic_decay_summary['mean_decay_rate']:.2%}",
            f"   高频噪声占比：{ic_decay_summary['high_frequency_noise_ratio']:.2%}",
            "",
            "   【年度 IC 对比】",
        ])
        
        for year in self.config.oos_years:
            if year in year_ic_stats:
                stat = year_ic_stats[year]
                lines.append(f"   {year}年：T+1={stat['mean_ic_t1']:.4f}, T+2={stat['mean_ic_t2']:.4f}, T+3={stat['mean_ic_t3']:.4f}")
        
        lines.extend([
            "",
            "3. 动态 Regime 分类统计",
            "   " + "-" * 50,
            f"   动量行情天数：{regime_summary.get('momentum_days', 0)}",
            f"   反转行情天数：{regime_summary.get('reversal_days', 0)}",
            f"   动量占比：{regime_summary.get('momentum_ratio', 0.0):.2%}",
            f"   平均行业离散度：{regime_summary.get('avg_dispersion', 0.0):.4f}",
            f"   平均波动率偏度：{regime_summary.get('avg_skew', 0.0):.4f}",
            "",
            "4. 行业中性化 2.0",
            "   " + "-" * 50,
            "   横截面行业调整已启用",
            "   选股逻辑：个股超额收益（非行业轮动）",
            "",
            "5. 指数退避重试统计",
            "   " + "-" * 50,
        ])
        
        # 解析重试报告
        for line in retry_report.split('\n'):
            if line.strip():
                lines.append(f"   {line}")
        
        lines.extend([
            "",
            "6. 硬性指标验证",
            "   " + "-" * 50,
            f"   指标 A (稳定性): {'✓' if hard_metrics['metric_a_pass'] else '✗'}",
            f"     - 三年度 Mean IC: {hard_metrics['metric_a_details']['avg_ic_t1']:.4f} (目标：>= {V86_RANK_IC_TARGET_MIN})",
            f"     - IC IR 达标：{'✓' if hard_metrics['metric_a_details']['ic_ir_pass'] else '✗'}",
            "",
            f"   指标 B (可交易性): {'✓' if hard_metrics['metric_b_pass'] else '✗'}",
            f"     - T+2 IC: {hard_metrics['metric_b_details']['mean_ic_t2']:.4f} (目标：> {V86_IC_DECAY_THRESHOLD})",
            "",
            f"   指标 C (纯净度): {'✓' if hard_metrics['metric_c_pass'] else '✗'}",
            f"     - 行业集中度违规：{'是' if hard_metrics['metric_c_details']['is_violated'] else '否'}",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'ic_decay_summary': {},
            'year_ic_stats': {},
            'regime_summary': {},
            'hard_metrics': {},
            'data_integrity': {},
            'retry_report': '',
            'stability_report': '',
            'ic_decay_report': '',
        }


# ===========================================
# 主程序
# ===========================================

def run_v86_backtest(config: V86EngineConfig = None) -> Dict[str, Any]:
    """运行 V86 回测"""
    engine = V86Engine(config=config)
    return engine.run_backtest()


def print_v86_report(result: Dict[str, Any]):
    """打印 V86 报告"""
    logger.info("=" * 60)
    logger.info("V86 最终报告")
    logger.info("=" * 60)
    
    # IC 衰减摘要
    ic_decay_summary = result.get('ic_decay_summary', {})
    logger.info("【IC 衰减统计】")
    logger.info(f"  T+1 Rank IC: {ic_decay_summary.get('mean_ic_t1', 0.0):.4f}")
    logger.info(f"  T+2 Rank IC: {ic_decay_summary.get('mean_ic_t2', 0.0):.4f}")
    logger.info(f"  T+3 Rank IC: {ic_decay_summary.get('mean_ic_t3', 0.0):.4f}")
    logger.info(f"  高频噪声占比：{ic_decay_summary.get('high_frequency_noise_ratio', 0.0):.2%}")
    
    # Regime 统计
    regime_summary = result.get('regime_summary', {})
    logger.info("")
    logger.info("【Regime 分类统计】")
    logger.info(f"  动量行情天数：{regime_summary.get('momentum_days', 0)}")
    logger.info(f"  反转行情天数：{regime_summary.get('reversal_days', 0)}")
    logger.info(f"  动量占比：{regime_summary.get('momentum_ratio', 0.0):.2%}")
    
    # 硬性指标
    hard_metrics = result.get('hard_metrics', {})
    logger.info("")
    logger.info("【硬性指标验证】")
    logger.info(f"  指标 A (稳定性): {'✓' if hard_metrics.get('metric_a_pass') else '✗'}")
    logger.info(f"  指标 B (可交易性): {'✓' if hard_metrics.get('metric_b_pass') else '✗'}")
    logger.info(f"  指标 C (纯净度): {'✓' if hard_metrics.get('metric_c_pass') else '✗'}")
    
    # 稳定性报告
    stability_report = result.get('stability_report', '')
    if stability_report:
        logger.info("")
        logger.info(stability_report)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 运行回测
    config = V86EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
    )
    
    result = run_v86_backtest(config)
    print_v86_report(result)
    
    # 保存结果
    output_path = "reports/v86_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换结果为可序列化格式
    serializable_result = {
        'ic_decay_summary': result.get('ic_decay_summary', {}),
        'year_ic_stats': result.get('year_ic_stats', {}),
        'regime_summary': result.get('regime_summary', {}),
        'hard_metrics': result.get('hard_metrics', {}),
        'data_integrity': result.get('data_integrity', {}),
        'retry_report': result.get('retry_report', ''),
        'stability_report': result.get('stability_report', ''),
        'ic_decay_report': result.get('ic_decay_report', ''),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V86: 结果已保存至 {output_path}")