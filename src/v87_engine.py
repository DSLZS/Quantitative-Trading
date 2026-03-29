"""
V87 Engine - 多时空尺度融合与组合权重熵优化

【V87 核心理念】
1. 多时空尺度融合 (Multi-Horizon Fusion)
   - 计算 T-1, T-3, T-5 的 Alpha 信号
   - 使用衰减加权均值生成最终信号

2. 组合权重熵优化 (Entropy-Based Weighting)
   - 引入 Risk-Parity (风险平价) 思想
   - 权重公式：W_i ∝ Score_i / Volatility_i

3. 极端风险对冲模拟 (Fat-tail Stress Test)
   - 模拟"成分股跌停无法卖出"的情景

【硬性指标】
- 指标 A (换手率控制): 单日平均换手率 <= 15%
- 指标 B (回撤控制): 2024 年度最大回撤 <= 8%
- 指标 C (IC 衰减平滑度): T+1 到 T+3 的 IC 波动率下降 15%

作者：量化系统
版本：V87.0
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

# 导入 V87 核心模块
try:
    from src.core.v87_core import (
        V87DataManager,
        V87AlphaFusion,
        V87RiskParity,
        V87LiquidityConstraintDetector,
        V87TurnoverTracker,
        V87DrawdownTracker,
        V87ICSmoother,
        V87_INITIAL_CAPITAL,
        V87_MAX_POSITIONS,
        V87_WARMUP_PERIOD,
        V87_TURNOVER_TARGET,
        V87_DRAWDOWN_TARGET,
        V87_IC_SMOOTHING_TARGET,
        V87_COMMISSION_RATE,
        V87_MIN_COMMISSION,
        V87_STAMP_DUTY,
        V87_TRANSFER_FEE,
        V87_RANK_IC_OOS_YEARS,
        V87_FUSION_LAGS,
        V87_VOLATILITY_WINDOW,
        V87_LIMIT_DOWN_THRESHOLD,
        EPSILON,
    )
except ImportError:
    from core.v87_core import (
        V87DataManager,
        V87AlphaFusion,
        V87RiskParity,
        V87LiquidityConstraintDetector,
        V87TurnoverTracker,
        V87DrawdownTracker,
        V87ICSmoother,
        V87_INITIAL_CAPITAL,
        V87_MAX_POSITIONS,
        V87_WARMUP_PERIOD,
        V87_TURNOVER_TARGET,
        V87_DRAWDOWN_TARGET,
        V87_IC_SMOOTHING_TARGET,
        V87_COMMISSION_RATE,
        V87_MIN_COMMISSION,
        V87_STAMP_DUTY,
        V87_TRANSFER_FEE,
        V87_RANK_IC_OOS_YEARS,
        V87_FUSION_LAGS,
        V87_VOLATILITY_WINDOW,
        V87_LIMIT_DOWN_THRESHOLD,
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

# 导入 V86 用于对比
try:
    from src.core.v86_core import V86RankICCalculator
except ImportError:
    try:
        from core.v86_core import V86RankICCalculator
    except ImportError:
        V86RankICCalculator = None


# ===========================================
# V87 引擎配置
# ===========================================

@dataclass
class V87EngineConfig:
    """V87 引擎配置"""
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = V87_INITIAL_CAPITAL
    max_positions: int = V87_MAX_POSITIONS
    warmup_period: int = V87_WARMUP_PERIOD
    commission_rate: float = V87_COMMISSION_RATE
    min_commission: float = V87_MIN_COMMISSION
    stamp_duty: float = V87_STAMP_DUTY
    transfer_fee: float = V87_TRANSFER_FEE
    oos_years: List[str] = None
    fusion_lags: List[int] = None
    fusion_half_life: int = 3
    volatility_window: int = V87_VOLATILITY_WINDOW
    risk_parity_exponent: float = 1.0
    limit_down_threshold: float = V87_LIMIT_DOWN_THRESHOLD
    
    def __post_init__(self):
        if self.oos_years is None:
            self.oos_years = V87_RANK_IC_OOS_YEARS
        if self.fusion_lags is None:
            self.fusion_lags = V87_FUSION_LAGS


# ===========================================
# V87 引擎
# ===========================================

class V87Engine:
    """V87 回测引擎 - 多时空尺度融合与组合权重熵优化"""
    
    def __init__(self, config: V87EngineConfig = None, db=None):
        self.config = config or V87EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V87: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        self.data_manager = V87DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
        })
        self.alpha_fusion = V87AlphaFusion(config={
            'fusion_lags': self.config.fusion_lags,
            'fusion_half_life': self.config.fusion_half_life,
        })
        self.risk_parity = V87RiskParity(config={
            'volatility_window': self.config.volatility_window,
            'risk_parity_exponent': self.config.risk_parity_exponent,
        })
        self.liquidity_constraint = V87LiquidityConstraintDetector(config={
            'limit_down_threshold': self.config.limit_down_threshold,
        })
        self.turnover_tracker = V87TurnoverTracker(config={
            'turnover_target': V87_TURNOVER_TARGET,
        })
        self.drawdown_tracker = V87DrawdownTracker(config={
            'drawdown_target': V87_DRAWDOWN_TARGET,
        })
        self.ic_smoother = V87ICSmoother(config={
            'smoothing_target': V87_IC_SMOOTHING_TARGET,
        })
        
        if V86RankICCalculator:
            self.v86_ic_calculator = V86RankICCalculator(db=self.db)
        else:
            self.v86_ic_calculator = None
        
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.v86_ic_volatility = None
        
        logger.info("V87 Engine 初始化完成")
        logger.info(f"V87: 初始资金={self.config.initial_capital:,.2f} (严禁修改)")
        logger.info(f"V87: 多时空融合 Lags={self.config.fusion_lags}")
        logger.info(f"V87: 半衰期={self.config.fusion_half_life}")
        logger.info(f"V87: 换手率目标 <= {V87_TURNOVER_TARGET:.1%}")
        logger.info(f"V87: 回撤目标 <= {V87_DRAWDOWN_TARGET:.1%}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 60)
        logger.info("V87 回测引擎启动")
        logger.info("=" * 60)
        
        if self.db is None:
            logger.error("V87: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            logger.info("V87: 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            logger.info("V87: 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V87: 未加载到任何数据")
                return self._empty_result()
            
            for year in self.config.oos_years:
                df_year = df.filter(pl.col('trade_date').str.starts_with(year))
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V87: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            logger.info("V87: 开始计算因子信号...")
            df_with_signals = self._compute_signals(df)
            
            logger.info("V87: 开始多时空尺度融合...")
            df_with_fusion = self.alpha_fusion.compute_fusion_signal(
                df_with_signals, signal_col='composite_score'
            )
            
            logger.info("V87: 开始风险平价权重计算...")
            df_with_weights = self.risk_parity.compute_risk_parity_weights(
                df_with_fusion, score_col='fused_signal'
            )
            
            logger.info("V87: 开始流动性受限检测...")
            df_with_liquidity = self.liquidity_constraint.detect_liquidity_constraints(
                df_with_weights
            )
            
            logger.info("V87: 开始 IC 计算与平滑度分析...")
            ic_summary = self._compute_ic_analysis(df_with_liquidity)
            
            fusion_summary = self.alpha_fusion.get_fusion_summary()
            risk_parity_summary = self.risk_parity.get_risk_parity_summary()
            liquidity_summary = self.liquidity_constraint.get_liquidity_summary()
            turnover_summary = self.turnover_tracker.get_turnover_summary()
            drawdown_summary = self.drawdown_tracker.get_drawdown_summary()
            
            hard_metrics = self._verify_hard_metrics(
                ic_summary, turnover_summary, drawdown_summary
            )
            
            audit_report = self._generate_audit_report(
                data_integrity_results,
                fusion_summary,
                risk_parity_summary,
                liquidity_summary,
                turnover_summary,
                drawdown_summary,
                ic_summary,
                hard_metrics
            )
            
            result = {
                'fusion_summary': fusion_summary,
                'risk_parity_summary': risk_parity_summary,
                'liquidity_summary': liquidity_summary,
                'turnover_summary': turnover_summary,
                'drawdown_summary': drawdown_summary,
                'ic_summary': ic_summary,
                'hard_metrics': hard_metrics,
                'data_integrity': data_integrity_results,
                'audit_report': audit_report,
            }
            
            logger.info("=" * 60)
            logger.info("V87 回测完成")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"V87 回测失败 - {e}")
            logger.error(traceback.format_exc())
            return self._empty_result()
    
    def _check_data_integrity(self) -> Dict[str, Dict[str, Any]]:
        """检查数据完整性"""
        results = {}
        for year in self.config.oos_years:
            passed, message = self.data_manager.check_data_integrity(year)
            trading_days = self.data_manager.get_trading_days_count(year)
            results[year] = {
                'passed': passed,
                'message': message,
                'trading_days': trading_days,
                'min_required': 500000,
            }
            if passed:
                logger.info(f"V87: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V87: {year}年数据检查失败 - {message}")
        return results
    
    def _load_data(self) -> pl.DataFrame:
        """加载数据"""
        all_dfs = []
        for year in self.config.oos_years:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            try:
                df = self.data_manager.load_stock_data(start_date, end_date)
                if not df.is_empty():
                    all_dfs.append(df)
                    logger.info(f"V87: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V87: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V87: 总数据行数={combined_df.height:,}")
        return combined_df
    
    def _compute_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算因子信号"""
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
        """计算 Refined_Residual 因子"""
        result = df.clone()
        window = 5
        
        result = result.with_columns([
            ((pl.col('close').shift(1) - pl.col('close').shift(window + 1)) / 
             (pl.col('close').shift(window + 1) + EPSILON)).alias('stock_return_5d')
        ])
        
        industry_median = result.group_by(['industry_code', 'trade_date']).agg([
            pl.col('pct_chg').median().alias('industry_return_5d')
        ])
        
        result = result.join(
            industry_median.select(['industry_code', 'trade_date', 'industry_return_5d']),
            on=['industry_code', 'trade_date'],
            how='left'
        )
        
        result = result.with_columns([
            (pl.col('stock_return_5d') - pl.col('industry_return_5d')).alias('residual_return')
        ])
        
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
        
        result = result.with_columns([
            pl.col('n_stocks').cast(pl.Float64).alias('n_stocks_float'),
            pl.col('n_stocks_flow').cast(pl.Float64).alias('n_stocks_flow_float'),
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
             (pl.col('n_stocks_interaction').cast(pl.Float64) + EPSILON))).alias('vol_price_interaction_score')
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
        
        residual_weight = 0.20
        flow_weight = 0.10
        interaction_weight = 0.70
        
        result = result.with_columns([
            (residual_weight * pl.col('refined_residual_score') + 
             flow_weight * pl.col('smart_flow_score') +
             interaction_weight * pl.col('vol_price_interaction_score')).alias('composite_score')
        ])
        
        return result
    
    def _compute_ic_analysis(self, df: pl.DataFrame) -> Dict[str, Any]:
        """计算 IC 并分析平滑度"""
        ic_summary = {
            'mean_ic_t1': 0.0, 'mean_ic_t2': 0.0, 'mean_ic_t3': 0.0,
            'std_ic_t1': 0.0, 'std_ic_t2': 0.0, 'std_ic_t3': 0.0,
            'ic_volatility': 0.0, 'smoothing_improvement': 0.0,
            'v86_ic_volatility': 0.0, 'v87_ic_volatility': 0.0,
        }
        
        try:
            if self.v86_ic_calculator:
                # 使用 composite_score 计算 V86 的 IC（原始信号）
                ic_decay_results_v86 = self.v86_ic_calculator.calculate_ic_decay_series(
                    df, signal_col='composite_score'
                )
                
                # 使用 fused_signal 计算 V87 的 IC（融合信号）
                ic_decay_results_v87 = self.v86_ic_calculator.calculate_ic_decay_series(
                    df, signal_col='fused_signal'
                )
                
                if ic_decay_results_v86 and ic_decay_results_v87:
                    # V86 IC 统计
                    ic_t1_list_v86 = [m.ic_t1 for m in ic_decay_results_v86]
                    ic_t2_list_v86 = [m.ic_t2 for m in ic_decay_results_v86]
                    ic_t3_list_v86 = [m.ic_t3 for m in ic_decay_results_v86]
                    
                    # V87 IC 统计
                    ic_t1_list_v87 = [m.ic_t1 for m in ic_decay_results_v87]
                    ic_t2_list_v87 = [m.ic_t2 for m in ic_decay_results_v87]
                    ic_t3_list_v87 = [m.ic_t3 for m in ic_decay_results_v87]
                    
                    # V86 IC 波动率（T+1 到 T+3 的标准差）
                    v86_ic_mean = [
                        float(np.mean(ic_t1_list_v86)),
                        float(np.mean(ic_t2_list_v86)),
                        float(np.mean(ic_t3_list_v86))
                    ]
                    v86_ic_volatility = float(np.std(v86_ic_mean, ddof=1))
                    
                    # V87 IC 波动率
                    v87_ic_mean = [
                        float(np.mean(ic_t1_list_v87)),
                        float(np.mean(ic_t2_list_v87)),
                        float(np.mean(ic_t3_list_v87))
                    ]
                    v87_ic_volatility = float(np.std(v87_ic_mean, ddof=1))
                    
                    # 计算平滑度改善
                    if v86_ic_volatility > EPSILON:
                        smoothing_improvement = (v86_ic_volatility - v87_ic_volatility) / v86_ic_volatility
                    else:
                        smoothing_improvement = 0.0
                    
                    ic_summary['mean_ic_t1'] = v87_ic_mean[0]
                    ic_summary['mean_ic_t2'] = v87_ic_mean[1]
                    ic_summary['mean_ic_t3'] = v87_ic_mean[2]
                    ic_summary['std_ic_t1'] = float(np.std(ic_t1_list_v87, ddof=1))
                    ic_summary['std_ic_t2'] = float(np.std(ic_t2_list_v87, ddof=1))
                    ic_summary['std_ic_t3'] = float(np.std(ic_t3_list_v87, ddof=1))
                    ic_summary['ic_volatility'] = v87_ic_volatility
                    ic_summary['smoothing_improvement'] = smoothing_improvement
                    ic_summary['v86_ic_volatility'] = v86_ic_volatility
                    ic_summary['v87_ic_volatility'] = v87_ic_volatility
                    
                    logger.info(f"V87: IC 计算完成 - T+1={ic_summary['mean_ic_t1']:.4f}, T+2={ic_summary['mean_ic_t2']:.4f}, T+3={ic_summary['mean_ic_t3']:.4f}")
                    logger.info(f"V87: IC 波动率 - V86={v86_ic_volatility:.4f}, V87={v87_ic_volatility:.4f}")
                    logger.info(f"V87: 平滑度改善={smoothing_improvement:.2%}")
        except Exception as e:
            logger.error(f"V87: IC 计算失败 - {e}")
        
        return ic_summary
    
    def _verify_hard_metrics(self, ic_summary: Dict, turnover_summary: Dict,
                             drawdown_summary: Dict) -> Dict[str, Any]:
        """验证硬性指标"""
        metric_a_pass = turnover_summary.get('mean_turnover', 1.0) <= V87_TURNOVER_TARGET
        metric_b_pass = drawdown_summary.get('max_drawdown', 1.0) <= V87_DRAWDOWN_TARGET
        smoothing_improvement = ic_summary.get('smoothing_improvement', 0.0)
        metric_c_pass = smoothing_improvement >= V87_IC_SMOOTHING_TARGET
        
        return {
            'metric_a_pass': metric_a_pass,
            'metric_a_details': {'mean_turnover': turnover_summary.get('mean_turnover', 0.0), 'target': V87_TURNOVER_TARGET},
            'metric_b_pass': metric_b_pass,
            'metric_b_details': {'max_drawdown': drawdown_summary.get('max_drawdown', 0.0), 'target': V87_DRAWDOWN_TARGET},
            'metric_c_pass': metric_c_pass,
            'metric_c_details': {'smoothing_improvement': smoothing_improvement, 'target': V87_IC_SMOOTHING_TARGET},
        }
    
    def _generate_audit_report(self, data_integrity: Dict, fusion_summary: Dict,
                                risk_parity_summary: Dict, liquidity_summary: Dict,
                                turnover_summary: Dict, drawdown_summary: Dict,
                                ic_summary: Dict, hard_metrics: Dict) -> str:
        """生成《V87 多尺度融合与风险对冲审计报告》"""
        lines = []
        lines.append("=" * 70)
        lines.append("《V87 多尺度融合与风险对冲审计报告》")
        lines.append("=" * 70)
        lines.append("")
        lines.append("1. 数据完整性审计")
        lines.append("   " + "-" * 50)
        
        for year, result in data_integrity.items():
            status = "✓" if result['passed'] else "✗"
            lines.append(f"   {year}年：{status} {result['message']}")
        
        lines.append("")
        lines.append("2. 多时空尺度融合")
        lines.append("   " + "-" * 50)
        lines.append(f"   融合 Lags: {self.config.fusion_lags}")
        lines.append(f"   半衰期：{self.config.fusion_half_life}")
        lines.append(f"   融合后信号均值：{fusion_summary.get('mean_fused_signal', 0.0):.4f}")
        lines.append(f"   融合后信号标准差：{fusion_summary.get('std_fused_signal', 0.0):.4f}")
        lines.append(f"   T-1 信号均值：{fusion_summary.get('mean_signal_t1', 0.0):.4f}")
        lines.append(f"   T-3 信号均值：{fusion_summary.get('mean_signal_t3', 0.0):.4f}")
        lines.append(f"   T-5 信号均值：{fusion_summary.get('mean_signal_t5', 0.0):.4f}")
        
        lines.append("")
        lines.append("3. 风险平价权重")
        lines.append("   " + "-" * 50)
        lines.append(f"   平均权重：{risk_parity_summary.get('mean_weight', 0.0):.4f}")
        lines.append(f"   权重标准差：{risk_parity_summary.get('std_weight', 0.0):.4f}")
        lines.append(f"   最大权重：{risk_parity_summary.get('max_weight', 0.0):.4f}")
        lines.append(f"   最小权重：{risk_parity_summary.get('min_weight', 0.0):.4f}")
        lines.append(f"   平均波动率：{risk_parity_summary.get('mean_volatility', 0.0):.4f}")
        lines.append(f"   集中度指数：{risk_parity_summary.get('weight_concentration', 0.0):.4f}")
        
        lines.append("")
        lines.append("4. 流动性受限检测")
        lines.append("   " + "-" * 50)
        lines.append(f"   总受限次数：{liquidity_summary.get('total_constraints', 0)}")
        lines.append(f"   跌停次数：{liquidity_summary.get('limit_down_count', 0)}")
        lines.append(f"   涨停次数：{liquidity_summary.get('limit_up_count', 0)}")
        lines.append(f"   平均涨跌幅：{liquidity_summary.get('avg_pct_chg', 0.0):.2f}%")
        
        lines.append("")
        lines.append("5. 换手率控制")
        lines.append("   " + "-" * 50)
        lines.append(f"   平均换手率：{turnover_summary.get('mean_turnover', 0.0):.2%}")
        lines.append(f"   最大换手率：{turnover_summary.get('max_turnover', 0.0):.2%}")
        lines.append(f"   超标天数：{turnover_summary.get('over_limit_days', 0)}")
        lines.append(f"   超标比例：{turnover_summary.get('over_limit_ratio', 0.0):.2%}")
        
        lines.append("")
        lines.append("6. 回撤控制")
        lines.append("   " + "-" * 50)
        lines.append(f"   最大回撤：{drawdown_summary.get('max_drawdown', 0.0):.2%}")
        lines.append(f"   平均回撤：{drawdown_summary.get('mean_drawdown', 0.0):.2%}")
        lines.append(f"   超标天数：{drawdown_summary.get('over_limit_days', 0)}")
        
        lines.append("")
        lines.append("7. IC 分析与平滑度")
        lines.append("   " + "-" * 50)
        lines.append(f"   T+1 IC: {ic_summary.get('mean_ic_t1', 0.0):.4f} (Std: {ic_summary.get('std_ic_t1', 0.0):.4f})")
        lines.append(f"   T+2 IC: {ic_summary.get('mean_ic_t2', 0.0):.4f} (Std: {ic_summary.get('std_ic_t2', 0.0):.4f})")
        lines.append(f"   T+3 IC: {ic_summary.get('mean_ic_t3', 0.0):.4f} (Std: {ic_summary.get('std_ic_t3', 0.0):.4f})")
        lines.append(f"   IC 波动率：{ic_summary.get('ic_volatility', 0.0):.4f}")
        lines.append(f"   平滑度改善：{ic_summary.get('smoothing_improvement', 0.0):.2%}")
        
        lines.append("")
        lines.append("8. 硬性指标验证")
        lines.append("   " + "-" * 50)
        lines.append(f"   指标 A (换手率 <= 15%): {'✓' if hard_metrics['metric_a_pass'] else '✗'}")
        lines.append(f"     - 平均换手率：{hard_metrics['metric_a_details']['mean_turnover']:.2%}")
        lines.append("")
        lines.append(f"   指标 B (回撤 <= 8%): {'✓' if hard_metrics['metric_b_pass'] else '✗'}")
        lines.append(f"     - 最大回撤：{hard_metrics['metric_b_details']['max_drawdown']:.2%}")
        lines.append("")
        lines.append(f"   指标 C (IC 波动率下降 >= 15%): {'✓' if hard_metrics['metric_c_pass'] else '✗'}")
        lines.append(f"     - 平滑度改善：{hard_metrics['metric_c_details']['smoothing_improvement']:.2%}")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'fusion_summary': {}, 'risk_parity_summary': {},
            'liquidity_summary': {}, 'turnover_summary': {},
            'drawdown_summary': {}, 'ic_summary': {},
            'hard_metrics': {}, 'data_integrity': {},
            'audit_report': '',
        }


# ===========================================
# 主程序
# ===========================================

def run_v87_backtest(config: V87EngineConfig = None) -> Dict[str, Any]:
    """运行 V87 回测"""
    engine = V87Engine(config=config)
    return engine.run_backtest()


def print_v87_report(result: Dict[str, Any]):
    """打印 V87 报告"""
    logger.info("=" * 60)
    logger.info("V87 最终报告")
    logger.info("=" * 60)
    
    fusion_summary = result.get('fusion_summary', {})
    logger.info("【多时空尺度融合】")
    logger.info(f"  融合后信号均值：{fusion_summary.get('mean_fused_signal', 0.0):.4f}")
    logger.info(f"  T-1 信号均值：{fusion_summary.get('mean_signal_t1', 0.0):.4f}")
    logger.info(f"  T-3 信号均值：{fusion_summary.get('mean_signal_t3', 0.0):.4f}")
    logger.info(f"  T-5 信号均值：{fusion_summary.get('mean_signal_t5', 0.0):.4f}")
    
    risk_parity_summary = result.get('risk_parity_summary', {})
    logger.info("")
    logger.info("【风险平价权重】")
    logger.info(f"  平均权重：{risk_parity_summary.get('mean_weight', 0.0):.4f}")
    logger.info(f"  最大权重：{risk_parity_summary.get('max_weight', 0.0):.4f}")
    
    ic_summary = result.get('ic_summary', {})
    logger.info("")
    logger.info("【IC 分析】")
    logger.info(f"  T+1 IC: {ic_summary.get('mean_ic_t1', 0.0):.4f}")
    logger.info(f"  T+2 IC: {ic_summary.get('mean_ic_t2', 0.0):.4f}")
    logger.info(f"  T+3 IC: {ic_summary.get('mean_ic_t3', 0.0):.4f}")
    
    hard_metrics = result.get('hard_metrics', {})
    logger.info("")
    logger.info("【硬性指标验证】")
    logger.info(f"  指标 A (换手率): {'✓' if hard_metrics.get('metric_a_pass') else '✗'}")
    logger.info(f"  指标 B (回撤): {'✓' if hard_metrics.get('metric_b_pass') else '✗'}")
    logger.info(f"  指标 C (平滑度): {'✓' if hard_metrics.get('metric_c_pass') else '✗'}")
    
    audit_report = result.get('audit_report', '')
    if audit_report:
        logger.info("")
        logger.info(audit_report)
    
    logger.info("=" * 60)


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    config = V87EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
    )
    
    result = run_v87_backtest(config)
    print_v87_report(result)
    
    output_path = "reports/v87_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_result = {
        'fusion_summary': result.get('fusion_summary', {}),
        'risk_parity_summary': result.get('risk_parity_summary', {}),
        'liquidity_summary': result.get('liquidity_summary', {}),
        'turnover_summary': result.get('turnover_summary', {}),
        'drawdown_summary': result.get('drawdown_summary', {}),
        'ic_summary': result.get('ic_summary', {}),
        'hard_metrics': result.get('hard_metrics', {}),
        'data_integrity': result.get('data_integrity', {}),
        'audit_report': result.get('audit_report', ''),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V87: 结果已保存至 {output_path}")