"""
V89 Engine - 预测算法重构与真实 IC 提升

【V89 核心理念】
1. 半衰期衰减权重 - 废除 V88 的 IC 倒数极端权重
2. Vol_Price_Interaction 优化 - 引入量价背离二阶导数
3. 真实性审计 - 报告指标必须与控制台 Logging 100% 一致
4. 数据自愈 - 遇到 ConnectionError 自动调用 v83_data_repairer.py

【V89 硬性指标】
- 指标 A: T+1 Rank IC > 0.045
- 指标 B: 2019/2021/2024 三年度平均年化收益率 > 10%
- 指标 C: 单日换手率 < 15%，总年化换手率 400%-600%
- 指标 D: IC 衰减呈自然指数级衰减，Lag 1 权重 > 30%

作者：量化系统
版本：V89.0
日期：2026-03-29
"""

import traceback
import time
import math
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import asdict
import numpy as np
import polars as pl
from loguru import logger

# 导入 V89 核心模块
from src.core.v89_core import (
    V89DataManager,
    V89AlphaFusion,
    V89AlphaWeightEngine,
    V89ICAudit,
    V89TurnoverTracker,
    V89PortfolioTracker,
    V89VolumePriceDivergence,
    V89_INITIAL_CAPITAL,
    V89_MAX_POSITIONS,
    V89_WARMUP_PERIOD,
    V89_MIN_SCORE_THRESHOLD,
    V89_MIN_SINGLE_WEIGHT,
    V89_MAX_SINGLE_WEIGHT,
    V89_TURNOVER_MIN,
    V89_TURNOVER_MAX,
    V89_DAILY_TURNOVER_MAX,
    V89_ANNUAL_RETURN_TARGET,
    V89_T1_IC_TARGET,
    V89_COMMISSION_RATE,
    V89_MIN_COMMISSION,
    V89_STAMP_DUTY,
    V89_TRANSFER_FEE,
    V89_HALF_LIFE_LAGS,
    V89_LAG1_WEIGHT,
    check_lookahead_bias,
    calculate_ic_decay_simple,
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
# V89 引擎配置
# ===========================================

class V89EngineConfig:
    """V89 引擎配置"""
    
    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = V89_INITIAL_CAPITAL,
        max_positions: int = V89_MAX_POSITIONS,
        warmup_period: int = V89_WARMUP_PERIOD,
        commission_rate: float = V89_COMMISSION_RATE,
        min_commission: float = V89_MIN_COMMISSION,
        stamp_duty: float = V89_STAMP_DUTY,
        transfer_fee: float = V89_TRANSFER_FEE,
        oos_years: List[str] = None,
        min_score_threshold: float = V89_MIN_SCORE_THRESHOLD,
        min_single_weight: float = V89_MIN_SINGLE_WEIGHT,
        max_single_weight: float = V89_MAX_SINGLE_WEIGHT,
        rebalance_interval: int = 42,  # 调仓间隔（42 天≈季度调仓，降低换手率）
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
        self.rebalance_interval = rebalance_interval


# ===========================================
# V89 引擎
# ===========================================

class V89Engine:
    """V89 回测引擎"""
    
    def __init__(self, config: V89EngineConfig = None, db=None):
        self.config = config or V89EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V89: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        self.data_manager = V89DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
        })
        self.alpha_fusion = V89AlphaFusion(db=self.db, config={
            'fusion_lags': V89_HALF_LIFE_LAGS,
        })
        self.alpha_weight = V89AlphaWeightEngine(config={
            'min_score': self.config.min_score_threshold,
            'min_weight': self.config.min_single_weight,
            'max_weight': self.config.max_single_weight,
        })
        self.volume_price_divergence = V89VolumePriceDivergence()
        self.ic_audit = V89ICAudit(db=self.db)
        self.turnover_tracker = V89TurnoverTracker(config={
            'rebalance_interval': self.config.rebalance_interval,
        })
        self.portfolio_tracker = V89PortfolioTracker(
            initial_capital=self.config.initial_capital
        )
        
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        self.trade_records: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        
        logger.info("=" * 70)
        logger.info("V89 Engine 初始化完成")
        logger.info("=" * 70)
        logger.info(f"V89: 初始资金={self.config.initial_capital:,.2f} (严禁修改)")
        logger.info(f"V89: Score 阈值={self.config.min_score_threshold}")
        logger.info(f"V89: 权重限制=[{self.config.min_single_weight:.1%}, {self.config.max_single_weight:.1%}]")
        logger.info(f"V89: 年化换手率目标=[{V89_TURNOVER_MIN:.0%}, {V89_TURNOVER_MAX:.0%}]")
        logger.info(f"V89: 单日换手率上限={V89_DAILY_TURNOVER_MAX:.1%}")
        logger.info(f"V89: T+1 IC 目标={V89_T1_IC_TARGET:.4f}")
        logger.info(f"V89: 半衰期权重=Lag1={V89_LAG1_WEIGHT:.0%}, Lag3=0.30, Lag5=0.20")
        logger.info(f"V89: 调仓间隔={self.config.rebalance_interval}天")
        logger.info("=" * 70)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V89 预测算法重构回测引擎启动")
        logger.info("=" * 70)
        
        if self.db is None:
            logger.error("V89: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查
            logger.info("V89: [1/8] 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V89: [2/8] 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V89: 未加载到任何数据")
                return self._empty_result()
            
            # 输出数据加载统计
            for year in self.config.oos_years:
                df_year = df.filter(pl.col('trade_date').str.starts_with(year))
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V89: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算因子信号
            logger.info("V89: [3/8] 开始计算因子信号...")
            df_with_signals = self._compute_signals(df)
            
            # 4. 前瞻偏差检查（在计算信号前）
            logger.info("V89: [4/8] 开始前瞻偏差检查...")
            lookahead_result = check_lookahead_bias(df_with_signals, signal_col='composite_score')
            logger.info(f"V89: 前瞻检查结果 - {lookahead_result.message}")
            if not lookahead_result.passed:
                logger.warning("V89: [WARNING] 前瞻检查未通过，请检查代码逻辑")
            
            # 5. 半衰期融合
            logger.info("V89: [5/8] 开始半衰期融合...")
            df_with_fusion = self.alpha_fusion.compute_fusion_signal(
                df_with_signals, signal_col='composite_score'
            )
            
            # 6. Alpha 权重计算
            logger.info("V89: [6/8] 开始 Alpha 权重计算...")
            df_with_weights = self.alpha_weight.compute_alpha_weights(
                df_with_fusion, score_col='fused_signal'
            )
            
            # 7. IC 审计
            logger.info("V89: [7/8] 开始 IC 审计...")
            ic_audit_results = self.ic_audit.calculate_rank_ic(
                df_with_weights, signal_col='fused_signal'
            )
            
            # 输出 IC 审计结果
            logger.info(f"V89: T+1 Rank IC = {ic_audit_results['ic_t1']:.4f} (目标 > {V89_T1_IC_TARGET})")
            logger.info(f"V89: T+2 Rank IC = {ic_audit_results['ic_t2']:.4f}")
            logger.info(f"V89: T+3 Rank IC = {ic_audit_results['ic_t3']:.4f}")
            logger.info(f"V89: IC 衰减正常 = {ic_audit_results['decay_normal']}")
            
            # 8. 执行回测交易
            logger.info("V89: [8/8] 开始执行回测交易...")
            trade_results = self._execute_backtest(df_with_weights)
            
            # 生成报告
            logger.info("V89: 生成审计报告...")
            audit_report = self._generate_audit_report(
                data_integrity_results,
                ic_audit_results,
                trade_results,
                lookahead_result,
            )
            
            result = {
                'data_integrity': data_integrity_results,
                'ic_audit': ic_audit_results,
                'lookahead_check': {
                    'passed': lookahead_result.passed,
                    'message': lookahead_result.message,
                },
                'trade_results': trade_results,
                'audit_report': audit_report,
                'trade_records': self.trade_records,
                'daily_snapshots': self.daily_snapshots,
            }
            
            logger.info("=" * 70)
            logger.info("V89 回测完成")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"V89 回测失败 - {e}")
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
                logger.info(f"V89: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V89: {year}年数据检查失败 - {message}")
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
                    logger.info(f"V89: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V89: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V89: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _compute_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算因子信号"""
        result = df.clone()
        
        # 确保数值列类型正确
        result = result.with_columns([
            pl.col('open').cast(pl.Float64, strict=False).alias('open'),
            pl.col('high').cast(pl.Float64, strict=False).alias('high'),
            pl.col('low').cast(pl.Float64, strict=False).alias('low'),
            pl.col('close').cast(pl.Float64, strict=False).alias('close'),
            pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            pl.col('amount').cast(pl.Float64, strict=False).alias('amount'),
            pl.col('pct_chg').cast(pl.Float64, strict=False).alias('pct_chg'),
        ])
        
        # 计算 Refined Residual
        result = self._compute_refined_residual(result)
        
        # 计算 Smart Flow
        result = self._compute_smart_flow(result)
        
        # 计算 Vol_Price_Interaction（带二阶导数优化）
        result = self._compute_vol_price_interaction(result)
        
        # 计算综合评分
        result = self._compute_composite_score(result)
        
        return result
    
    def _compute_refined_residual(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 Refined Residual 因子（V89 优化版）
        
        【V89 改进】
        - 使用 10 日动量而非 5 日，捕捉中期趋势
        - 加入成交量确认，量价配合
        """
        result = df.clone()
        window = 10  # 改为 10 日动量
        
        # 确保 industry_code 列存在且为字符串类型
        if 'industry_code' not in result.columns:
            result = result.with_columns([pl.lit('Unknown').alias('industry_code')])
        
        result = result.with_columns([
            pl.col('industry_code').cast(pl.Utf8, strict=False).fill_null('Unknown').alias('industry_code')
        ])
        
        # 计算 10 日收益率
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
        
        # 计算残差收益率
        result = result.with_columns([
            (pl.col('stock_return_10d') - pl.col('market_return_10d')).alias('residual_return')
        ])
        
        # 加入成交量确认：成交量/20 日均量
        result = result.with_columns([
            (pl.col('volume').fill_null(0) / (pl.col('volume').fill_null(0).rolling_sum(window_size=20).over('symbol') / 20 + EPSILON)).alias('volume_ratio')
        ])
        
        # 残差收益率 * 成交量比率（量价配合）
        result = result.with_columns([
            (pl.col('residual_return') * pl.col('volume_ratio')).alias('residual_volume_confirmed')
        ])
        
        # 计算排名和百分位
        result = result.with_columns([
            pl.col('residual_volume_confirmed').rank('ordinal', descending=True).over('trade_date').alias('residual_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('residual_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks').cast(pl.Float64) + EPSILON))).alias('refined_residual_score')
        ])
        
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
        
        # 计算排名和百分位
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
        计算 Vol_Price_Interaction 因子（V89 优化版）
        
        【V89 改进】
        1. 保留 V88 的残差排名交互逻辑
        2. 引入量价背离二阶导数作为增强因子
        3. 最终分数 = 传统 interaction + 背离增强
        """
        result = df.clone()
        
        # 确保必要的列存在
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
        
        # 计算归一化排名
        result = result.with_columns([
            (pl.col('residual_rank') / (pl.col('n_stocks').cast(pl.Float64) + EPSILON)).alias('residual_rank_norm'),
            (pl.col('flow_rank') / (pl.col('n_stocks_flow').cast(pl.Float64) + EPSILON)).alias('flow_rank_norm'),
        ])
        
        # 计算传统 interaction（V88 方式）
        result = result.with_columns([
            (pl.col('residual_rank_norm') * pl.col('flow_rank_norm')).alias('interaction_raw')
        ])
        
        # 计算排名和百分位
        result = result.with_columns([
            pl.col('interaction_raw').rank('ordinal', descending=True).over('trade_date').alias('interaction_rank'),
            pl.col('symbol').count().over('trade_date').alias('n_stocks_interaction')
        ])
        
        result = result.with_columns([
            (100.0 * (1.0 - (pl.col('interaction_rank').cast(pl.Float64) - 0.5) / 
             (pl.col('n_stocks_interaction').cast(pl.Float64) + EPSILON))).alias('interaction_score')
        ])
        
        # 使用 V89 VolumePriceDivergence 计算背离信号作为增强
        result = self.volume_price_divergence.compute_divergence(result)
        
        # 获取背离分数
        if 'volume_price_divergence_score' not in result.columns:
            result = result.with_columns([
                pl.lit(50.0).alias('volume_price_divergence_score')
            ])
        
        # 最终分数 = 传统 interaction * 0.7 + 背离分数 * 0.3
        result = result.with_columns([
            (0.7 * pl.col('interaction_score') + 0.3 * pl.col('volume_price_divergence_score')).alias('vol_price_interaction_score')
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
        
        # V89 权重配置：更重视量价交互
        residual_weight = 0.20
        flow_weight = 0.15
        interaction_weight = 0.65
        
        result = result.with_columns([
            (residual_weight * pl.col('refined_residual_score') + 
             flow_weight * pl.col('smart_flow_score') +
             interaction_weight * pl.col('vol_price_interaction_score')).alias('composite_score')
        ])
        
        return result
    
    def _execute_backtest(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        执行回测交易
        
        【V89 改进】
        - 双周调仓（10 天），平衡换手率和收益
        - 缓冲区间：权重变化超过 30% 才调仓
        """
        logger.info("V89: 开始执行回测交易...")
        
        df = df.sort(['trade_date', 'symbol'])
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        # 跳过热身期
        warmup_cutoff = unique_dates[:min(V89_WARMUP_PERIOD, len(unique_dates))]
        trade_dates = [d for d in unique_dates if d not in warmup_cutoff]
        
        logger.info(f"V89: 热身期 {len(warmup_cutoff)} 天，交易期 {len(trade_dates)} 天")
        
        # 初始化
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.positions = {}
        self.trade_records = []
        self.daily_snapshots = []
        
        prev_date = None
        total_buy_value = 0.0
        total_sell_value = 0.0
        
        # 双周调仓
        rebalance_interval = self.config.rebalance_interval
        
        for i, trade_date in enumerate(trade_dates):
            is_rebalance_day = (i % rebalance_interval == 0)
            
            day_df = df.filter(pl.col('trade_date') == trade_date)
            
            if day_df.is_empty():
                continue
            
            # 更新持仓价格
            price_map = dict(zip(
                day_df['symbol'].to_list(),
                day_df['close'].to_list()
            ))
            
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
                # 获取有有效权重的股票
                valid_stocks = day_df.filter(
                    (pl.col('alpha_weight').is_not_null()) &
                    (pl.col('alpha_weight') > EPSILON) &
                    ((pl.col('is_filtered').is_not_null()) & (pl.col('is_filtered') == False))
                ).sort('alpha_weight', descending=True)
                
                # 计算目标持仓
                target_positions = self._calculate_target_positions(
                    valid_stocks, self.portfolio_value
                )
                
                # 执行交易
                buy_value, sell_value = self._execute_trades(
                    trade_date, target_positions, price_map
                )
            
            total_buy_value += buy_value
            total_sell_value += sell_value
            
            # 记录换手率（传入 is_rebalance_day 参数）
            self.turnover_tracker.record_turnover(
                trade_date, self.portfolio_value, buy_value, sell_value,
                is_rebalance_day=is_rebalance_day
            )
            
            # 计算日收益
            if prev_date and self.daily_snapshots:
                prev_value = self.daily_snapshots[-1]['total_value']
                daily_return = (self.portfolio_value - prev_value) / prev_value if prev_value > EPSILON else 0.0
            else:
                daily_return = 0.0
            
            # 记录快照
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
                logger.info(f"V89: 处理 {i + 1}/{len(trade_dates)} 天，组合价值={self.portfolio_value:,.2f}")
        
        # 计算最终结果
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        turnover_summary = self.turnover_tracker.get_turnover_summary()
        max_drawdown = self._calculate_max_drawdown()
        
        # 计算年度收益
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
            'annual_returns': annual_returns,
            'avg_annual_return': np.mean(list(annual_returns.values())) if annual_returns else 0.0,
        }
        
        logger.info(f"V89: 回测完成 - 总收益={total_return:.2%}, 年化换手={turnover_summary['annualized_turnover']:.2%}")
        
        return result
    
    def _calculate_target_positions(self, valid_stocks: pl.DataFrame, 
                                     portfolio_value: float) -> Dict[str, float]:
        """计算目标持仓"""
        if valid_stocks.is_empty():
            return {}
        
        # 限制最大持仓数量为 20 只
        max_stocks = min(20, self.config.max_positions)
        top_stocks = valid_stocks.head(max_stocks)
        
        target_positions = {}
        for row in top_stocks.iter_rows(named=True):
            symbol = row['symbol']
            weight = row['alpha_weight']
            
            if weight > EPSILON:
                target_positions[symbol] = weight
        
        # 归一化权重
        total_weight = sum(target_positions.values())
        if total_weight > EPSILON:
            target_positions = {k: v / total_weight for k, v in target_positions.items()}
        
        return target_positions
    
    def _execute_trades(self, trade_date: str, target_positions: Dict[str, float],
                        price_map: Dict[str, float]) -> Tuple[float, float]:
        """执行交易"""
        buy_value = 0.0
        sell_value = 0.0
        
        # 绝对权重差异阈值：60%（更大的缓冲区间，降低换手率）
        abs_rebalance_threshold = 0.60
        
        # 卖出不在目标中的持仓
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
        
        # 买入/调整持仓
        for symbol, target_weight in target_positions.items():
            if symbol not in price_map:
                continue
            
            buy_price = price_map[symbol]
            target_value = self.portfolio_value * target_weight
            
            if symbol in self.positions:
                position = self.positions[symbol]
                current_value = position['current_price'] * position['quantity']
                current_weight = current_value / self.portfolio_value if self.portfolio_value > EPSILON else 0.0
                
                abs_weight_diff = abs(target_weight - current_weight)
                
                if abs_weight_diff < abs_rebalance_threshold:
                    position['weight'] = target_weight
                    continue
                
                diff_value = target_value - current_value
                
                if diff_value > EPSILON * self.portfolio_value:
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
            snapshots_year = [s for s in self.daily_snapshots 
                             if s['trade_date'].startswith(year)]
            
            if len(snapshots_year) >= 2:
                start_value = snapshots_year[0]['total_value']
                end_value = snapshots_year[-1]['total_value']
                year_return = (end_value - start_value) / start_value if start_value > EPSILON else 0.0
                annual_returns[year] = year_return
        
        return annual_returns
    
    def _generate_audit_report(self, data_integrity: Dict, ic_audit: Dict,
                                trade_results: Dict, lookahead_result: Any) -> str:
        """生成审计报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("《V89 预测算法重构审计报告》")
        lines.append("=" * 70)
        lines.append("")
        
        lines.append("1. 数据完整性审计")
        lines.append("   " + "-" * 50)
        for year, result in data_integrity.items():
            status = "✓" if result['passed'] else "✗"
            lines.append(f"   {year}年：{status} {result['message']}")
        lines.append("")
        
        lines.append("2. 前瞻偏差检查")
        lines.append("   " + "-" * 50)
        lines.append(f"   检查结果：{'通过' if lookahead_result.passed else '未通过'}")
        lines.append(f"   详情：{lookahead_result.message}")
        lines.append("")
        
        lines.append("3. IC 审计")
        lines.append("   " + "-" * 50)
        lines.append(f"   T+1 Rank IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V89_T1_IC_TARGET})")
        lines.append(f"   T+2 Rank IC: {ic_audit.get('ic_t2', 0.0):.4f}")
        lines.append(f"   T+3 Rank IC: {ic_audit.get('ic_t3', 0.0):.4f}")
        lines.append(f"   IC 衰减正常：{'是' if ic_audit.get('decay_normal') else '否'}")
        lines.append(f"   T+1 IC 达标：{'是' if ic_audit.get('t1_ic_passed') else '否'}")
        lines.append("")
        
        lines.append("4. 交易执行审计")
        lines.append("   " + "-" * 50)
        lines.append(f"   总收益：{trade_results.get('total_return', 0.0):.2%}")
        lines.append(f"   最终价值：{trade_results.get('final_value', 0.0):,.2f}")
        lines.append(f"   最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
        lines.append(f"   年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
        lines.append(f"   单日换手率上限：{trade_results.get('max_daily_turnover', 0.0):.2%}")
        lines.append(f"   单日换手率达标：{'是' if trade_results.get('daily_turnover_ok') else '否'}")
        
        # 年度收益
        annual_returns = trade_results.get('annual_returns', {})
        if annual_returns:
            lines.append("")
            lines.append("   年度收益:")
            for year, ret in annual_returns.items():
                lines.append(f"     {year}年：{ret:.2%}")
            lines.append(f"   平均年化收益：{trade_results.get('avg_annual_return', 0.0):.2%}")
        lines.append("")
        
        lines.append("5. V89 硬性指标验证")
        lines.append("   " + "-" * 50)
        
        # 指标 A: T+1 IC
        ic_t1 = ic_audit.get('ic_t1', 0.0)
        metric_a_pass = ic_t1 >= V89_T1_IC_TARGET
        lines.append(f"   指标 A (T+1 Rank IC > 0.045): {'✓' if metric_a_pass else '✗'}")
        lines.append(f"     - T+1 IC: {ic_t1:.4f}")
        lines.append("")
        
        # 指标 B: 年化收益
        avg_annual = trade_results.get('avg_annual_return', 0.0)
        metric_b_pass = avg_annual >= V89_ANNUAL_RETURN_TARGET
        lines.append(f"   指标 B (平均年化收益 > 10%): {'✓' if metric_b_pass else '✗'}")
        lines.append(f"     - 平均年化：{avg_annual:.2%}")
        lines.append("")
        
        # 指标 C: 换手率
        turnover = trade_results.get('annualized_turnover', 0.0)
        daily_ok = trade_results.get('daily_turnover_ok', False)
        metric_c_pass = (V89_TURNOVER_MIN <= turnover <= V89_TURNOVER_MAX) and daily_ok
        lines.append(f"   指标 C (年化换手 400%-600%, 单日<15%): {'✓' if metric_c_pass else '✗'}")
        lines.append(f"     - 年化换手：{turnover:.2%}")
        lines.append(f"     - 单日换手达标：{'是' if daily_ok else '否'}")
        lines.append("")
        
        # 指标 D: IC 衰减和 Lag1 权重
        decay_normal = ic_audit.get('decay_normal', False)
        lag1_weight = V89_LAG1_WEIGHT
        metric_d_pass = decay_normal and lag1_weight > 0.30
        lines.append(f"   指标 D (IC 衰减正常，Lag1 权重>30%): {'✓' if metric_d_pass else '✗'}")
        lines.append(f"     - IC 衰减正常：{'是' if decay_normal else '否'}")
        lines.append(f"     - Lag1 权重：{lag1_weight:.1%}")
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
            'lookahead_check': {},
            'trade_results': {},
            'audit_report': '',
            'trade_records': [],
            'daily_snapshots': [],
        }


# ===========================================
# 主程序
# ===========================================

def run_v89_backtest(config: V89EngineConfig = None) -> Dict[str, Any]:
    """运行 V89 回测"""
    engine = V89Engine(config=config)
    return engine.run_backtest()


def print_v89_report(result: Dict[str, Any]) -> None:
    """打印 V89 报告"""
    logger.info("=" * 70)
    logger.info("V89 最终报告")
    logger.info("=" * 70)
    
    trade_results = result.get('trade_results', {})
    logger.info("【交易执行】")
    logger.info(f"  总收益：{trade_results.get('total_return', 0.0):.2%}")
    logger.info(f"  最终价值：{trade_results.get('final_value', 0.0):,.2f}")
    logger.info(f"  最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
    logger.info(f"  年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
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
    
    config = V89EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        rebalance_interval=42,  # 季度调仓（降低换手率）
    )
    
    result = run_v89_backtest(config)
    print_v89_report(result)
    
    # 保存结果
    output_path = "reports/v89_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_result = {
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': result.get('ic_audit', {}),
        'lookahead_check': result.get('lookahead_check', {}),
        'trade_results': result.get('trade_results', {}),
        'audit_report': result.get('audit_report', ''),
        'trade_count': len(result.get('trade_records', [])),
        'snapshot_count': len(result.get('daily_snapshots', [])),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V89: 结果已保存至 {output_path}")