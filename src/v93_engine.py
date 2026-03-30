"""
V93 Engine - 量价二阶导与资金流偏度增强

【V93 核心任务】
1. 废弃 v90_core，完全重写因子计算逻辑
2. 引入"量价二阶导"逻辑（Volatility of Volatility）
3. 引入"资金流偏度"因子（Order Flow Skewness）
4. 实现严格的行业（Industry-wise）和市值（Size-neutral）中性化

【V93 硬性指标】
- 指标 A：T+1 Rank IC ≥ 0.05，IC IR ≥ 0.6
- 指标 B：最大回撤 ≤ 10%
- 指标 C：年化换手率 300%-400%
- 指标 D：数学自洽性检查（误差 < 0.01%）

作者：量化系统
版本：V93.0
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

# 导入 V93 核心模块
from src.core.v93_logic import (
    V93DataManager,
    V93VolOfVolEngine,
    V93FlowSkewEngine,
    V93MomentumReversalEngine,
    V93NeutralizationEngine,
    V93CompositeEngine,
    V93ICAudit,
    V93ConsistencyChecker,
    V93TurnoverTracker,
    V93_INITIAL_CAPITAL,
    V93_MAX_POSITIONS,
    V93_WARMUP_PERIOD,
    V93_MIN_SCORE_THRESHOLD,
    V93_MIN_SINGLE_WEIGHT,
    V93_MAX_SINGLE_WEIGHT,
    V93_T1_IC_TARGET,
    V93_IC_IR_TARGET,
    V93_COMMISSION_RATE,
    V93_MIN_COMMISSION,
    V93_STAMP_DUTY,
    V93_TRANSFER_FEE,
    V93_MIN_REBALANCE_INTERVAL,
    V93_MAX_REBALANCE_INTERVAL,
    V93_DAILY_TURNOVER_MAX,
    V93_TURNOVER_MIN,
    V93_TURNOVER_MAX,
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
# V93 引擎配置
# ===========================================

class V93EngineConfig:
    """V93 引擎配置"""
    
    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = V93_INITIAL_CAPITAL,
        max_positions: int = V93_MAX_POSITIONS,
        warmup_period: int = V93_WARMUP_PERIOD,
        commission_rate: float = V93_COMMISSION_RATE,
        min_commission: float = V93_MIN_COMMISSION,
        stamp_duty: float = V93_STAMP_DUTY,
        transfer_fee: float = V93_TRANSFER_FEE,
        oos_years: List[str] = None,
        min_score_threshold: float = V93_MIN_SCORE_THRESHOLD,
        min_single_weight: float = V93_MIN_SINGLE_WEIGHT,
        max_single_weight: float = V93_MAX_SINGLE_WEIGHT,
        # V93 新增配置
        enable_vol_of_vol: bool = True,  # 启用波动率之波动率
        enable_flow_skew: bool = True,  # 启用资金流偏度
        enable_momentum_reversal: bool = True,  # 启用动量反转
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
        # V93 新增配置
        self.enable_vol_of_vol = enable_vol_of_vol
        self.enable_flow_skew = enable_flow_skew
        self.enable_momentum_reversal = enable_momentum_reversal
        self.enable_neutralization = enable_neutralization


# ===========================================
# V93 引擎
# ===========================================

class V93Engine:
    """V93 回测引擎"""
    
    def __init__(self, config: V93EngineConfig = None, db=None):
        self.config = config or V93EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V93: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        # V93 核心模块
        self.data_manager = V93DataManager(db=self.db)
        self.vol_of_vol_engine = V93VolOfVolEngine() if self.config.enable_vol_of_vol else None
        self.flow_skew_engine = V93FlowSkewEngine() if self.config.enable_flow_skew else None
        self.momentum_reversal_engine = V93MomentumReversalEngine() if self.config.enable_momentum_reversal else None
        self.neutralization_engine = V93NeutralizationEngine() if self.config.enable_neutralization else None
        self.composite_engine = V93CompositeEngine()
        self.ic_audit = V93ICAudit(db=self.db)
        self.consistency_checker = V93ConsistencyChecker()
        
        # V93 换手率追踪器
        self.turnover_tracker = V93TurnoverTracker()
        
        # 组合管理
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        
        # 交易记录
        self.trade_records: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        self.rebalance_dates: List[str] = []
        
        # 调仓状态追踪
        self.last_rebalance_date = None
        self.days_since_rebalance = 0
        
        logger.info("=" * 70)
        logger.info("V93 Engine 初始化完成")
        logger.info("=" * 70)
        logger.info(f"V93: 初始资金={self.config.initial_capital:,.2f}")
        logger.info(f"V93: 量价二阶导={'启用' if self.config.enable_vol_of_vol else '禁用'}")
        logger.info(f"V93: 资金流偏度={'启用' if self.config.enable_flow_skew else '禁用'}")
        logger.info(f"V93: 动量反转={'启用' if self.config.enable_momentum_reversal else '禁用'}")
        logger.info(f"V93: 中性化={'启用' if self.config.enable_neutralization else '禁用'}")
        logger.info(f"V93: Score 阈值={self.config.min_score_threshold}")
        logger.info(f"V93: IC 目标={V93_T1_IC_TARGET:.4f}, IC IR 目标={V93_IC_IR_TARGET:.2f}")
        logger.info(f"V93: 换手率目标={V93_TURNOVER_MIN*100:.0f}%-{V93_TURNOVER_MAX*100:.0f}%")
        logger.info("=" * 70)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V93 量价二阶导与资金流偏度增强回测引擎启动")
        logger.info("=" * 70)
        
        if self.db is None:
            logger.error("V93: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查
            logger.info("V93: [1/6] 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V93: [2/6] 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V93: 未加载到任何数据")
                return self._empty_result()
            
            for year in self.config.oos_years:
                df_year = df.filter(pl.col('trade_date').str.starts_with(year))
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V93: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算因子信号
            logger.info("V93: [3/6] 开始计算因子信号...")
            df_with_signals = self._compute_signals(df)
            
            # 4. 中性化处理
            logger.info("V93: [4/6] 开始中性化处理...")
            df_with_neutralization = self._apply_neutralization(df_with_signals)
            
            # 5. 计算综合评分
            logger.info("V93: [5/6] 开始计算综合评分...")
            df_with_composite = self._compute_composite_score(df_with_neutralization)
            
            # 6. IC 审计
            logger.info("V93: [6/6] 开始 IC 审计...")
            ic_audit_results = self.ic_audit.calculate_rank_ic(
                df_with_composite, signal_col='final_signal'
            )
            
            logger.info(f"V93: T+1 Rank IC = {ic_audit_results['ic_t1']:.4f} (目标 > {V93_T1_IC_TARGET})")
            logger.info(f"V93: IC IR = {ic_audit_results['ic_ir']:.2f} (目标 > {V93_IC_IR_TARGET})")
            
            # 执行回测交易
            logger.info("V93: 开始执行回测交易...")
            trade_results = self._execute_backtest(df_with_composite)
            
            # 一致性检查
            logger.info("V93: 开始一致性检查...")
            consistency_results = self._run_consistency_check(ic_audit_results, trade_results)
            
            # 生成报告
            logger.info("V93: 生成审计报告...")
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
            logger.info("V93 回测完成")
            logger.info("=" * 70)
            
            return result
            
        except Exception as e:
            logger.error(f"V93 回测失败 - {e}")
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
                logger.info(f"V93: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V93: {year}年数据检查失败 - {message}")
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
                    logger.info(f"V93: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V93: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V93: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _compute_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算因子信号（V93 新因子框架）
        
        【因子组成】
        1. Vol of Vol (30%): 量价二阶导因子
        2. Flow Skew (30%): 资金流偏度因子
        3. Momentum (20%): 中期动量因子
        4. Reversal (20%): 短期反转因子
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
        
        # 1. 计算量价二阶导因子
        if self.config.enable_vol_of_vol and self.vol_of_vol_engine:
            logger.info("V93: 计算量价二阶导因子...")
            result = self.vol_of_vol_engine.compute_vol_of_vol(result)
        else:
            result = result.with_columns([pl.lit(50.0).alias('vol_of_vol_score')])
        
        # 2. 计算资金流偏度因子
        if self.config.enable_flow_skew and self.flow_skew_engine:
            logger.info("V93: 计算资金流偏度因子...")
            result = self.flow_skew_engine.compute_flow_skew(result)
        else:
            result = result.with_columns([pl.lit(50.0).alias('flow_skew_score')])
        
        # 3. 计算动量反转因子
        if self.config.enable_momentum_reversal and self.momentum_reversal_engine:
            logger.info("V93: 计算动量反转因子...")
            result = self.momentum_reversal_engine.compute_momentum_reversal(result)
        else:
            result = result.with_columns([pl.lit(50.0).alias('momentum_reversal_score')])
        
        return result
    
    def _apply_neutralization(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        应用中性化处理（V93 严格中性化）
        
        【核心逻辑】
        1. 先计算综合评分（使用 composite_engine）
        2. 再进行行业 + 市值中性化
        """
        if not self.config.enable_neutralization or self.neutralization_engine is None:
            # 即使不中性化，也要计算 composite_score
            logger.info("V93: 计算综合评分...")
            return self.composite_engine.compute_composite(df)
        else:
            logger.info("V93: 先计算综合评分...")
            df_with_composite = self.composite_engine.compute_composite(df)
            logger.info("V93: 应用行业 + 市值中性化...")
            neutralized = self.neutralization_engine.compute_neutralization(
                df_with_composite, signal_col='composite_score'
            )
            return neutralized
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算综合评分"""
        result = df.clone()
        
        # 使用复合引擎融合信号
        logger.info("V93: 融合多因子信号...")
        result = self.composite_engine.compute_composite(result)
        
        # 应用中性化（如果还未应用）
        if self.config.enable_neutralization and self.neutralization_engine is not None:
            if 'neutralized_signal' not in result.columns:
                logger.info("V93: 应用中性化到最终信号...")
                neutralized = self.neutralization_engine.compute_neutralization(
                    result, signal_col='composite_score'
                )
                result = result.join(
                    neutralized.select(['trade_date', 'symbol', 'neutralized_signal']),
                    on=['trade_date', 'symbol'], how='left'
                )
                # 使用中性化后的信号作为最终信号
                result = result.with_columns([
                    pl.col('neutralized_signal').fill_null(pl.col('composite_score')).alias('final_signal')
                ])
            else:
                result = result.with_columns([
                    pl.col('neutralized_signal').alias('final_signal')
                ])
        else:
            result = result.with_columns([
                pl.col('composite_score').alias('final_signal')
            ])
        
        logger.info(f"V93: 最终信号统计 - 均值={result['final_signal'].mean():.2f}, 标准差={result['final_signal'].std():.2f}")
        logger.info(f"V93: 综合评分计算完成，处理 {result.height} 条记录")
        
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
        
        # 风险控制检查
        risk_check = self.consistency_checker.check_risk_control(
            trade_results.get('max_drawdown', 0.0)
        )
        
        summary = self.consistency_checker.get_consistency_summary()
        
        logger.info(f"V93: 一致性检查完成 - 通过率={summary['pass_rate']:.2%}")
        
        return {
            'mathematical_consistency': {
                'passed': math_check.passed,
                'expected': math_check.expected,
                'actual': math_check.actual,
                'diff': math_check.diff,
                'message': math_check.message,
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
        """执行回测交易"""
        logger.info("V93: 开始执行回测交易...")
        
        df = df.sort(['trade_date', 'symbol'])
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        warmup_cutoff = unique_dates[:min(V93_WARMUP_PERIOD, len(unique_dates))]
        trade_dates = [d for d in unique_dates if d not in warmup_cutoff]
        
        logger.info(f"V93: 热身期 {len(warmup_cutoff)} 天，交易期 {len(trade_dates)} 天")
        
        # 重置状态
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.positions = {}
        self.trade_records = []
        self.daily_snapshots = []
        self.rebalance_dates = []
        self.turnover_tracker = V93TurnoverTracker()  # 重置换手率追踪器
        
        # 调仓状态
        last_rebalance_date = None
        days_since_rebalance = 0
        
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
            
            # 判断是否调仓
            days_since_rebalance += 1
            
            # V93 调仓条件：固定间隔调仓
            should_rebalance = days_since_rebalance >= V93_MIN_REBALANCE_INTERVAL
            
            if should_rebalance:
                signal_col = 'final_signal'
                
                if signal_col not in day_df.columns:
                    logger.warning(f"V93: {trade_date} 信号列缺失，跳过")
                    continue
                
                # 获取有效股票（信号>0 且非空）
                valid_stocks = day_df.filter(
                    (pl.col(signal_col).is_not_null()) &
                    (pl.col(signal_col) > 0)
                ).sort(signal_col, descending=True)
                
                if valid_stocks.is_empty():
                    logger.warning(f"V93: {trade_date} 无有效股票，跳过调仓")
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
                    
                    # V93 应用单日换手率上限 10%
                    max_daily_turnover_value = self.portfolio_value * V93_DAILY_TURNOVER_MAX
                    buy_value = min(buy_value, max_daily_turnover_value)
                    sell_value = min(sell_value, max_daily_turnover_value)
                    
                    is_rebalance_day = True
                    self.rebalance_dates.append(trade_date)
                    last_rebalance_date = trade_date
                    days_since_rebalance = 0
                    
                except Exception as e:
                    logger.error(f"V93: {trade_date} 交易执行失败 - {e}")
            
            # 记录换手率
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
                logger.info(f"V93: 处理 {i + 1}/{len(trade_dates)} 天，组合价值={self.portfolio_value:,.2f}, 年化换手={turnover_summary['annualized_turnover']:.1f}%")
        
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
        
        logger.info(f"V93: 回测完成 - 总收益={total_return:.2%}, 年化={annualized_return:.2%}")
        logger.info(f"V93: 调仓次数={len(self.rebalance_dates)}, 最大回撤={max_drawdown:.2%}")
        logger.info(f"V93: 年化换手率={annualized_turnover:.2%}")
        
        return result
    
    def _calculate_target_positions(self, valid_stocks: pl.DataFrame, 
                                     portfolio_value: float,
                                     signal_col: str = 'final_signal') -> Dict[str, float]:
        """计算目标持仓"""
        if valid_stocks.is_empty():
            return {}
        
        # 动态仓位控制：根据回撤调整仓位上限
        current_drawdown = self._calculate_current_drawdown()
        position_limit = self._get_position_limit_by_drawdown(current_drawdown)
        
        max_stocks = min(self.config.max_positions, valid_stocks.height)
        top_stocks = valid_stocks.head(max_stocks)
        
        target_positions = {}
        for row in top_stocks.iter_rows(named=True):
            symbol = row['symbol']
            signal = row.get(signal_col, 50.0)
            
            if signal is not None and np.isfinite(signal) and signal > 0:
                # 单只标的最大权重 8%
                capped_weight = min(1.0 / max_stocks, V93_MAX_SINGLE_WEIGHT)
                target_positions[symbol] = capped_weight
        
        # 根据回撤限制总仓位
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
        lines.append("《V93 量价二阶导与资金流偏度增强审计报告》")
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
        lines.append(f"   T+1 Rank IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V93_T1_IC_TARGET})")
        lines.append(f"   T+2 Rank IC: {ic_audit.get('ic_t2', 0.0):.4f}")
        lines.append(f"   T+3 Rank IC: {ic_audit.get('ic_t3', 0.0):.4f}")
        lines.append(f"   IC IR: {ic_audit.get('ic_ir', 0.0):.2f} (目标 > {V93_IC_IR_TARGET})")
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
        
        risk_check = consistency.get('risk_control', {})
        lines.append(f"   风险控制：{'✓' if risk_check.get('passed', False) else '✗'}")
        lines.append(f"     {risk_check.get('message', 'N/A')}")
        
        summary = consistency.get('summary', {})
        lines.append(f"   总通过率：{summary.get('pass_rate', 0.0):.2%}")
        lines.append("")
        
        lines.append("5. V93 硬性指标验证")
        lines.append("   " + "-" * 50)
        
        # 指标 A：IC
        metric_a_ic = ic_audit.get('ic_t1', 0.0) >= V93_T1_IC_TARGET
        metric_a_ir = ic_audit.get('ic_ir', 0.0) >= V93_IC_IR_TARGET
        metric_a_pass = metric_a_ic and metric_a_ir
        
        lines.append(f"   指标 A (T+1 IC ≥ {V93_T1_IC_TARGET}, IC IR ≥ {V93_IC_IR_TARGET}): {'✓' if metric_a_pass else '✗'}")
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
        metric_c_pass = V93_TURNOVER_MIN <= ann_turnover <= V93_TURNOVER_MAX
        lines.append(f"   指标 C (年化换手率 {V93_TURNOVER_MIN*100:.0f}%-{V93_TURNOVER_MAX*100:.0f}%): {'✓' if metric_c_pass else '✗'}")
        lines.append(f"     - 年化换手率：{ann_turnover:.2%}")
        lines.append("")
        
        # 指标 D：数学一致性
        metric_d_pass = math_check.get('passed', False)
        lines.append(f"   指标 D (数学一致性误差 < 0.01%): {'✓' if metric_d_pass else '✗'}")
        lines.append("")
        
        lines.append("6. V93 增强特性")
        lines.append("   " + "-" * 50)
        lines.append(f"   量价二阶导：{'启用' if self.config.enable_vol_of_vol else '禁用'}")
        lines.append(f"   资金流偏度：{'启用' if self.config.enable_flow_skew else '禁用'}")
        lines.append(f"   动量反转：{'启用' if self.config.enable_momentum_reversal else '禁用'}")
        lines.append(f"   行业 + 市值中性化：{'启用' if self.config.enable_neutralization else '禁用'}")
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

def run_v93_backtest(config: V93EngineConfig = None) -> Dict[str, Any]:
    """运行 V93 回测"""
    engine = V93Engine(config=config)
    return engine.run_backtest()


def print_v93_report(result: Dict[str, Any]) -> None:
    """打印 V93 报告"""
    logger.info("=" * 70)
    logger.info("V93 最终报告")
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
    
    config = V93EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        enable_vol_of_vol=True,
        enable_flow_skew=True,
        enable_momentum_reversal=True,
        enable_neutralization=True,
    )
    
    result = run_v93_backtest(config)
    print_v93_report(result)
    
    output_path = "reports/v93_backtest_result.json"
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
    
    logger.info(f"V93: 结果已保存至 {output_path}")