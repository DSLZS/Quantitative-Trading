"""
V99 Engine - 净收益保卫战：持仓缓冲与交易成本强约束

【V99 核心改进】
1. 持仓缓冲逻辑（Position Buffer）
   - 买入条件：Alpha 排名进入前 20 名
   - 卖出条件：跌出前 60 名（缓冲区域 20-60 名）
   - 禁止"全仓换血"

2. 每日调仓比例限制
   - 每日调仓不超过 15%
   - 分批执行，避免一次性换仓

3. 交易成本强约束
   - TRANSACTION_COST = 0.0015（单边）
   - 硬编码扣费，输出"扣费后净收益"

4. 时间序列稳定性过滤
   - 剔除过去 5 天 IC 波动率过大的因子分量

5. industry_code 缺失检测
   - 缺失比例 > 10% 时主动调用重新补取数据

【V99 验收硬指标】
| 指标 | 目标值 | 失败判定 |
| :--- | :--- | :--- |
| 扣费后年化收益 | > 15% | 任何因换手率导致的亏损直接判定为负优化 |
| 年化换手率 | 300% - 450% | 超过 500% 立即重写持仓逻辑 |
| T+1 Rank IC | > 0.048 | 必须维持在正向且稳定 |
| IC Stability | > 0.4 | 证明预测不是随机的 |

作者：量化系统
版本：V99.0
日期：2026-03-31
"""

import sys
import os
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import numpy as np
import polars as pl
from loguru import logger

# 导入 V99 核心模块
from src.core.v99_core import (
    DirectionalError,
    IndustryDataMissingError,
    V99_INITIAL_CAPITAL,
    V99_MAX_POSITIONS,
    V99_WARMUP_PERIOD,
    V99_TURNOVER_MIN,
    V99_TURNOVER_MAX,
    V99_DAILY_TURNOVER_MAX,
    V99_BUY_RANK_THRESHOLD,
    V99_SELL_RANK_THRESHOLD,
    V99_MAX_DAILY_REBALANCE_RATIO,
    V99_TRANSACTION_COST,
    V99_T1_IC_TARGET,
    V99_IC_STABILITY_TARGET,
    V99_COMMISSION_RATE,
    V99_MIN_COMMISSION,
    V99_STAMP_DUTY,
    V99_TRANSFER_FEE,
    V99_RESIDUAL_WEIGHT,
    V99_FLOW_WEIGHT,
    V99_EMA_WINDOW,
    V99_AUDIT_MODE,
    V99_INDUSTRY_MISSING_THRESHOLD,
    V99DataManager,
    V99SignalSmoother,
    V99PositionBuffer,
    V99RebalanceLimiter,
    V99ICStabilityFilter,
    V99IndustryChecker,
    V99TurnoverTracker,
    V99TransactionCostCalculator,
    V99ICAudit,
    V99ResidualMomentumEngine,
    V99SmartFlowEngine,
    V99AlphaFusion,
    V99AlphaWeightEngine,
    V99Position,
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
# V99 引擎配置
# ===========================================

class V99EngineConfig:
    """V99 引擎配置"""
    
    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        initial_capital: float = V99_INITIAL_CAPITAL,
        max_positions: int = V99_MAX_POSITIONS,
        warmup_period: int = V99_WARMUP_PERIOD,
        commission_rate: float = V99_COMMISSION_RATE,
        min_commission: float = V99_MIN_COMMISSION,
        stamp_duty: float = V99_STAMP_DUTY,
        transfer_fee: float = V99_TRANSFER_FEE,
        transaction_cost: float = V99_TRANSACTION_COST,
        oos_years: List[str] = None,
        min_score_threshold: float = 45.0,
        # 持仓缓冲配置
        buy_rank_threshold: int = V99_BUY_RANK_THRESHOLD,
        sell_rank_threshold: int = V99_SELL_RANK_THRESHOLD,
        # 调仓比例限制
        max_daily_rebalance_ratio: float = V99_MAX_DAILY_REBALANCE_RATIO,
        # 审计
        audit_mode: bool = True,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital  # 严格锁定 100,000.00
        self.max_positions = max_positions
        self.warmup_period = warmup_period
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty = stamp_duty
        self.transfer_fee = transfer_fee
        self.transaction_cost = transaction_cost  # 硬编码 0.0015
        self.oos_years = oos_years or ["2019", "2021", "2024"]
        self.min_score_threshold = min_score_threshold
        
        # 持仓缓冲配置
        self.buy_rank_threshold = buy_rank_threshold
        self.sell_rank_threshold = sell_rank_threshold
        
        # 调仓比例限制
        self.max_daily_rebalance_ratio = max_daily_rebalance_ratio
        
        self.audit_mode = audit_mode


# ===========================================
# V99 引擎
# ===========================================

class V99Engine:
    """V99 回测引擎 - 净收益保卫战"""
    
    def __init__(self, config: V99EngineConfig = None, db=None):
        self.config = config or V99EngineConfig()
        
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
                logger.info("V99: 数据库连接池已初始化")
            except Exception as e:
                logger.error(f"V99: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        self.data_manager = V99DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
        })
        
        # V99 核心组件
        self.position_buffer = V99PositionBuffer(config={
            'buy_threshold': self.config.buy_rank_threshold,
            'sell_threshold': self.config.sell_rank_threshold,
        })
        
        self.rebalance_limiter = V99RebalanceLimiter(config={
            'max_rebalance_ratio': self.config.max_daily_rebalance_ratio,
        })
        
        self.industry_checker = V99IndustryChecker(config={
            'missing_threshold': V99_INDUSTRY_MISSING_THRESHOLD,
        })
        
        # V90 基准因子
        self.residual_momentum = V99ResidualMomentumEngine()
        self.smart_flow = V99SmartFlowEngine()
        
        self.alpha_fusion = V99AlphaFusion(db=self.db, config={
            'fusion_lags': [1, 3, 5],
        })
        self.alpha_weight = V99AlphaWeightEngine(config={
            'min_score': self.config.min_score_threshold,
        })
        self.ic_audit = V99ICAudit(db=self.db, config={
            'lookback_days': 5,
            'std_threshold': 0.15,
        })
        self.turnover_tracker = V99TurnoverTracker()
        self.cost_calculator = V99TransactionCostCalculator(config={
            'transaction_cost': self.config.transaction_cost,
            'commission_rate': self.config.commission_rate,
            'min_commission': self.config.min_commission,
            'stamp_duty': self.config.stamp_duty,
            'transfer_fee': self.config.transfer_fee,
        })
        
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, V99Position] = {}
        
        self.trade_records: List[Dict] = []
        self.daily_snapshots: List[Dict] = []
        self.rebalance_dates: List[str] = []
        
        # 扣费后收益追踪
        self.total_transaction_cost = 0.0
        self.net_portfolio_value = self.config.initial_capital
        
        logger.info("=" * 70)
        logger.info("V99 Engine 初始化完成 - 净收益保卫战")
        logger.info("=" * 70)
        logger.info(f"V99: 初始资金 = {self.config.initial_capital:,.2f} (严格锁定)")
        logger.info(f"V99: 最大持仓数 = {self.config.max_positions}")
        logger.info(f"V99: 评分门槛 = {self.config.min_score_threshold}")
        logger.info(f"V99: 交易成本 = {self.config.transaction_cost:.2%} (单边硬编码)")
        logger.info("=" * 70)
        logger.info("V99 持仓缓冲配置:")
        logger.info(f"  买入阈值：前 {self.config.buy_rank_threshold} 名")
        logger.info(f"  卖出阈值：跌出前 {self.config.sell_rank_threshold} 名")
        logger.info(f"  缓冲区域：{self.config.buy_rank_threshold}-{self.config.sell_rank_threshold} 名")
        logger.info("=" * 70)
        logger.info(f"V99: 每日调仓上限 = {self.config.max_daily_rebalance_ratio:.1%}")
        logger.info("=" * 70)
        logger.info("V99 验收硬指标:")
        logger.info(f"  扣费后年化收益 > 15%")
        logger.info(f"  年化换手率 {V99_TURNOVER_MIN:.0f}% - {V99_TURNOVER_MAX:.0f}% (超过 500% 失败)")
        logger.info(f"  T+1 Rank IC > {V99_T1_IC_TARGET}")
        logger.info(f"  IC Stability > {V99_IC_STABILITY_TARGET}")
        logger.info("=" * 70)
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V99 净收益保卫战引擎启动")
        logger.info("=" * 70)
        
        if self.db is None:
            logger.error("V99: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查
            logger.info("V99: [1/7] 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V99: [2/7] 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V99: 未加载到任何数据")
                return self._empty_result()
            
            for year in self.config.oos_years:
                df_year = df.filter(
                    pl.col('trade_date').cast(pl.Utf8).str.starts_with(year)
                )
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V99: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
            
            # 3. 计算 V90 基准因子信号
            logger.info("V99: [3/7] 开始计算 V90 基准因子信号...")
            df_with_signals = self._compute_v90_signals(df)
            
            # 4. 计算综合评分
            logger.info("V99: [4/7] 开始计算综合评分...")
            df_with_signals = self._compute_composite_score(df_with_signals)
            
            # 5. 消融实验：单因子 IC 审计
            if self.config.audit_mode:
                logger.info("V99: [5/7] 开始消融实验 - 单因子 IC 审计...")
                self._run_ablation_study(df_with_signals)
            
            # 6. 半衰期融合 + EMA 平滑
            logger.info("V99: [6/7] 开始半衰期融合 + EMA 平滑...")
            df_with_fusion = self.alpha_fusion.compute_fusion_signal(
                df_with_signals, signal_col='composite_score'
            )
            
            # 7. Alpha 权重与 IC 审计
            logger.info("V99: [7/7] 开始 Alpha 权重计算与 IC 审计...")
            df_with_weights = self.alpha_weight.compute_alpha_weights(
                df_with_fusion, score_col='smoothed_signal'
            )
            
            ic_audit_results = self.ic_audit.calculate_rank_ic(
                df_with_weights, signal_col='smoothed_signal'
            )
            
            logger.info(f"V99: T+1 Rank IC = {ic_audit_results['ic_t1']:.4f} (目标 > {V99_T1_IC_TARGET})")
            logger.info(f"V99: T+2 Rank IC = {ic_audit_results['ic_t2']:.4f}")
            logger.info(f"V99: T+3 Rank IC = {ic_audit_results['ic_t3']:.4f}")
            logger.info(f"V99: IC Stability = {ic_audit_results['ic_stability']:.3f} (目标 > {V99_IC_STABILITY_TARGET})")
            
            # 执行回测交易
            logger.info("V99: 开始执行回测交易...")
            trade_results = self._execute_backtest(df_with_weights)
            
            # 生成报告
            logger.info("V99: 生成审计报告...")
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
            }
            
            logger.info("=" * 70)
            logger.info("V99 回测完成")
            logger.info("=" * 70)
            
            return result
            
        except DirectionalError as e:
            logger.error(f"V99: 因子方向错误 - {e}")
            return self._empty_result()
        except IndustryDataMissingError as e:
            logger.error(f"V99: 行业数据缺失 - {e}")
            return self._empty_result()
        except Exception as e:
            logger.error(f"V99 回测失败 - {e}")
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
                logger.info(f"V99: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V99: {year}年数据检查失败 - {message}")
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
                    logger.info(f"V99: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V99: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V99: 总数据行数={combined_df.height:,}")
        
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
    
    def _compute_composite_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 V99 综合评分"""
        result = df.clone()
        
        for col, default in [
            ('residual_momentum_score', 50.0),
            ('smart_flow_score', 50.0),
        ]:
            if col not in result.columns:
                result = result.with_columns([pl.lit(default).alias(col)])
        
        # V99: V90 基准权重
        result = result.with_columns([
            (V99_RESIDUAL_WEIGHT * pl.col('residual_momentum_score') + 
             V99_FLOW_WEIGHT * pl.col('smart_flow_score')).alias('composite_score')
        ])
        
        return result
    
    def _run_ablation_study(self, df: pl.DataFrame) -> None:
        """消融实验：单因子 IC 审计"""
        logger.info("=" * 50)
        logger.info("V99 消融实验 - 单因子 IC 审计")
        logger.info("=" * 50)
        
        self.factor_ic_records = {}
        
        # V90 基准因子 IC
        try:
            ic_residual = self.ic_audit.calculate_single_factor_ic(
                df, 'residual_momentum', 'residual_momentum_score'
            )
            
            self.factor_ic_records['residual_momentum'] = {
                'ic_t1': ic_residual.ic_t1,
                'ic_t2': ic_residual.ic_t2,
                'ic_t3': ic_residual.ic_t3,
                'ic_ir': ic_residual.ic_ir,
                'ic_stability': ic_residual.ic_stability,
                'ic_std_5d': ic_residual.ic_std_5d,
                'passed_stability': ic_residual.passed_stability_filter,
            }
            
            logger.info(f"V90 基准因子 IC:")
            logger.info(f"  残差动量：T+1={ic_residual.ic_t1:.4f}, T+2={ic_residual.ic_t2:.4f}, T+3={ic_residual.ic_t3:.4f}")
            logger.info(f"  残差动量：IC IR={ic_residual.ic_ir:.3f}, IC Stability={ic_residual.ic_stability:.3f}")
            logger.info(f"  残差动量：IC Std(5d)={ic_residual.ic_std_5d:.4f}, 稳定性过滤={'通过' if ic_residual.passed_stability_filter else '未通过'}")
            logger.info(f"  残差动量：达标={'是' if ic_residual.passed_threshold else '否'}")
            
        except DirectionalError as e:
            logger.error(f"V99: 残差动量因子方向错误 - {e}")
            raise
        
        ic_flow = self.ic_audit.calculate_single_factor_ic(
            df, 'smart_flow', 'smart_flow_score'
        )
        
        self.factor_ic_records['smart_flow'] = {
            'ic_t1': ic_flow.ic_t1,
            'ic_t2': ic_flow.ic_t2,
            'ic_t3': ic_flow.ic_t3,
            'ic_ir': ic_flow.ic_ir,
            'ic_stability': ic_flow.ic_stability,
            'ic_std_5d': ic_flow.ic_std_5d,
            'passed_stability': ic_flow.passed_stability_filter,
        }
        
        logger.info(f"  聪明资金流：T+1={ic_flow.ic_t1:.4f}, T+2={ic_flow.ic_t2:.4f}, T+3={ic_flow.ic_t3:.4f}")
        logger.info(f"  聪明资金流：IC IR={ic_flow.ic_ir:.3f}, IC Stability={ic_flow.ic_stability:.3f}")
        logger.info(f"  聪明资金流：IC Std(5d)={ic_flow.ic_std_5d:.4f}, 稳定性过滤={'通过' if ic_flow.passed_stability_filter else '未通过'}")
        
        # V90 基准组合 IC
        v90_score = (V99_RESIDUAL_WEIGHT * df['residual_momentum_score'] + 
                     V99_FLOW_WEIGHT * df['smart_flow_score'])
        df_v90 = df.clone()
        df_v90 = df_v90.with_columns([v90_score.alias('v90_combined_score')])
        ic_v90 = self.ic_audit.calculate_rank_ic(df_v90, 'v90_combined_score')
        
        self.factor_ic_records['v90_baseline'] = {
            'ic_t1': ic_v90['ic_t1'],
            'ic_t2': ic_v90['ic_t2'],
            'ic_t3': ic_v90['ic_t3'],
            'ic_stability': ic_v90.get('ic_stability', 0.0),
        }
        logger.info(f"V90 基准组合 IC: T+1={ic_v90['ic_t1']:.4f}, IC Stability={ic_v90.get('ic_stability', 0.0):.3f}")
        
        logger.info("=" * 50)
    
    def _execute_backtest(self, df: pl.DataFrame) -> Dict[str, Any]:
        """执行回测交易（带持仓缓冲与交易成本强约束）"""
        logger.info("V99: 开始执行回测交易...")
        
        df = df.sort(['trade_date', 'symbol'])
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        warmup_cutoff = unique_dates[:min(V99_WARMUP_PERIOD, len(unique_dates))]
        trade_dates = [d for d in unique_dates if d not in warmup_cutoff]
        
        logger.info(f"V99: 热身期 {len(warmup_cutoff)} 天，交易期 {len(trade_dates)} 天")
        
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.positions = {}
        self.trade_records = []
        self.daily_snapshots = []
        self.rebalance_dates = []
        self.total_transaction_cost = 0.0
        
        prev_date = None
        total_position_count = 0
        snapshot_count = 0
        
        prev_ranks: Dict[str, int] = {}
        
        for i, trade_date in enumerate(trade_dates):
            day_df = df.filter(pl.col('trade_date') == trade_date)
            
            if day_df.is_empty():
                continue
            
            # V99 强制：行业数据缺失检测
            passed, missing_ratio = self.industry_checker.check_industry_coverage(
                day_df, str(trade_date)
            )
            if not passed:
                logger.error(f"V99: {trade_date} 行业数据缺失比例 {missing_ratio:.1%} > 10%")
                # 尝试修复数据
                day_df = self.industry_checker.repair_industry_data(day_df, self.db)
            
            price_map = dict(zip(
                day_df['symbol'].to_list(),
                day_df['close'].to_list()
            ))
            
            # 更新持仓价格和排名
            for symbol, position in self.positions.items():
                if symbol in price_map:
                    position.current_price = price_map[symbol]
                    position.pnl = (price_map[symbol] - position.entry_price) * position.quantity
            
            # 计算组合价值
            if self.positions:
                position_value = sum(
                    p.current_price * p.quantity for p in self.positions.values()
                )
            else:
                position_value = 0.0
            self.portfolio_value = self.cash + position_value
            
            # 计算扣费后组合价值
            self.net_portfolio_value = self.portfolio_value - self.total_transaction_cost
            
            # 提取信号并计算排名
            signals = {}
            for row in day_df.iter_rows(named=True):
                symbol = row['symbol']
                signal = row.get('smoothed_signal', row.get('fused_signal', 0.0))
                if signal is not None and np.isfinite(signal):
                    signals[symbol] = float(signal)
            
            # 计算股票排名
            stock_ranks = self._calculate_stock_ranks(signals)
            
            buy_value = 0.0
            sell_value = 0.0
            
            # V99 核心：持仓缓冲逻辑
            target_buy, symbols_to_sell = self.position_buffer.get_target_positions(
                stock_ranks, self.positions, self.config.max_positions
            )
            
            # 强制：限制目标持仓数量不超过 max_positions
            target_buy = target_buy[:self.config.max_positions]
            
            # V99 核心：调仓比例限制（严格控制换手率）
            # 每 5 天调仓一次，大幅降低换手率
            should_rebalance = (i % 5 == 0)
            
            # 准备交易列表
            trades_to_execute = []
            
            # 只在调仓日执行交易
            if should_rebalance:
                # 卖出交易 - 优先级 1: 跌出前 60 名的股票
                for symbol in symbols_to_sell:
                    if symbol in self.positions and symbol in price_map:
                        position = self.positions[symbol]
                        sell_amount = price_map[symbol] * position.quantity
                        cost = self.cost_calculator.calculate_sell_cost(sell_amount)
                        
                        trades_to_execute.append({
                            'symbol': symbol,
                            'action': 'sell',
                            'price': price_map[symbol],
                            'quantity': position.quantity,
                            'amount': sell_amount,
                            'cost': cost,
                        })
                
                # 强制：如果当前持仓超过 max_positions，卖出排名最差的
                if len(self.positions) > self.config.max_positions:
                    # 获取当前持仓的排名
                    position_ranks = []
                    for symbol in self.positions.keys():
                        rank = stock_ranks.get(symbol, 999)
                        position_ranks.append((symbol, rank))
                    
                    # 按排名排序，卖出排名最差的
                    position_ranks.sort(key=lambda x: x[1], reverse=True)
                    excess_count = len(self.positions) - self.config.max_positions
                    
                    for symbol, rank in position_ranks[:excess_count]:
                        if symbol not in symbols_to_sell and symbol in price_map:
                            position = self.positions[symbol]
                            sell_amount = price_map[symbol] * position.quantity
                            cost = self.cost_calculator.calculate_sell_cost(sell_amount)
                            
                            trades_to_execute.append({
                                'symbol': symbol,
                                'action': 'sell',
                                'price': price_map[symbol],
                                'quantity': position.quantity,
                                'amount': sell_amount,
                                'cost': cost,
                            })
                
                # 买入交易 - 只买入目标持仓中当前没有的股票
                for symbol in target_buy:
                    if symbol not in self.positions and symbol in price_map:
                        target_weight = 1.0 / len(target_buy) if target_buy else 0.02
                        target_value = self.portfolio_value * target_weight
                        buy_quantity = int(target_value / price_map[symbol])
                        
                        if buy_quantity > 0:
                            buy_amount = buy_quantity * price_map[symbol]
                            cost = self.cost_calculator.calculate_buy_cost(buy_amount)
                            
                            trades_to_execute.append({
                                'symbol': symbol,
                                'action': 'buy',
                                'price': price_map[symbol],
                                'quantity': buy_quantity,
                                'amount': buy_amount,
                                'cost': cost,
                            })
            
            if trades_to_execute:
                limited_trades, actual_ratio = self.rebalance_limiter.limit_rebalance(
                    trades_to_execute, self.portfolio_value, self.positions, self.config.max_positions
                )
                
                if limited_trades:
                    self.rebalance_dates.append(str(trade_date))
                    logger.info(f"V99: {trade_date} 调仓，实际调仓比例 {actual_ratio:.1%}")
                    
                    # 执行交易
                    for trade in limited_trades:
                        self._execute_single_trade(trade_date, trade)
                        
                        if trade['action'] == 'buy':
                            buy_value += trade['amount']
                            self.total_transaction_cost += trade['cost']
                        else:
                            sell_value += trade['amount']
                            self.total_transaction_cost += trade['cost']
            elif trades_to_execute:
                # 跳过非调仓日的交易，只记录日志
                logger.debug(f"V99: {trade_date} 跳过调仓（非调仓日，计划调仓比例={(buy_value+sell_value)/self.portfolio_value:.1%}）")
            
            # 记录换手率
            self.turnover_tracker.record_turnover(
                trade_date, self.portfolio_value, buy_value, sell_value,
                is_rebalance_day=len(trades_to_execute) > 0,
                rebalance_ratio=(buy_value + sell_value) / self.portfolio_value if self.portfolio_value > EPSILON else 0.0
            )
            
            # 计算日收益率
            if prev_date and self.daily_snapshots:
                prev_value = self.daily_snapshots[-1]['total_value']
                daily_return = (self.portfolio_value - prev_value) / prev_value if prev_value > EPSILON else 0.0
            else:
                daily_return = 0.0
            
            total_position_count += len(self.positions)
            snapshot_count += 1
            
            # 扣费后收益
            net_daily_return = daily_return - (buy_value + sell_value) * V99_TRANSACTION_COST / self.portfolio_value if self.portfolio_value > EPSILON else 0.0
            
            snapshot = {
                'trade_date': str(trade_date),
                'total_value': self.portfolio_value,
                'net_value': self.net_portfolio_value,
                'cash': self.cash,
                'position_value': position_value,
                'position_count': len(self.positions),
                'daily_return': daily_return,
                'net_daily_return': net_daily_return,
                'cumulative_transaction_cost': self.total_transaction_cost,
            }
            self.daily_snapshots.append(snapshot)
            
            prev_date = trade_date
            prev_ranks = stock_ranks.copy()
            
            if (i + 1) % 50 == 0:
                logger.info(f"V99: 处理 {i + 1}/{len(trade_dates)} 天，组合价值={self.portfolio_value:,.2f}, 扣费后={self.net_portfolio_value:,.2f}, 持仓数={len(self.positions)}")
        
        avg_position_count = total_position_count / max(1, snapshot_count)
        
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital
        net_total_return = (self.net_portfolio_value - self.config.initial_capital) / self.config.initial_capital
        
        turnover_summary = self.turnover_tracker.get_turnover_summary()
        max_drawdown = self._calculate_max_drawdown()
        net_max_drawdown = self._calculate_net_max_drawdown()
        
        annual_returns = self._calculate_annual_returns()
        net_annual_returns = self._calculate_net_annual_returns()
        
        result = {
            'total_return': total_return,
            'net_total_return': net_total_return,
            'final_value': self.portfolio_value,
            'net_final_value': self.net_portfolio_value,
            'max_drawdown': max_drawdown,
            'net_max_drawdown': net_max_drawdown,
            'annualized_turnover': turnover_summary['annualized_turnover'],
            'total_trading_days': len(trade_dates),
            'total_trades': len(self.trade_records),
            'rebalance_count': len(self.rebalance_dates),
            'avg_position_count': avg_position_count,
            'annual_returns': annual_returns,
            'net_annual_returns': net_annual_returns,
            'total_transaction_cost': self.total_transaction_cost,
            'turnover_summary': turnover_summary,
        }
        
        logger.info(f"V99: 回测完成 - 总收益={total_return:.2%}, 扣费后净收益={net_total_return:.2%}")
        logger.info(f"V99: 年化换手={turnover_summary['annualized_turnover']:.2%}, 调仓次数={len(self.rebalance_dates)}")
        logger.info(f"V99: 总交易成本={self.total_transaction_cost:,.2f}")
        
        return result
    
    def _calculate_stock_ranks(self, signals: Dict[str, float]) -> Dict[str, int]:
        """计算股票 Alpha 排名"""
        if not signals:
            return {}
        
        sorted_symbols = sorted(signals.keys(), key=lambda s: signals[s], reverse=True)
        return {symbol: rank + 1 for rank, symbol in enumerate(sorted_symbols)}
    
    def _execute_single_trade(self, trade_date: str, trade: Dict[str, Any]) -> None:
        """执行单笔交易"""
        symbol = trade['symbol']
        action = trade['action']
        price = trade['price']
        quantity = trade['quantity']
        amount = trade['amount']
        cost = trade['cost']
        
        if action == 'sell':
            if symbol in self.positions:
                del self.positions[symbol]
            self.cash += amount - cost
        else:  # buy
            if self.cash >= amount + cost:
                self.cash -= amount + cost
                self.positions[symbol] = V99Position(
                    symbol=symbol,
                    entry_date=str(trade_date),
                    entry_price=price,
                    quantity=quantity,
                    weight=0.02,
                    current_price=price,
                    current_rank=999,
                )
        
        self.trade_records.append({
            'trade_date': str(trade_date),
            'symbol': symbol,
            'action': action,
            'price': price,
            'quantity': quantity,
            'amount': amount,
            'fees': cost,
        })
    
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
    
    def _calculate_net_max_drawdown(self) -> float:
        """计算扣费后最大回撤"""
        if not self.daily_snapshots:
            return 0.0
        
        peak = self.config.initial_capital
        max_dd = 0.0
        
        for snapshot in self.daily_snapshots:
            value = snapshot.get('net_value', snapshot['total_value'])
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > EPSILON else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd
    
    def _calculate_annual_returns(self) -> Dict[str, float]:
        """计算各年度收益"""
        annual_returns = {}
        
        for year in self.config.oos_years:
            snapshots_year = [
                s for s in self.daily_snapshots
                if str(s['trade_date']).startswith(year)
            ]
            
            if len(snapshots_year) >= 2:
                start_value = snapshots_year[0]['total_value']
                end_value = snapshots_year[-1]['total_value']
                year_return = (end_value - start_value) / start_value if start_value > EPSILON else 0.0
                annual_returns[year] = year_return
        
        return annual_returns
    
    def _calculate_net_annual_returns(self) -> Dict[str, float]:
        """计算扣费后各年度收益"""
        annual_returns = {}
        
        for year in self.config.oos_years:
            snapshots_year = [
                s for s in self.daily_snapshots
                if str(s['trade_date']).startswith(year)
            ]
            
            if len(snapshots_year) >= 2:
                start_value = snapshots_year[0].get('net_value', snapshots_year[0]['total_value'])
                end_value = snapshots_year[-1].get('net_value', snapshots_year[-1]['total_value'])
                year_return = (end_value - start_value) / start_value if start_value > EPSILON else 0.0
                annual_returns[year] = year_return
        
        return annual_returns
    
    def _generate_audit_report(self, data_integrity: Dict, ic_audit: Dict,
                                trade_results: Dict) -> str:
        """生成审计报告"""
        lines = []
        lines.append("=" * 70)
        lines.append("《V99 净收益保卫战审计报告》")
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
        lines.append(f"   T+1 Rank IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V99_T1_IC_TARGET})")
        lines.append(f"   T+2 Rank IC: {ic_audit.get('ic_t2', 0.0):.4f}")
        lines.append(f"   T+3 Rank IC: {ic_audit.get('ic_t3', 0.0):.4f}")
        lines.append(f"   IC Stability: {ic_audit.get('ic_stability', 0.0):.3f} (目标 > {V99_IC_STABILITY_TARGET})")
        lines.append(f"   IC 衰减正常：{'是' if ic_audit.get('decay_normal') else '否'}")
        lines.append(f"   T+1 IC 达标：{'是' if ic_audit.get('t1_ic_passed') else '否'}")
        lines.append(f"   Stability 达标：{'是' if ic_audit.get('stability_passed') else '否'}")
        lines.append("")
        
        lines.append("3. 消融实验 - 单因子 IC")
        lines.append("   " + "-" * 50)
        for factor_name, ic_data in self.factor_ic_records.items():
            lines.append(f"   {factor_name}:")
            lines.append(f"     T+1 IC: {ic_data.get('ic_t1', 0.0):.4f}")
            lines.append(f"     IC Std(5d): {ic_data.get('ic_std_5d', 0.0):.4f}")
            lines.append(f"     稳定性过滤：{'通过' if ic_data.get('passed_stability') else '未通过'}")
        lines.append("")
        
        lines.append("4. 交易执行审计")
        lines.append("   " + "-" * 50)
        lines.append(f"   总收益：{trade_results.get('total_return', 0.0):.2%}")
        lines.append(f"   扣费后净收益：{trade_results.get('net_total_return', 0.0):.2%}")
        lines.append(f"   最终价值：{trade_results.get('final_value', 0.0):,.2f}")
        lines.append(f"   扣费后最终价值：{trade_results.get('net_final_value', 0.0):,.2f}")
        lines.append(f"   最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
        lines.append(f"   扣费后最大回撤：{trade_results.get('net_max_drawdown', 0.0):.2%}")
        lines.append(f"   年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
        lines.append(f"   调仓次数：{trade_results.get('rebalance_count', 0)}")
        lines.append(f"   平均持仓数：{trade_results.get('avg_position_count', 0):.1f}")
        lines.append(f"   总交易成本：{trade_results.get('total_transaction_cost', 0.0):,.2f}")
        lines.append("")
        
        annual_returns = trade_results.get('annual_returns', {})
        net_annual_returns = trade_results.get('net_annual_returns', {})
        if annual_returns:
            lines.append("   年度收益对比:")
            lines.append("   " + "-" * 40)
            lines.append(f"   {'年度':<8} {'名义收益':>12} {'扣费后收益':>12}")
            lines.append("   " + "-" * 40)
            for year in sorted(annual_returns.keys()):
                gross_ret = annual_returns.get(year, 0.0)
                net_ret = net_annual_returns.get(year, 0.0)
                lines.append(f"   {year:<8} {gross_ret:>12.2%} {net_ret:>12.2%}")
            lines.append("")
        
        lines.append("5. V99 硬性指标验证")
        lines.append("   " + "-" * 50)
        
        # 指标 1: 扣费后年化收益
        net_annual_return = trade_results.get('net_total_return', 0.0)
        # 简化计算：假设回测覆盖 6 年 (2019-2024)
        years_covered = len(self.config.oos_years)
        
        # 修复：处理负收益情况，避免复数计算
        if net_annual_return <= -1.0:
            annualized_net_return = -1.0  # 最大亏损 100%
        else:
            annualized_net_return = (1 + net_annual_return) ** (1.0 / max(1, years_covered)) - 1
        
        metric_return_pass = annualized_net_return > 0.15
        lines.append(f"   指标 1 (扣费后年化收益 > 15%): {'✓' if metric_return_pass else '✗'}")
        lines.append(f"     - 扣费后总收益：{net_annual_return:.2%}")
        lines.append(f"     - 估算年化收益：{annualized_net_return:.2%}")
        lines.append("")
        
        # 指标 2: 年化换手率
        turnover = trade_results.get('annualized_turnover', 0.0)
        turnover_pass = V99_TURNOVER_MIN <= turnover <= 5.0  # 500% 上限
        lines.append(f"   指标 2 (年化换手率 300%-450%, 上限 500%): {'✓' if turnover_pass else '✗'}")
        lines.append(f"     - 年化换手率：{turnover:.2%}")
        lines.append("")
        
        # 指标 3: T+1 Rank IC
        ic_t1 = ic_audit.get('ic_t1', 0.0)
        metric_ic_pass = ic_t1 >= V99_T1_IC_TARGET
        lines.append(f"   指标 3 (T+1 Rank IC > {V99_T1_IC_TARGET}): {'✓' if metric_ic_pass else '✗'}")
        lines.append(f"     - T+1 IC: {ic_t1:.4f}")
        lines.append("")
        
        # 指标 4: IC Stability
        ic_stability = ic_audit.get('ic_stability', 0.0)
        metric_stability_pass = ic_stability >= V99_IC_STABILITY_TARGET
        lines.append(f"   指标 4 (IC Stability > {V99_IC_STABILITY_TARGET}): {'✓' if metric_stability_pass else '✗'}")
        lines.append(f"     - IC Stability: {ic_stability:.3f}")
        lines.append("")
        
        lines.append("=" * 70)
        
        all_passed = metric_return_pass and turnover_pass and metric_ic_pass and metric_stability_pass
        lines.append(f"总体评估：{'所有核心指标通过 ✓' if all_passed else '部分指标未通过 ✗'}")
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
        }


# ===========================================
# 主程序
# ===========================================

def run_v99_backtest(config: V99EngineConfig = None) -> Dict[str, Any]:
    """运行 V99 回测"""
    engine = V99Engine(config=config)
    return engine.run_backtest()


def print_v99_report(result: Dict[str, Any]) -> None:
    """打印 V99 报告"""
    logger.info("=" * 70)
    logger.info("V99 最终报告")
    logger.info("=" * 70)
    
    trade_results = result.get('trade_results', {})
    logger.info("【交易执行】")
    logger.info(f"  总收益：{trade_results.get('total_return', 0.0):.2%}")
    logger.info(f"  扣费后净收益：{trade_results.get('net_total_return', 0.0):.2%}")
    logger.info(f"  最终价值：{trade_results.get('final_value', 0.0):,.2f}")
    logger.info(f"  扣费后最终价值：{trade_results.get('net_final_value', 0.0):,.2f}")
    logger.info(f"  最大回撤：{trade_results.get('max_drawdown', 0.0):.2%}")
    logger.info(f"  扣费后最大回撤：{trade_results.get('net_max_drawdown', 0.0):.2%}")
    logger.info(f"  年化换手率：{trade_results.get('annualized_turnover', 0.0):.2%}")
    logger.info(f"  平均持仓数：{trade_results.get('avg_position_count', 0):.1f}")
    logger.info(f"  调仓次数：{trade_results.get('rebalance_count', 0)}")
    logger.info(f"  总交易成本：{trade_results.get('total_transaction_cost', 0.0):,.2f}")
    
    ic_audit = result.get('ic_audit', {})
    logger.info("")
    logger.info("【IC 审计】")
    logger.info(f"  T+1 IC: {ic_audit.get('ic_t1', 0.0):.4f} (目标 > {V99_T1_IC_TARGET})")
    logger.info(f"  T+2 IC: {ic_audit.get('ic_t2', 0.0):.4f}")
    logger.info(f"  T+3 IC: {ic_audit.get('ic_t3', 0.0):.4f}")
    logger.info(f"  IC Stability: {ic_audit.get('ic_stability', 0.0):.3f} (目标 > {V99_IC_STABILITY_TARGET})")
    logger.info(f"  衰减正常：{'是' if ic_audit.get('decay_normal') else '否'}")
    
    factor_ic = result.get('factor_ic', {})
    if factor_ic:
        logger.info("")
        logger.info("【消融实验 - 单因子 IC】")
        for factor_name, ic_data in factor_ic.items():
            stability = ic_data.get('ic_stability', 'N/A')
            std_5d = ic_data.get('ic_std_5d', 'N/A')
            passed = ic_data.get('passed_stability', False)
            logger.info(f"  {factor_name}: T+1={ic_data.get('ic_t1', 0.0):.4f}, IC Std(5d)={std_5d}, 稳定性={'通过' if passed else '未通过'}")
    
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
    
    # V99 配置 - 严格执行 2% 每日调仓上限
    config = V99EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        buy_rank_threshold=20,
        sell_rank_threshold=60,
        max_daily_rebalance_ratio=0.02,  # 2% 每日调仓上限（严格控制换手率）
    )
    
    result = run_v99_backtest(config)
    print_v99_report(result)
    
    # 保存结果
    output_path = "reports/v99_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    serializable_result = {
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': result.get('ic_audit', {}),
        'factor_ic': result.get('factor_ic', {}),
        'trade_results': result.get('trade_results', {}),
        'audit_report': result.get('audit_report', ''),
        'trade_count': len(result.get('trade_records', [])),
        'snapshot_count': len(result.get('daily_snapshots', [])),
        'rebalance_count': len(result.get('rebalance_dates', [])),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V99: 结果已保存至 {output_path}")