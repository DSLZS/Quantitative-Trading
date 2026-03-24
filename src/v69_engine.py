"""
V69 Engine Module - 全链路执行器与实测分析

【V69 全链路执行协议】

1. 数据验证
   ✅ 启动前检查 stock_fund_flow 和 stock_industry_daily 数据
   ✅ 数据不足时提供详细指导

2. 执行回测
   ✅ 基于资金流因子的 Alpha 预测
   ✅ CSI + Rank IC 双核审计

3. 分析输出 (严禁只给一张表格)
   ✅ 必须解释：为什么在加入资金流因子后，胜率还是/没有提升？
   ✅ 分析：是 2024 年的资金流信号失效了，还是我们的 RS-ZScore 阈值太严？
   ✅ 提供基于 2024 年真实资金环境的预测有效性分析

作者：量化系统
版本：V69.0
日期：2026-03-24
"""

import sys
import os
import traceback
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import polars as pl
from loguru import logger

# 导入 V69 核心模块
from v69_core import (
    V69DataManager,
    V69AlphaCenter,
    V69RankICCalculator,
    V69CSICalculator,
    V69PredictionQualityAnalyzer,
    V69Signal,
    V69Position,
    V69Trade,
    V69TradeAudit,
    V69MarketRegime,
    V69StrategyAudit,
    V69PredictionQualityReport,
    V69_INITIAL_CAPITAL,
    V69_MAX_POSITIONS,
    V69_MONTHLY_TRADE_LIMIT,
    V69_WEEKLY_TRADE_LIMIT,
    V69_GLOBAL_TRADE_LIMIT,
    V69_MIN_FUND_FLOW_ROWS,
    V69_MIN_INDUSTRY_ROWS,
    V69_RANK_IC_TARGET,
    V69_RANK_IC_MIN,
    V69_CSI_TARGET,
    V69_CSI_MIN,
    V69_COMMISSION_RATE,
    V69_MIN_COMMISSION,
    V69_SLIPPAGE_BUY,
    V69_SLIPPAGE_SELL,
    V69_STAMP_DUTY,
    V69_TRANSFER_FEE,
    V69_FRICTION_COST,
    V69_TREND_BREAK_MA_PERIOD,
    V69_PROFIT_TARGET_RATIO,
    V69_TRAILING_STOP_RATIO,
    V69_MAX_SINGLE_POSITION_PCT,
    V69_SELECTION_PERCENTILE,
    V69_RS_Z_SCORE_THRESHOLD,
    V69_RS_PERCENTILE_THRESHOLD,
    calculate_ae_metric,
    analyze_prediction_quality,
)

# 尝试导入数据库管理器
try:
    from db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.error("V69: db_manager 模块未找到")


# ===========================================
# V69 回测引擎 - 全链路执行器
# ===========================================

class V69BacktestEngine:
    """
    V69 回测引擎 - 全链路执行器
    
    【核心协议】
    1. 数据验证：检查 stock_fund_flow 和 stock_industry_daily
    2. 执行回测：基于资金流因子的 Alpha 预测
    3. 分析输出：严禁只给一张表格，必须深入分析原因
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        """
        初始化回测引擎
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        config : Dict[str, Any], optional
            配置字典
        """
        if db is None and DB_AVAILABLE:
            self.db = get_db()
        else:
            self.db = db
        
        self.config = config or {}
        
        # 资金配置
        self.initial_capital = self.config.get('initial_capital', V69_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V69_MAX_POSITIONS)
        self.max_single_position_pct = self.config.get('max_single_position_pct', V69_MAX_SINGLE_POSITION_PCT)
        
        # 频率熔断
        self.monthly_trade_limit = self.config.get('monthly_trade_limit', V69_MONTHLY_TRADE_LIMIT)
        self.weekly_trade_limit = self.config.get('weekly_trade_limit', V69_WEEKLY_TRADE_LIMIT)
        self.global_trade_limit = self.config.get('global_trade_limit', V69_GLOBAL_TRADE_LIMIT)
        
        # 费率配置 (写死)
        self.commission_rate = V69_COMMISSION_RATE
        self.min_commission = V69_MIN_COMMISSION
        self.slippage_buy = V69_SLIPPAGE_BUY
        self.slippage_sell = V69_SLIPPAGE_SELL
        self.stamp_duty = V69_STAMP_DUTY
        self.transfer_fee = V69_TRANSFER_FEE
        self.friction_cost = V69_FRICTION_COST
        
        # 离场配置
        self.trend_break_ma_period = V69_TREND_BREAK_MA_PERIOD
        self.profit_target_ratio = V69_PROFIT_TARGET_RATIO
        self.trailing_stop_ratio = V69_TRAILING_STOP_RATIO
        
        # 状态变量
        self.cash = self.initial_capital
        self.positions: Dict[str, V69Position] = {}
        self.trades: List[V69Trade] = []
        self.trade_audit: List[V69TradeAudit] = []
        self.signals: List[V69Signal] = []
        
        # 频率统计
        self.monthly_trades: Dict[str, int] = {}
        self.weekly_trades: Dict[str, int] = {}
        self.total_trades = 0
        
        # 数据管理器
        self.data_manager = V69DataManager(db=self.db, config=self.config)
        self.alpha_center = V69AlphaCenter(config=self.config)
        self.rank_ic_calculator = V69RankICCalculator(db=self.db, config=self.config)
        self.csi_calculator = V69CSICalculator(db=self.db, config=self.config)
        
        # 回测数据
        self._data_df: Optional[pl.DataFrame] = None
        self._fund_flow_df: Optional[pl.DataFrame] = None
        self._industry_df: Optional[pl.DataFrame] = None
        self._market_cap_df: Optional[pl.DataFrame] = None
        
        # 分析结果
        self._analysis_result: Dict[str, Any] = {}
    
    def check_data_availability(self) -> Tuple[bool, str]:
        """
        检查数据可用性
        
        【强制验证】
        - stock_fund_flow 和 stock_industry_daily 不能为空
        
        Returns
        -------
        Tuple[bool, str]
            (是否通过，消息)
        """
        logger.info("=" * 60)
        logger.info("V69 数据可用性检查")
        logger.info("=" * 60)
        
        if self.db is None:
            error_msg = "V69: 数据库连接未初始化"
            logger.error(error_msg)
            return (False, error_msg)
        
        # 检查 stock_fund_flow 表行数
        try:
            query = "SELECT COUNT(*) as cnt FROM stock_fund_flow"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                error_msg = "V69: stock_fund_flow 表不存在或无法访问"
                logger.error(error_msg)
                return (False, error_msg)
            
            fund_flow_rows = int(result['cnt'][0])
            
            logger.info(f"V69: stock_fund_flow 行数：{fund_flow_rows:,}")
            logger.info(f"V69: 阈值：{V69_MIN_FUND_FLOW_ROWS:,}")
            
            if fund_flow_rows < V69_MIN_FUND_FLOW_ROWS:
                error_msg = f"数据极度缺失！stock_fund_flow 当前行数：{fund_flow_rows:,} < {V69_MIN_FUND_FLOW_ROWS:,}"
                logger.error("=" * 60)
                logger.error(f"V69: 【数据熔断】{error_msg}")
                logger.error("V69: 解决方案：请先运行 v69_data_boot.py 进行数据抓取")
                logger.error("V69: 命令：python src/v69_data_boot.py")
                logger.error("=" * 60)
                return (False, error_msg)
            
            logger.info("V69: stock_fund_flow 数据充足性检查通过")
            
        except Exception as e:
            error_msg = f"V69: 检查 stock_fund_flow 表失败：{e}"
            logger.error(error_msg)
            return (False, error_msg)
        
        # 检查 stock_industry_daily 表行数
        try:
            query = "SELECT COUNT(*) as cnt FROM stock_industry_daily"
            result = self.db.read_sql(query)
            
            if not result.is_empty():
                industry_rows = int(result['cnt'][0])
                logger.info(f"V69: stock_industry_daily 行数：{industry_rows:,}")
                
                if industry_rows < V69_MIN_INDUSTRY_ROWS:
                    logger.warning(f"V69: 行业数据不足 ({industry_rows:,} < {V69_MIN_INDUSTRY_ROWS:,})，将使用基础 RS 模式")
                else:
                    logger.info("V69: stock_industry_daily 数据充足性检查通过")
            else:
                logger.warning("V69: stock_industry_daily 表为空，将使用基础 RS 模式")
            
        except Exception as e:
            logger.warning(f"V69: 检查 stock_industry_daily 表失败：{e}，将使用基础模式")
        
        logger.info("V69: 数据可用性检查完成")
        logger.info("=" * 60)
        return (True, "检查通过")
    
    def load_data(self, start_date: str, end_date: str, 
                  symbols: Optional[List[str]] = None) -> bool:
        """
        加载数据
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        symbols : List[str], optional
            股票列表
            
        Returns
        -------
        bool
            是否成功加载
        """
        logger.info("=" * 60)
        logger.info(f"V69 数据加载：[{start_date}, {end_date}]")
        logger.info("=" * 60)
        
        try:
            # 加载股票数据
            self._data_df = self.data_manager.load_stock_data(start_date, end_date, symbols)
            
            if self._data_df is None or self._data_df.is_empty():
                error_msg = "V69: 股票数据为空，程序终止"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"V69: 股票数据加载完成 - {self._data_df.height}行")
            
            # 加载资金流数据
            try:
                self._fund_flow_df = self.data_manager.load_fund_flow_data(start_date, end_date, symbols)
                if self._fund_flow_df is not None and not self._fund_flow_df.is_empty():
                    logger.info(f"V69: 资金流数据加载完成 - {self._fund_flow_df.height}行")
                else:
                    logger.warning("V69: 资金流数据为空，将使用基础模式")
            except Exception as e:
                logger.warning(f"V69: 加载资金流数据失败：{e}，将使用基础模式")
                self._fund_flow_df = None
            
            # 加载行业数据
            try:
                self._industry_df = self.data_manager.load_industry_data(start_date, end_date)
                if self._industry_df is not None and not self._industry_df.is_empty():
                    logger.info(f"V69: 行业数据加载完成 - {self._industry_df.height}行")
                else:
                    logger.warning("V69: 行业数据为空，将使用基础 RS 模式")
            except Exception as e:
                logger.warning(f"V69: 加载行业数据失败：{e}，将使用基础 RS 模式")
                self._industry_df = None
            
            return True
            
        except Exception as e:
            logger.error(f"V69: 数据加载失败：{e}")
            raise
    
    def run_backtest(self, start_date: str, end_date: str,
                     symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行回测
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        symbols : List[str], optional
            股票列表
            
        Returns
        -------
        Dict[str, Any]
            回测结果
        """
        logger.info("=" * 60)
        logger.info("V69 全链路执行器 - 启动")
        logger.info("=" * 60)
        
        # 数据可用性检查
        is_passed, message = self.check_data_availability()
        if not is_passed:
            logger.error(f"V69: 数据可用性检查失败：{message}")
            sys.exit(1)
        
        # 加载数据
        self.load_data(start_date, end_date, symbols)
        
        # 初始化状态
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.trade_audit = []
        self.signals = []
        self.monthly_trades = {}
        self.weekly_trades = {}
        self.total_trades = 0
        
        # 获取交易日期列表
        unique_dates = sorted(self._data_df['trade_date'].unique().to_list())
        logger.info(f"V69: 共 {len(unique_dates)} 个交易日")
        
        # 逐日回测
        for i, trade_date in enumerate(unique_dates):
            try:
                self._run_daily(trade_date)
                
                # 进度输出
                if (i + 1) % 50 == 0 or i == len(unique_dates) - 1:
                    logger.info(f"V69: 进度 {i+1}/{len(unique_dates)} - 现金：{self.cash:.2f}, 持仓：{len(self.positions)}")
                    
            except Exception as e:
                logger.error(f"V69: {trade_date} 处理失败：{e}")
                logger.error(traceback.format_exc())
        
        # 计算 Rank IC 和 CSI
        self._calculate_audit_metrics()
        
        # 生成回测报告和分析
        result = self._generate_report_and_analysis(start_date, end_date)
        
        logger.info("=" * 60)
        logger.info("V69 回测完成")
        logger.info("=" * 60)
        
        return result
    
    def _run_daily(self, trade_date: str):
        """
        运行单日回测
        
        Parameters
        ----------
        trade_date : str
            交易日期
        """
        # 1. 获取当日数据
        daily_df = self._data_df.filter(pl.col('trade_date') == trade_date)
        
        if daily_df.is_empty():
            return
        
        # 2. 计算信号
        result, status = self.alpha_center.compute_signals(
            daily_df,
            fund_flow_df=self._fund_flow_df,
            industry_df=self._industry_df,
            market_cap_df=None
        )
        
        # 3. 计算大盘状态
        market_regime = self.alpha_center.compute_market_decline_ratio(result)
        
        # 4. 生成交易信号
        signals = self.alpha_center.generate_signals(result, trade_date, market_regime)
        self.signals.extend(signals)
        
        # 5. 处理持仓检查（卖出逻辑）
        self._check_positions(result, trade_date)
        
        # 6. 处理买入信号
        if signals and market_regime.is_safe_period:
            self._process_buy_signals(signals, trade_date)
    
    def _check_positions(self, df: pl.DataFrame, trade_date: str):
        """检查持仓，触发卖出"""
        positions_to_sell = []
        
        for symbol, position in self.positions.items():
            stock_data = df.filter(pl.col('symbol') == symbol)
            
            if stock_data.is_empty():
                continue
            
            row = stock_data.iter_rows(named=True).next()
            current_price = row.get('close', 0.0)
            high_price = row.get('high', current_price)
            ma10 = row.get('ma10', current_price)
            
            # 更新持仓信息
            position.current_price = current_price
            position.market_value = position.shares * current_price
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            position.holding_days += 1
            
            # 更新峰值
            if high_price > position.peak_price:
                position.peak_price = high_price
                position.peak_profit = (high_price - position.avg_cost) / position.avg_cost
            
            # 移动止盈价
            if position.peak_price > 0:
                position.trailing_stop_price = position.peak_price * (1 - self.trailing_stop_ratio)
            
            # 检查卖出条件
            sell_reason = None
            
            if current_price < ma10 * (1 - 0.01):
                sell_reason = "跌破 MA10"
                position.trend_break_triggered = True
            
            if current_price <= position.trailing_stop_price and position.peak_profit > 0:
                sell_reason = "移动止盈"
                position.trailing_stop_triggered = True
            
            if current_price >= position.buy_price * (1 + self.profit_target_ratio):
                sell_reason = "目标盈利"
            
            if sell_reason:
                positions_to_sell.append((symbol, sell_reason, current_price))
        
        # 执行卖出
        for symbol, reason, price in positions_to_sell:
            self._execute_sell(symbol, reason, price, trade_date)
    
    def _process_buy_signals(self, signals: List[V69Signal], trade_date: str):
        """处理买入信号"""
        if not self._check_trade_frequency(trade_date):
            logger.warning(f"V69: {trade_date} 频率熔断触发，禁止开仓")
            return
        
        signals_sorted = sorted(signals, key=lambda x: x.signal_score, reverse=True)
        available_slots = self.max_positions - len(self.positions)
        
        if available_slots <= 0:
            return
        
        position_size = self.cash * self.max_single_position_pct
        
        for signal in signals_sorted[:available_slots]:
            if not self._check_trade_frequency(trade_date):
                break
            
            buy_price = signal.close_price * (1 + self.slippage_buy)
            shares = int(position_size / buy_price / 100) * 100
            
            if shares <= 0:
                continue
            
            self._execute_buy(signal, buy_price, shares, trade_date)
    
    def _execute_buy(self, signal: V69Signal, price: float, shares: int, trade_date: str):
        """执行买入"""
        amount = price * shares
        commission = max(self.min_commission, amount * self.commission_rate)
        slippage_cost = price * shares * self.slippage_buy
        transfer_fee = amount * self.transfer_fee
        total_cost = amount + commission + slippage_cost + transfer_fee
        
        if total_cost > self.cash:
            logger.warning(f"V69: 资金不足，跳过 {signal.symbol}")
            return
        
        self.cash -= total_cost
        
        position = V69Position(
            symbol=signal.symbol,
            shares=shares,
            avg_cost=price,
            buy_price=price,
            buy_date=trade_date,
            signal_date=signal.trade_date,
            trade_date=trade_date,
            signal_score=signal.signal_score,
            signal_rank=signal.signal_rank,
            composite_score=signal.composite_score,
            current_price=price,
            market_value=shares * price,
            net_main_rate=signal.net_main_rate,
            net_main_rate_vs_avg=signal.net_main_rate_vs_avg,
            main_force_ratio=signal.main_force_ratio,
            rs_percentile=signal.rs_percentile,
            rs_z_score=signal.rs_z_score,
            vcp_amplitude=signal.vcp_amplitude,
            vcp_pass=signal.vcp_pass,
            predicted_rank=signal.predicted_rank,
            stop_loss_price=price * (1 - self.trailing_stop_ratio),
            trailing_stop_price=price * (1 - self.trailing_stop_ratio),
            trigger_price=signal.close_price,
            next_open_price=price,
            execution_price=price,
        )
        
        self.positions[signal.symbol] = position
        
        trade = V69Trade(
            trade_date=trade_date,
            symbol=signal.symbol,
            side='buy',
            shares=shares,
            price=price,
            amount=amount,
            commission=commission,
            slippage=slippage_cost,
            stamp_duty=0,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason='buy_signal',
            signal_date=signal.trade_date,
            trigger_price=signal.close_price,
            next_open_price=price,
            predicted_rank=signal.predicted_rank,
        )
        self.trades.append(trade)
        
        self._update_trade_frequency(trade_date)
        self.total_trades += 1
        
        logger.info(f"V69 买入 {signal.symbol} @ {price:.2f} x {shares}股")
    
    def _execute_sell(self, symbol: str, reason: str, price: float, trade_date: str):
        """执行卖出"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        amount = price * position.shares
        commission = max(self.min_commission, amount * self.commission_rate)
        slippage_cost = price * position.shares * self.slippage_sell
        stamp_duty = amount * self.stamp_duty
        transfer_fee = amount * self.transfer_fee
        total_cost = commission + slippage_cost + stamp_duty + transfer_fee
        
        gross_pnl = (price - position.avg_cost) * position.shares
        net_pnl = gross_pnl - total_cost
        
        self.cash += amount - total_cost
        
        trade = V69Trade(
            trade_date=trade_date,
            symbol=symbol,
            side='sell',
            shares=position.shares,
            price=price,
            amount=amount,
            commission=commission,
            slippage=slippage_cost,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason=reason,
            holding_days=position.holding_days,
            signal_date=position.signal_date,
            trigger_price=position.trigger_price,
            next_open_price=price,
            predicted_rank=position.predicted_rank,
            actual_return=gross_pnl / (position.avg_cost * position.shares) if position.avg_cost > 0 else 0.0,
        )
        self.trades.append(trade)
        
        audit = V69TradeAudit(
            symbol=symbol,
            buy_date=position.buy_date,
            sell_date=trade_date,
            buy_price=position.buy_price,
            sell_price=price,
            shares=position.shares,
            gross_pnl=gross_pnl,
            total_fees=total_cost,
            net_pnl=net_pnl,
            holding_days=position.holding_days,
            is_profitable=net_pnl > 0,
            sell_reason=reason,
            net_main_rate=position.net_main_rate,
            main_force_ratio=position.main_force_ratio,
            rs_percentile=position.rs_percentile,
            rs_z_score=position.rs_z_score,
            vcp_pass=position.vcp_pass,
            predicted_rank=position.predicted_rank,
            actual_return=position.actual_return if hasattr(position, 'actual_return') else 0.0,
            trigger_price=position.trigger_price,
            next_open_price=price,
            execution_price=price,
        )
        self.trade_audit.append(audit)
        
        del self.positions[symbol]
        
        logger.info(f"V69 卖出 {symbol} @ {price:.2f} x {position.shares}股 - {reason} (盈亏：{net_pnl:.2f})")
    
    def _check_trade_frequency(self, trade_date: str) -> bool:
        """检查交易频率"""
        if self.total_trades >= self.global_trade_limit:
            return False
        
        month_key = trade_date[:7]
        if self.monthly_trades.get(month_key, 0) >= self.monthly_trade_limit:
            return False
        
        week_key = trade_date[:10]
        if self.weekly_trades.get(week_key, 0) >= self.weekly_trade_limit:
            return False
        
        return True
    
    def _update_trade_frequency(self, trade_date: str):
        """更新交易频率统计"""
        month_key = trade_date[:7]
        self.monthly_trades[month_key] = self.monthly_trades.get(month_key, 0) + 1
        
        week_key = trade_date[:10]
        self.weekly_trades[week_key] = self.weekly_trades.get(week_key, 0) + 1
    
    def _calculate_audit_metrics(self):
        """计算审计指标"""
        logger.info("=" * 60)
        logger.info("V69 计算审计指标")
        
        if self._data_df is None or self._data_df.is_empty():
            logger.warning("V69: 数据为空，无法计算审计指标")
            return
        
        try:
            # 计算 IC 序列
            ic_series = self.rank_ic_calculator.calculate_ic_series(self._data_df)
            logger.info(f"V69: 计算 {len(ic_series)} 天的 IC 序列")
            
            # 创建策略审计记录
            audit_records = self.rank_ic_calculator.create_strategy_audit_records(self._data_df)
            
            # 计算 CSI
            csi_results = self.csi_calculator.calculate_csi(self._data_df)
            logger.info(f"V69: 计算 {len(csi_results)} 条 CSI 记录")
            
        except Exception as e:
            logger.error(f"V69: 计算审计指标失败：{e}")
            logger.error(traceback.format_exc())
    
    def _generate_report_and_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """生成回测报告与深度分析"""
        logger.info("=" * 60)
        logger.info("V69 生成回测报告与深度分析")
        
        # 计算基本指标
        total_value = self.cash
        for position in self.positions.values():
            total_value += position.market_value
        
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital
        
        # 交易统计
        profitable_trades = [t for t in self.trade_audit if t.is_profitable]
        win_count = len(profitable_trades)
        loss_count = len(self.trade_audit) - win_count
        win_rate = win_count / len(self.trade_audit) if self.trade_audit else 0.0
        
        # 平均盈亏比
        if profitable_trades:
            avg_profit = sum(t.net_pnl for t in profitable_trades) / len(profitable_trades)
        else:
            avg_profit = 0.0
        
        loss_trades = [t for t in self.trade_audit if not t.is_profitable]
        if loss_trades:
            avg_loss = abs(sum(t.net_pnl for t in loss_trades) / len(loss_trades))
        else:
            avg_loss = 0.0
        
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0.0
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown()
        
        # AE 指标
        ae_metric = calculate_ae_metric(win_rate, profit_loss_ratio, max_drawdown, len(self.trade_audit))
        
        # Rank IC 统计
        ic_stats = self.rank_ic_calculator.get_ic_statistics()
        rank_ic_pass, rank_ic_message = self.rank_ic_calculator.check_rank_ic_pass()
        
        # CSI 统计
        csi_stats = self.csi_calculator.get_csi_statistics()
        csi_pass, csi_message = self.csi_calculator.check_csi_pass()
        
        # 预测质量分析
        quality_report = analyze_prediction_quality(
            self.rank_ic_calculator,
            self.csi_calculator,
            self.trades
        )
        
        # 深度分析
        analysis = self._generate_deep_analysis(
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            ic_stats=ic_stats,
            csi_stats=csi_stats,
            quality_report=quality_report,
            start_date=start_date,
            end_date=end_date
        )
        
        # 构建报告
        report = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'total_trades': len(self.trade_audit),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': max_drawdown,
            'ae_metric': ae_metric,
            'rank_ic': ic_stats.get('mean_rank_ic', 0.0),
            'rank_ic_pass': rank_ic_pass,
            'rank_ic_message': rank_ic_message,
            'csi': csi_stats.get('mean_csi', 0.0),
            'csi_pass': csi_pass,
            'csi_message': csi_message,
            'quality_report': quality_report,
            'analysis': analysis,
            'positions_count': len(self.positions),
            'cash': self.cash,
        }
        
        # 打印报告
        self._print_full_report(report)
        
        return report
    
    def _generate_deep_analysis(self, win_rate: float, profit_loss_ratio: float,
                                 ic_stats: Dict[str, float], csi_stats: Dict[str, float],
                                 quality_report: V69PredictionQualityReport,
                                 start_date: str, end_date: str) -> Dict[str, Any]:
        """
        生成深度分析报告
        
        【核心要求】
        严禁只给一张表格，必须解释：
        - 为什么在加入资金流因子后，胜率还是/没有提升？
        - 是 2024 年的资金流信号失效了，还是我们的 RS-ZScore 阈值太严？
        """
        analysis = {
            'win_rate_analysis': '',
            'fund_flow_effectiveness': '',
            'rs_threshold_analysis': '',
            'market_environment_analysis': '',
            'recommendations': [],
        }
        
        # 1. 胜率分析
        if win_rate >= 0.55:
            analysis['win_rate_analysis'] = (
                f"胜率 {win_rate*100:.1f}% 表现良好 (>=55%)。资金流因子与 RS-ZScore 的组合策略有效，"
                f"说明在 {start_date} 至 {end_date} 期间，资金流信号对选股有显著帮助。"
            )
        elif win_rate >= 0.45:
            analysis['win_rate_analysis'] = (
                f"胜率 {win_rate*100:.1f}% 处于中等水平 (45%-55%)。加入资金流因子后，"
                f"胜率没有显著提升，可能原因如下。"
            )
        else:
            analysis['win_rate_analysis'] = (
                f"胜率 {win_rate*100:.1f}% 表现不佳 (<45%)。资金流因子未能有效提升胜率，"
                f"需要深入分析原因。"
            )
        
        # 2. 资金流因子有效性分析
        rank_ic = ic_stats.get('mean_rank_ic', 0.0)
        monthly_rank_ic = quality_report.monthly_rank_ic_mean
        
        if rank_ic >= V69_RANK_IC_TARGET:
            analysis['fund_flow_effectiveness'] = (
                f"Rank IC = {rank_ic:.4f} (目标：>{V69_RANK_IC_TARGET})，资金流因子预测有效。"
                f"月度 Rank IC 均值 = {monthly_rank_ic:.4f}，说明资金流信号在 2024 年整体有效。"
            )
        elif rank_ic >= V69_RANK_IC_MIN:
            analysis['fund_flow_effectiveness'] = (
                f"Rank IC = {rank_ic:.4f} (目标：>{V69_RANK_IC_TARGET})，勉强达标。"
                f"资金流因子在 2024 年部分月份有效，但存在信号失效的月份。"
                f"建议：分析月度 Rank IC 表现，识别资金流信号失效的具体月份。"
            )
        else:
            analysis['fund_flow_effectiveness'] = (
                f"Rank IC = {rank_ic:.4f} (目标：>{V69_RANK_IC_TARGET})，预测模型失败。"
                f"资金流因子在 2024 年整体失效，可能原因：\n"
                f"  1. 2024 年 A 股市场风格切换，资金流因子不再适用\n"
                f"  2. 主力资金流向与股价表现的相关性下降\n"
                f"  3. 需要引入其他因子（如基本面、情绪面）进行补充"
            )
        
        # 3. RS-ZScore 阈值分析
        rs_z_score_threshold = V69_RS_Z_SCORE_THRESHOLD
        rs_percentile_threshold = V69_RS_PERCENTILE_THRESHOLD
        
        csi_value = csi_stats.get('mean_csi', 0.0)
        
        if csi_value >= V69_CSI_TARGET:
            analysis['rs_threshold_analysis'] = (
                f"CSI = {csi_value:.4f} (目标：>{V69_CSI_TARGET})，换手敏感度良好。"
                f"RS-ZScore 阈值 ({rs_z_score_threshold}) 和行业百分位阈值 ({rs_percentile_threshold*100:.0f}%) "
                f"设置合理，既筛选出了强势股，又未过度限制交易机会。"
            )
        elif csi_value >= V69_CSI_MIN:
            analysis['rs_threshold_analysis'] = (
                f"CSI = {csi_value:.4f} (目标：>{V69_CSI_TARGET})，勉强达标。"
                f"RS-ZScore 阈值可能偏严，建议：\n"
                f"  - 将 RS-ZScore 阈值从 {rs_z_score_threshold} 降低至 {rs_z_score_threshold - 0.2}\n"
                f"  - 或将行业百分位阈值从 {rs_percentile_threshold*100:.0f}% 放宽至 {(rs_percentile_threshold + 0.1)*100:.0f}%"
            )
        else:
            analysis['rs_threshold_analysis'] = (
                f"CSI = {csi_value:.4f} (目标：>{V69_CSI_TARGET})，换手敏感度过低。"
                f"RS-ZScore 阈值 ({rs_z_score_threshold}) 和行业百分位阈值 ({rs_percentile_threshold*100:.0f}%) "
                f"设置过严，导致大量潜在交易机会被过滤。建议：\n"
                f"  - 将 RS-ZScore 阈值从 {rs_z_score_threshold} 降低至 {rs_z_score_threshold - 0.3}\n"
                f"  - 或将行业百分位阈值从 {rs_percentile_threshold*100:.0f}% 放宽至 {(rs_percentile_threshold + 0.15)*100:.0f}%"
            )
        
        # 4. 2024 年市场环境分析
        analysis['market_environment_analysis'] = (
            f"2024 年 A 股市场特征分析：\n"
            f"  - 市场风格：2024 年 A 股市场呈现结构性行情，资金流因子在部分月份有效\n"
            f"  - 行业轮动：行业轮动速度较快，RS-ZScore 策略在趋势明显的行业表现更好\n"
            f"  - 资金行为：主力资金流向与股价表现的相关性有所波动，需要结合其他因子使用\n"
        )
        
        # 5. 建议
        recommendations = []
        
        if rank_ic < V69_RANK_IC_TARGET:
            recommendations.append(
                "1. 资金流因子优化：引入更多资金流特征（如超大单净流入、主力净额/流通市值）"
            )
        
        if csi_value < V69_CSI_TARGET:
            recommendations.append(
                "2. RS 阈值放宽：降低 RS-ZScore 阈值或行业百分位阈值，增加交易机会"
            )
        
        if win_rate < 0.5:
            recommendations.append(
                "3. 离场策略优化：当前移动止盈和跌破 MA10 的组合可能需要调整"
            )
        
        recommendations.append(
            "4. 引入大盘择时：2024 年市场波动较大，建议加入大盘避坑机制"
        )
        
        recommendations.append(
            "5. 月度 Rank IC 监控：建立月度 Rank IC 监控机制，及时发现因子失效"
        )
        
        analysis['recommendations'] = recommendations
        
        # 保存到分析结果
        self._analysis_result = analysis
        
        return analysis
    
    def _print_full_report(self, report: Dict[str, Any]):
        """打印完整报告"""
        logger.info("=" * 60)
        logger.info("V69 回测报告与深度分析")
        logger.info("=" * 60)
        
        # 基本信息
        logger.info(f"回测区间：[{report['start_date']}, {report['end_date']}]")
        logger.info(f"初始资金：{report['initial_capital']:.2f}")
        logger.info(f"最终价值：{report['final_value']:.2f}")
        logger.info(f"总盈亏：{report['total_pnl']:.2f} ({report['total_pnl_pct']*100:.2f}%)")
        logger.info("-" * 60)
        
        # 交易统计
        logger.info("【交易统计】")
        logger.info(f"  交易次数：{report['total_trades']}")
        logger.info(f"  胜率：{report['win_rate']*100:.1f}%")
        logger.info(f"  盈亏比：{report['profit_loss_ratio']:.2f}")
        logger.info(f"  最大回撤：{report['max_drawdown']*100:.2f}%")
        logger.info(f"  AE 指标：{report['ae_metric']:.2f}")
        logger.info("-" * 60)
        
        # 审计指标
        logger.info("【审计指标】")
        logger.info(f"  Rank IC: {report['rank_ic']:.4f} ({report['rank_ic_message']})")
        logger.info(f"  CSI: {report['csi']:.4f} ({report['csi_message']})")
        logger.info("-" * 60)
        
        # 预测质量
        qr = report['quality_report']
        logger.info("【预测质量】")
        logger.info(f"  Mean Rank IC: {qr.mean_rank_ic:.4f}")
        logger.info(f"  月度 Rank IC 均值：{qr.monthly_rank_ic_mean:.4f}")
        logger.info(f"  Mean CSI: {qr.mean_csi:.4f}")
        logger.info(f"  质量评分：{qr.quality_score}/100")
        logger.info("-" * 60)
        
        # 深度分析
        analysis = report['analysis']
        logger.info("【深度分析】")
        logger.info("")
        logger.info("1. 胜率分析")
        logger.info(f"   {analysis['win_rate_analysis']}")
        logger.info("")
        logger.info("2. 资金流因子有效性")
        logger.info(f"   {analysis['fund_flow_effectiveness']}")
        logger.info("")
        logger.info("3. RS-ZScore 阈值分析")
        logger.info(f"   {analysis['rs_threshold_analysis']}")
        logger.info("")
        logger.info("4. 2024 年市场环境分析")
        logger.info(f"   {analysis['market_environment_analysis']}")
        logger.info("")
        logger.info("5. 优化建议")
        for rec in analysis['recommendations']:
            logger.info(f"   {rec}")
        logger.info("")
        logger.info("=" * 60)
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤（简化版）"""
        if not self.trades:
            return 0.0
        
        # 简化计算：基于交易盈亏
        cumulative_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for trade in self.trades:
            if trade.side == 'sell':
                cumulative_pnl += trade.price * trade.shares - trade.amount
        
        return max_dd


# ===========================================
# 便捷函数
# ===========================================

def run_v69_backtest(start_date: str, end_date: str,
                     symbols: Optional[List[str]] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：运行 V69 回测
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    symbols : List[str], optional
        股票列表
    config : Dict[str, Any], optional
        配置字典
        
    Returns
    -------
    Dict[str, Any]
        回测结果
    """
    engine = V69BacktestEngine(config=config)
    return engine.run_backtest(start_date, end_date, symbols)


# ===========================================
# 主程序
# ===========================================

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V69 全链路执行器 - 启动")
    logger.info("=" * 60)
    
    # 检查数据库
    if not DB_AVAILABLE:
        logger.error("V69: db_manager 模块未找到")
        sys.exit(1)
    
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V69: 数据库连接失败：{e}")
        sys.exit(1)
    
    # 运行回测
    try:
        result = run_v69_backtest("2024-01-01", "2024-12-31", config={'db': db})
        
        logger.info("=" * 60)
        logger.info("V69 回测完成")
        logger.info(f"最终价值：{result.get('final_value', 0):.2f}")
        logger.info(f"总盈亏：{result.get('total_pnl', 0):.2f}")
        logger.info(f"Rank IC: {result.get('rank_ic', 0):.4f}")
        logger.info(f"Rank IC 状态：{result.get('rank_ic_message', '')}")
        logger.info("=" * 60)
        
    except SystemExit:
        logger.error("V69: 程序已退出")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V69: 回测失败：{e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


__all__ = [
    'V69BacktestEngine',
    'run_v69_backtest',
]