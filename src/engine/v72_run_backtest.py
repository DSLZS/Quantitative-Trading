"""
V72 Engine Module - SNR 审计 + 行业背离滤网回测引擎

【V72 全链路执行协议】

1. 数据验证
   ✅ 启动前检查 stock_fund_flow 和 stock_industry_daily 数据
   ✅ 数据不足时提供详细指导

2. 执行回测
   ✅ SNR (信噪比) 过滤：Z-Score > 2.0 且 SNR > 0.15
   ✅ 行业背离滤网：个股主力为正但行业连续 3 日净流出则剔除
   ✅ 市场宽度规避：行业净流出占比 > 70% 强制空仓

3. 分析输出
   ✅ 月度 Rank IC > 0.02（输出每个月的 IC 变化曲线）
   ✅ 最大回撤控制：对比 V64 观察回撤是否因行业滤网而收敛

作者：量化系统
版本：V72.0
日期：2026-03-25
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

# 导入 V72 核心模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.v72_logic import (
    V72DataManager,
    V72AlphaCenter,
    V72RankICCalculator,
    V72PredictionQualityAnalyzer,
    V72Signal,
    V72Position,
    V72Trade,
    V72TradeAudit,
    V72MarketRegime,
    V72PredictionQualityReport,
    V72_INITIAL_CAPITAL,
    V72_MAX_POSITIONS,
    V72_Z_SCORE_THRESHOLD,
    V72_SNR_THRESHOLD,
    V72_INDUSTRY_NET_OUTFLOW_DAYS,
    V72_MARKET_WIDTH_THRESHOLD,
    V72_COMMISSION_RATE,
    V72_MIN_COMMISSION,
    V72_SLIPPAGE_BUY,
    V72_SLIPPAGE_SELL,
    V72_STAMP_DUTY,
    V72_TRANSFER_FEE,
    V72_STOP_LOSS_RATIO,
    V72_PROFIT_TARGET_RATIO,
    V72_TRAILING_STOP_RATIO,
    V72_MAX_SINGLE_POSITION_PCT,
    V72_SELECTION_PERCENTILE,
    analyze_prediction_quality,
)

# 尝试导入数据库管理器
try:
    from db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.error("V72: db_manager 模块未找到")


# ===========================================
# V72 回测引擎 - 全链路执行器
# ===========================================

class V72BacktestEngine:
    """
    V72 回测引擎 - 全链路执行器
    
    【核心协议】
    1. 数据验证：检查 stock_fund_flow 和 stock_industry_daily
    2. 执行回测：SNR 审计 + 行业背离滤网
    3. 分析输出：月度 Rank IC 曲线 + 回撤对比
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
        
        # 资金配置 - 强制设定为 100,000.00 元
        self.initial_capital = self.config.get('initial_capital', V72_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V72_MAX_POSITIONS)
        self.max_single_position_pct = self.config.get('max_single_position_pct', V72_MAX_SINGLE_POSITION_PCT)
        
        # 费率配置 (写死)
        self.commission_rate = V72_COMMISSION_RATE
        self.min_commission = V72_MIN_COMMISSION
        self.slippage_buy = V72_SLIPPAGE_BUY
        self.slippage_sell = V72_SLIPPAGE_SELL
        self.stamp_duty = V72_STAMP_DUTY
        self.transfer_fee = V72_TRANSFER_FEE
        
        # 离场配置 - 止损 5%，止盈 15%
        self.stop_loss_ratio = V72_STOP_LOSS_RATIO
        self.profit_target_ratio = V72_PROFIT_TARGET_RATIO
        self.trailing_stop_ratio = V72_TRAILING_STOP_RATIO
        
        # 状态变量
        self.cash = self.initial_capital
        self.positions: Dict[str, V72Position] = {}
        self.trades: List[V72Trade] = []
        self.trade_audit: List[V72TradeAudit] = []
        self.signals: List[V72Signal] = []
        
        # 数据管理器
        self.data_manager = V72DataManager(db=self.db, config=self.config)
        self.alpha_center = V72AlphaCenter(config=self.config)
        self.rank_ic_calculator = V72RankICCalculator(db=self.db, config=self.config)
        
        # 回测数据
        self._data_df: Optional[pl.DataFrame] = None
        self._fund_flow_df: Optional[pl.DataFrame] = None
        self._industry_df: Optional[pl.DataFrame] = None
        
        # 分析结果
        self._analysis_result: Dict[str, Any] = {}
        self._equity_curve: List[Tuple[str, float]] = []
        self._monthly_returns: Dict[str, float] = {}
    
    def check_data_availability(self) -> Tuple[bool, str]:
        """检查数据可用性"""
        logger.info("=" * 60)
        logger.info("V72 数据可用性检查")
        logger.info("=" * 60)
        
        if self.db is None:
            error_msg = "V72: 数据库连接未初始化"
            logger.error(error_msg)
            return (False, error_msg)
        
        # 检查 stock_fund_flow 表
        try:
            query = "SELECT COUNT(*) as cnt FROM stock_fund_flow"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                error_msg = "V72: stock_fund_flow 表不存在或无法访问"
                logger.error(error_msg)
                return (False, error_msg)
            
            fund_flow_rows = int(result['cnt'][0])
            logger.info(f"V72: stock_fund_flow 行数：{fund_flow_rows:,}")
            
            if fund_flow_rows < 100000:
                logger.warning(f"V72: stock_fund_flow 行数不足 ({fund_flow_rows:,})")
            
            # 检查 net_main_rate 是否有数据
            query = "SELECT COUNT(*) as cnt FROM stock_fund_flow WHERE net_main_rate IS NOT NULL AND net_main_rate != 0"
            result = self.db.read_sql(query)
            if not result.is_empty():
                valid_rows = int(result['cnt'][0])
                logger.info(f"V72: stock_fund_flow 有效行数 (net_main_rate!=0): {valid_rows:,}")
            
            logger.info("V72: stock_fund_flow 数据检查通过")
            
        except Exception as e:
            error_msg = f"V72: 检查 stock_fund_flow 表失败：{e}"
            logger.error(error_msg)
            return (False, error_msg)
        
        # 检查 stock_industry_daily 表
        try:
            query = "SELECT COUNT(*) as cnt FROM stock_industry_daily"
            result = self.db.read_sql(query)
            
            if not result.is_empty():
                industry_rows = int(result['cnt'][0])
                logger.info(f"V72: stock_industry_daily 行数：{industry_rows:,}")
                
                if industry_rows < 10000:
                    logger.warning(f"V72: 行业数据不足 ({industry_rows:,})，将使用基础模式")
                else:
                    logger.info("V72: stock_industry_daily 数据检查通过")
            else:
                logger.warning("V72: stock_industry_daily 表为空，将使用基础模式")
            
        except Exception as e:
            logger.warning(f"V72: 检查 stock_industry_daily 表失败：{e}，将使用基础模式")
        
        logger.info("V72: 数据可用性检查完成")
        logger.info("=" * 60)
        return (True, "检查通过")
    
    def load_data(self, start_date: str, end_date: str, 
                  symbols: Optional[List[str]] = None) -> bool:
        """加载数据 - V72 以资金流数据为主"""
        logger.info("=" * 60)
        logger.info(f"V72 数据加载：[{start_date}, {end_date}]")
        logger.info("=" * 60)
        
        try:
            # V72: 先加载资金流数据（覆盖更广）
            self._fund_flow_df = self.data_manager.load_fund_flow_data(start_date, end_date, symbols)
            if self._fund_flow_df is not None and not self._fund_flow_df.is_empty():
                logger.info(f"V72: 资金流数据加载完成 - {self._fund_flow_df.height}行")
            else:
                logger.warning("V72: 资金流数据为空")
                self._fund_flow_df = None
            
            # 加载行业数据
            try:
                self._industry_df = self.data_manager.load_industry_data(start_date, end_date)
                if self._industry_df is not None and not self._industry_df.is_empty():
                    logger.info(f"V72: 行业数据加载完成 - {self._industry_df.height}行")
                else:
                    logger.warning("V72: 行业数据为空，将使用基础模式")
            except Exception as e:
                logger.warning(f"V72: 加载行业数据失败：{e}，将使用基础模式")
                self._industry_df = None
            
            # V72: 加载股票数据用于获取价格
            self._stock_price_df = self.data_manager.load_stock_data(start_date, end_date, symbols)
            if self._stock_price_df is not None and not self._stock_price_df.is_empty():
                logger.info(f"V72: 股票价格数据加载完成 - {self._stock_price_df.height}行")
            else:
                logger.warning("V72: 股票价格数据为空")
                self._stock_price_df = None
            
            # V72: 以资金流数据为主，合并股票价格数据
            if self._fund_flow_df is not None and not self._fund_flow_df.is_empty():
                self._data_df = self._fund_flow_df.clone()
                
                # 合并价格数据
                if self._stock_price_df is not None and not self._stock_price_df.is_empty():
                    price_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
                    available_price_cols = [c for c in price_cols if c in self._stock_price_df.columns]
                    price_data = self._stock_price_df.select(available_price_cols)
                    
                    self._data_df = self._data_df.join(price_data, on=['symbol', 'trade_date'], how='left')
                    
                    # 填充缺失的价格数据
                    self._data_df = self._data_df.with_columns([
                        pl.col('close').fill_null(0.0).alias('close'),
                        pl.col('open').fill_null(0.0).alias('open'),
                        pl.col('high').fill_null(0.0).alias('high'),
                        pl.col('low').fill_null(0.0).alias('low'),
                        pl.col('volume').fill_null(0.0).alias('volume'),
                        pl.col('amount').fill_null(0.0).alias('amount'),
                    ])
                    
                    logger.info(f"V72: 数据合并完成 - {self._data_df.height}行")
                else:
                    # 没有价格数据，添加占位符
                    self._data_df = self._data_df.with_columns([
                        pl.lit(0.0).alias('close'),
                        pl.lit(0.0).alias('open'),
                        pl.lit(0.0).alias('high'),
                        pl.lit(0.0).alias('low'),
                        pl.lit(0.0).alias('volume'),
                        pl.lit(0.0).alias('amount'),
                    ])
            else:
                self._data_df = self._stock_price_df
            
            # 验证最终数据
            if self._data_df is None or self._data_df.is_empty():
                error_msg = "V72: 最终数据为空，程序终止"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # 统计有价格数据的股票数量
            valid_close = self._data_df.filter(pl.col('close') > 0).height
            logger.info(f"V72: 有效价格数据 - {valid_close}行")
            
            return True
            
        except Exception as e:
            logger.error(f"V72: 数据加载失败：{e}")
            raise
    
    def run_backtest(self, start_date: str, end_date: str,
                     symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 60)
        logger.info("V72 全链路执行器 - 启动")
        logger.info("=" * 60)
        logger.info(f"初始资金：{self.initial_capital:.2f} 元")
        logger.info(f"止损：{self.stop_loss_ratio*100:.1f}%, 止盈：{self.profit_target_ratio*100:.1f}%")
        logger.info(f"SNR 阈值：Z-Score > {V72_Z_SCORE_THRESHOLD}, SNR > {V72_SNR_THRESHOLD}")
        logger.info(f"行业背离滤网：连续 {V72_INDUSTRY_NET_OUTFLOW_DAYS} 日净流出")
        logger.info(f"市场宽度规避：行业净流出占比 > {V72_MARKET_WIDTH_THRESHOLD*100:.0f}%")
        logger.info("=" * 60)
        
        # 数据可用性检查
        is_passed, message = self.check_data_availability()
        if not is_passed:
            logger.error(f"V72: 数据可用性检查失败：{message}")
            sys.exit(1)
        
        # 加载数据
        self.load_data(start_date, end_date, symbols)
        
        # 初始化状态
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.trade_audit = []
        self.signals = []
        self._equity_curve = [(start_date, self.initial_capital)]
        self._monthly_returns = {}
        
        # 获取交易日期列表
        unique_dates = sorted(self._data_df['trade_date'].unique().to_list())
        logger.info(f"V72: 共 {len(unique_dates)} 个交易日")
        
        # 逐日回测
        for i, trade_date in enumerate(unique_dates):
            try:
                self._run_daily(trade_date)
                
                # 记录权益曲线
                total_value = self._calculate_total_value()
                self._equity_curve.append((trade_date, total_value))
                
                # 记录月度收益
                month_key = trade_date[:7]
                if month_key not in self._monthly_returns:
                    self._monthly_returns[month_key] = 0.0
                
                # 进度输出
                if (i + 1) % 50 == 0 or i == len(unique_dates) - 1:
                    logger.info(f"V72: 进度 {i+1}/{len(unique_dates)} - 现金：{self.cash:.2f}, 持仓：{len(self.positions)}, 总值：{total_value:.2f}")
                    
            except Exception as e:
                logger.error(f"V72: {trade_date} 处理失败：{e}")
                logger.error(traceback.format_exc())
        
        # 计算 Rank IC
        self._calculate_rank_ic()
        
        # 生成回测报告和分析
        result = self._generate_report_and_analysis(start_date, end_date)
        
        logger.info("=" * 60)
        logger.info("V72 回测完成")
        logger.info("=" * 60)
        
        return result
    
    def _run_daily(self, trade_date: str):
        """运行单日回测"""
        # 1. 获取当日数据
        daily_df = self._data_df.filter(pl.col('trade_date') == trade_date)
        
        if daily_df.is_empty():
            return
        
        # 2. 计算信号
        result, status = self.alpha_center.compute_signals(
            daily_df,
            fund_flow_df=self._fund_flow_df,
            industry_df=self._industry_df
        )
        
        # 3. 计算市场宽度
        market_regime = self.alpha_center.compute_market_width(
            self._fund_flow_df,
            self._industry_df,
            trade_date
        )
        
        # 4. 生成交易信号
        signals = self.alpha_center.generate_signals(result, trade_date, market_regime)
        self.signals.extend(signals)
        
        # 调试：打印市场状态
        if not market_regime.is_safe_period:
            logger.warning(f"V72: {trade_date} 市场危险 ({market_regime.regime_reason})，禁止开仓")
        elif signals:
            logger.info(f"V72: {trade_date} 生成 {len(signals)} 个信号，市场安全，准备买入")
        
        # 5. 处理持仓检查（卖出逻辑）
        self._check_positions(result, trade_date)
        
        # 6. 处理买入信号
        if signals and market_regime.is_safe_period:
            self._process_buy_signals(signals, trade_date)
        elif not market_regime.is_safe_period:
            logger.debug(f"V72: {trade_date} 市场宽度规避触发，跳过买入")
    
    def _calculate_total_value(self) -> float:
        """计算总价值"""
        total_value = self.cash
        for position in self.positions.values():
            total_value += position.market_value
        return total_value
    
    def _check_positions(self, df: pl.DataFrame, trade_date: str):
        """检查持仓，触发卖出"""
        positions_to_sell = []
        
        for symbol, position in self.positions.items():
            stock_data = df.filter(pl.col('symbol') == symbol)
            
            if stock_data.is_empty():
                continue
            
            try:
                row = next(stock_data.iter_rows(named=True))
            except StopIteration:
                continue
            current_price = row.get('close', 0.0)
            low_price = row.get('low', current_price)
            
            # 更新持仓信息
            position.current_price = current_price
            position.market_value = position.shares * current_price
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            position.holding_days += 1
            
            # 更新峰值
            if current_price > position.peak_price:
                position.peak_price = current_price
                position.peak_profit = (current_price - position.avg_cost) / position.avg_cost
            
            # 移动止盈价
            if position.peak_price > 0:
                position.trailing_stop_price = position.peak_price * (1 - self.trailing_stop_ratio)
            
            # 检查卖出条件
            sell_reason = None
            
            # 止损检查
            if current_price <= position.avg_cost * (1 - self.stop_loss_ratio):
                sell_reason = "止损"
                position.stop_loss_triggered = True
            
            # 移动止盈检查
            if current_price <= position.trailing_stop_price and position.peak_profit > 0.05:
                sell_reason = "移动止盈"
                position.trailing_stop_triggered = True
            
            # 目标盈利检查
            if current_price >= position.buy_price * (1 + self.profit_target_ratio):
                sell_reason = "目标盈利"
            
            if sell_reason:
                positions_to_sell.append((symbol, sell_reason, current_price))
        
        # 执行卖出
        for symbol, reason, price in positions_to_sell:
            self._execute_sell(symbol, reason, price, trade_date)
    
    def _process_buy_signals(self, signals: List[V72Signal], trade_date: str):
        """处理买入信号"""
        signals_sorted = sorted(signals, key=lambda x: x.signal_score, reverse=True)
        available_slots = self.max_positions - len(self.positions)
        
        if available_slots <= 0:
            return
        
        position_size = self.cash * self.max_single_position_pct
        
        for signal in signals_sorted[:available_slots]:
            buy_price = signal.close_price * (1 + self.slippage_buy)
            shares = int(position_size / buy_price / 100) * 100
            
            if shares <= 0:
                continue
            
            self._execute_buy(signal, buy_price, shares, trade_date)
    
    def _execute_buy(self, signal: V72Signal, price: float, shares: int, trade_date: str):
        """执行买入"""
        amount = price * shares
        commission = max(self.min_commission, amount * self.commission_rate)
        slippage_cost = price * shares * self.slippage_buy
        transfer_fee = amount * self.transfer_fee
        total_cost = amount + commission + slippage_cost + transfer_fee
        
        if total_cost > self.cash:
            logger.warning(f"V72: 资金不足，跳过 {signal.symbol}")
            return
        
        self.cash -= total_cost
        
        position = V72Position(
            symbol=signal.symbol,
            shares=shares,
            avg_cost=price,
            buy_price=price,
            buy_date=trade_date,
            signal_date=signal.trade_date,
            trade_date=trade_date,
            signal_score=signal.signal_score,
            composite_score=signal.composite_score,
            current_price=price,
            market_value=shares * price,
            z_score=signal.z_score,
            snr_value=signal.snr_value,
            net_main_rate=signal.net_main_rate,
            industry_name=signal.industry_name,
            industry_net_flow_3d=signal.industry_net_flow_3d,
            industry背离=signal.industry背离,
            stop_loss_price=price * (1 - self.stop_loss_ratio),
            trailing_stop_price=price * (1 - self.trailing_stop_ratio),
        )
        
        self.positions[signal.symbol] = position
        
        trade = V72Trade(
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
        )
        self.trades.append(trade)
        
        logger.info(f"V72 买入 {signal.symbol} @ {price:.2f} x {shares}股 (Z-Score: {signal.z_score:.2f}, SNR: {signal.snr_value:.3f})")
    
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
        
        trade = V72Trade(
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
        )
        self.trades.append(trade)
        
        audit = V72TradeAudit(
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
            z_score=position.z_score,
            snr_value=position.snr_value,
            industry背离=position.industry背离,
        )
        self.trade_audit.append(audit)
        
        del self.positions[symbol]
        
        logger.info(f"V72 卖出 {symbol} @ {price:.2f} x {position.shares}股 - {reason} (盈亏：{net_pnl:.2f})")
    
    def _calculate_rank_ic(self):
        """计算 Rank IC"""
        logger.info("=" * 60)
        logger.info("V72 计算 Rank IC")
        
        if self._data_df is None or self._data_df.is_empty():
            logger.warning("V72: 数据为空，无法计算 Rank IC")
            return
        
        try:
            ic_series = self.rank_ic_calculator.calculate_ic_series(self._data_df)
            logger.info(f"V72: 计算 {len(ic_series)} 天的 IC 序列")
            
        except Exception as e:
            logger.error(f"V72: 计算 Rank IC 失败：{e}")
            logger.error(traceback.format_exc())
    
    def _generate_report_and_analysis(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """生成回测报告与深度分析"""
        logger.info("=" * 60)
        logger.info("V72 生成回测报告与深度分析")
        
        # 计算基本指标
        total_value = self._calculate_total_value()
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
        
        # 年化收益
        days = len(self._equity_curve)
        annual_return = (1 + total_pnl_pct) ** (252 / max(days, 1)) - 1
        
        # 夏普比率
        if len(self._equity_curve) > 1:
            returns = []
            for i in range(1, len(self._equity_curve)):
                ret = (self._equity_curve[i][1] - self._equity_curve[i-1][1]) / self._equity_curve[i-1][1]
                returns.append(ret)
            if returns:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0
        
        # Rank IC 统计
        ic_stats = self.rank_ic_calculator.get_ic_statistics()
        monthly_ic_stats = self.rank_ic_calculator.get_monthly_rank_ic_statistics()
        rank_ic_pass, rank_ic_message = self.rank_ic_calculator.check_rank_ic_pass()
        
        # 预测质量分析
        quality_report = analyze_prediction_quality(self.rank_ic_calculator, self.trades)
        
        # 构建报告
        report = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': self.initial_capital,
            'final_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'total_trades': len(self.trade_audit),
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': max_drawdown,
            'rank_ic': ic_stats.get('mean_rank_ic', 0.0),
            'rank_ic_pass': rank_ic_pass,
            'rank_ic_message': rank_ic_message,
            'monthly_rank_ic_mean': monthly_ic_stats.get('monthly_mean_rank_ic', 0.0),
            'monthly_rank_ic_std': monthly_ic_stats.get('monthly_std', 0.0),
            'monthly_rank_ic_pass': monthly_ic_stats.get('monthly_pass', False),
            'quality_report': quality_report,
            'equity_curve': self._equity_curve,
            'monthly_returns': self._monthly_returns,
        }
        
        # 打印报告
        self._print_full_report(report)
        
        # 打印月度 Rank IC 曲线
        self._print_monthly_rank_ic_curve()
        
        return report
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self._equity_curve) < 2:
            return 0.0
        
        peak = self._equity_curve[0][1]
        max_dd = 0.0
        
        for date, value in self._equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _print_full_report(self, report: Dict[str, Any]):
        """打印完整报告"""
        logger.info("=" * 60)
        logger.info("V72 回测报告")
        logger.info("=" * 60)
        
        # 基本信息
        logger.info(f"回测区间：[{report['start_date']}, {report['end_date']}]")
        logger.info(f"初始资金：{report['initial_capital']:.2f} 元")
        logger.info(f"最终价值：{report['final_value']:.2f} 元")
        logger.info(f"总盈亏：{report['total_pnl']:.2f} 元 ({report['total_pnl_pct']*100:.2f}%)")
        logger.info(f"年化收益：{report['annual_return']*100:.2f}%")
        logger.info(f"夏普比率：{report['sharpe']:.2f}")
        logger.info("-" * 60)
        
        # 交易统计
        logger.info("【交易统计】")
        logger.info(f"  交易次数：{report['total_trades']}")
        logger.info(f"  胜率：{report['win_rate']*100:.1f}%")
        logger.info(f"  盈亏比：{report['profit_loss_ratio']:.2f}")
        logger.info(f"  最大回撤：{report['max_drawdown']*100:.2f}%")
        logger.info("-" * 60)
        
        # 审计指标
        logger.info("【审计指标】")
        logger.info(f"  Rank IC: {report['rank_ic']:.4f} ({report['rank_ic_message']})")
        logger.info(f"  月度 Rank IC 均值：{report['monthly_rank_ic_mean']:.4f} (目标：>0.02)")
        logger.info(f"  月度 Rank IC 标准差：{report['monthly_rank_ic_std']:.4f}")
        logger.info(f"  月度 Rank IC 达标：{report['monthly_rank_ic_pass']}")
        logger.info("-" * 60)
        
        # 预测质量
        qr = report['quality_report']
        logger.info("【预测质量】")
        logger.info(f"  Mean Rank IC: {qr.mean_rank_ic:.4f}")
        logger.info(f"  月度 Rank IC 均值：{qr.monthly_rank_ic_mean:.4f}")
        logger.info(f"  质量评分：{qr.quality_score}/100")
        logger.info(f"  总体达标：{qr.overall_pass}")
        logger.info("=" * 60)
    
    def _print_monthly_rank_ic_curve(self):
        """打印月度 Rank IC 变化曲线"""
        logger.info("=" * 60)
        logger.info("V72 月度 Rank IC 变化曲线")
        logger.info("=" * 60)
        
        ic_results = self.rank_ic_calculator.ic_results
        if not ic_results:
            logger.warning("V72: 无 IC 数据")
            return
        
        # 按月份分组
        monthly_ic: Dict[str, List[float]] = {}
        for ic_metric in ic_results:
            month = ic_metric.trade_date[:7]
            if month not in monthly_ic:
                monthly_ic[month] = []
            monthly_ic[month].append(ic_metric.rank_ic)
        
        # 打印曲线
        logger.info("")
        logger.info("月份        | Rank IC 均值 | 样本天数 | 曲线")
        logger.info("-" * 60)
        
        for month in sorted(monthly_ic.keys()):
            ics = monthly_ic[month]
            mean_ic = np.mean(ics)
            days = len(ics)
            
            # 绘制简单的 ASCII 曲线
            bar_length = int((mean_ic + 0.1) * 200)  # 放大以便显示
            bar = "#" * max(0, bar_length) if mean_ic > 0 else "-" * max(0, -bar_length)
            
            status = "PASS" if mean_ic >= 0.02 else "FAIL"
            logger.info(f"{month}    | {mean_ic:+.4f}      | {days:3d}     | {bar} {status}")
        
        logger.info("")
        logger.info("Legend: PASS=达标 (>=0.02)  FAIL=不达标 (<0.02)")
        logger.info("=" * 60)


# ===========================================
# 便捷函数
# ===========================================

def run_v72_backtest(start_date: str, end_date: str,
                     symbols: Optional[List[str]] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：运行 V72 回测
    
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
    engine = V72BacktestEngine(config=config)
    return engine.run_backtest(start_date, end_date, symbols)


# ===========================================
# 主程序
# ===========================================

if __name__ == "__main__":
    # 配置日志 - 设置为 DEBUG 级别以查看调试信息
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="DEBUG"
    )
    
    logger.info("=" * 60)
    logger.info("V72 SNR 审计 + 行业背离滤网 - 回测启动")
    logger.info("=" * 60)
    
    # 检查数据库
    if not DB_AVAILABLE:
        logger.error("V72: db_manager 模块未找到")
        sys.exit(1)
    
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V72: 数据库连接失败：{e}")
        sys.exit(1)
    
    # 运行 2024 年回测
    try:
        config = {
            'db': db,
            'initial_capital': V72_INITIAL_CAPITAL,  # 强制 100,000 元
        }
        
        result = run_v72_backtest("2024-01-01", "2024-12-31", config=config)
        
        logger.info("=" * 60)
        logger.info("V72 回测完成")
        logger.info(f"最终价值：{result.get('final_value', 0):.2f} 元")
        logger.info(f"总盈亏：{result.get('total_pnl', 0):.2f} 元")
        logger.info(f"年化收益：{result.get('annual_return', 0)*100:.2f}%")
        logger.info(f"夏普比率：{result.get('sharpe', 0):.2f}")
        logger.info(f"最大回撤：{result.get('max_drawdown', 0)*100:.2f}%")
        logger.info(f"Rank IC: {result.get('rank_ic', 0):.4f}")
        logger.info(f"月度 Rank IC 均值：{result.get('monthly_rank_ic_mean', 0):.4f}")
        logger.info("=" * 60)
        
    except SystemExit:
        logger.error("V72: 程序已退出")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V72: 回测失败：{e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


__all__ = [
    'V72BacktestEngine',
    'run_v72_backtest',
]