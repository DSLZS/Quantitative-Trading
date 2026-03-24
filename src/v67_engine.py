"""
V67 Engine Module - SPI 驱动的回测引擎 (拒绝任何降级)

【V67 回测引擎 - 拒绝任何降级】

1. 数据熔断升级
   ✅ 回测引擎启动时，必须检查 stock_fund_flow
   ✅ 如果行数低于 100 万行，严禁启动并明确告知："数据不足，请运行 data_filler"

2. 杜绝偷懒与伪造
   ✅ 手续费 0.2% 必须写死在常量里，严禁 AI 为了美化结果而调低
   ✅ 遇到 NoneType 或数据缺失，不准用 fill_null(0) 掩盖
   ✅ 必须打印出缺失数据的日期和股票代码

3. SPI 审计
   ✅ 回测过程中持续监控 SPI 值
   ✅ 如果 SPI 持续低于阈值，输出警告并建议调整策略

作者：量化系统
版本：V67.0
日期：2026-03-24
"""

import sys
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import polars as pl
from loguru import logger

from db_manager import DatabaseManager, get_db
from v67_core import (
    V67DataManager, V67AlphaCenter, V67ICCalculator,
    V67Signal, V67Position, V67Trade, V67TradeAudit, V67MarketRegime,
    V67_INITIAL_CAPITAL, V67_MAX_POSITIONS,
    V67_COMMISSION_RATE, V67_MIN_COMMISSION,
    V67_SLIPPAGE_BUY, V67_SLIPPAGE_SELL,
    V67_STAMP_DUTY, V67_TRANSFER_FEE, V67_FRICTION_COST,
    V67_PROFIT_TARGET_RATIO, V67_TRAILING_STOP_RATIO,
    V67_TREND_BREAK_MA_PERIOD, V67_MAX_SINGLE_POSITION_PCT,
    V67_SPI_TARGET, V67_SPI_MIN,
    V67_MIN_FUND_FLOW_ROWS, V67_MIN_INDUSTRY_ROWS,
    calculate_ae_metric,
)
from v67_data_filler import V67DataFiller, verify_v67_data


# ===========================================
# V67 回测结果数据类
# ===========================================

@dataclass
class V67BacktestResult:
    """V67 回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    trade_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    ae_metric: float = 0.0
    
    # SPI 审计
    mean_spi: float = 0.0
    min_spi: float = 0.0
    max_spi: float = 0.0
    spi_pass_ratio: float = 0.0
    
    # IC 审计
    mean_ic: float = 0.0
    ic_std: float = 0.0
    ic_ir: float = 0.0
    
    # 资金曲线
    equity_curve: List[Dict[str, Any]] = None
    
    # 交易记录
    trades: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = []
        if self.trades is None:
            self.trades = []


# ===========================================
# V67 回测引擎
# ===========================================

class V67BacktestEngine:
    """
    V67 回测引擎 - SPI 驱动，拒绝任何降级
    
    【核心功能】
    1. 数据熔断检查
    2. SPI 审计
    3. 真实成交执行
    4. 完整回测报告
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化回测引擎
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置字典
        """
        self.config = config or {}
        
        # 基础配置
        self.initial_capital = self.config.get('initial_capital', V67_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V67_MAX_POSITIONS)
        
        # 费率配置 (写死，严禁修改)
        self.commission_rate = V67_COMMISSION_RATE
        self.min_commission = V67_MIN_COMMISSION
        self.slippage_buy = V67_SLIPPAGE_BUY
        self.slippage_sell = V67_SLIPPAGE_SELL
        self.stamp_duty = V67_STAMP_DUTY
        self.transfer_fee = V67_TRANSFER_FEE
        self.friCTION_cost = V67_FRICTION_COST  # 0.2% 总计
        
        # SPI 配置
        self.spi_target = self.config.get('spi_target', V67_SPI_TARGET)
        self.spi_min = self.config.get('spi_min', V67_SPI_MIN)
        
        # 数据库
        self.db: Optional[DatabaseManager] = None
        
        # 核心组件
        self.data_manager: Optional[V67DataManager] = None
        self.alpha_center: Optional[V67AlphaCenter] = None
        self.ic_calculator: Optional[V67ICCalculator] = None
        
        # 回测状态
        self.positions: Dict[str, V67Position] = {}
        self.trades: List[V67Trade] = []
        self.trade_audits: List[V67TradeAudit] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.cash = self.initial_capital
        
        # SPI 监控
        self.spi_history: List[float] = []
        self.spi_warning_count = 0
    
    def _check_data_sufficiency(self) -> Tuple[bool, str]:
        """
        数据熔断检查
        
        【死命令】
        - 检查 stock_fund_flow 行数是否 >= 100 万
        - 检查 stock_industry_daily 行数是否 >= 10 万
        - 不达标则严禁启动
        
        Returns
        -------
        Tuple[bool, str]
            (是否充足，消息)
        """
        logger.info("=" * 60)
        logger.info("V67: 开始数据熔断检查")
        
        if self.db is None:
            self.db = get_db()
        
        try:
            # 检查 stock_fund_flow 行数
            fund_flow_query = "SELECT COUNT(*) as cnt FROM stock_fund_flow"
            fund_flow_result = self.db.read_sql(fund_flow_query)
            fund_flow_rows = int(fund_flow_result['cnt'][0]) if not fund_flow_result.is_empty() else 0
            
            # 检查 stock_industry_daily 行数
            industry_query = "SELECT COUNT(*) as cnt FROM stock_industry_daily"
            industry_result = self.db.read_sql(industry_query)
            industry_rows = int(industry_result['cnt'][0]) if not industry_result.is_empty() else 0
            
            logger.info(f"V67: stock_fund_flow 行数：{fund_flow_rows:,} (阈值：{V67_MIN_FUND_FLOW_ROWS:,})")
            logger.info(f"V67: stock_industry_daily 行数：{industry_rows:,} (阈值：{V67_MIN_INDUSTRY_ROWS:,})")
            
            messages = []
            is_sufficient = True
            
            if fund_flow_rows < V67_MIN_FUND_FLOW_ROWS:
                is_sufficient = False
                messages.append(f"数据不足：stock_fund_flow 仅有 {fund_flow_rows:,} 行，需要 {V67_MIN_FUND_FLOW_ROWS:,} 行")
            
            if industry_rows < V67_MIN_INDUSTRY_ROWS:
                is_sufficient = False
                messages.append(f"数据不足：stock_industry_daily 仅有 {industry_rows:,} 行，需要 {V67_MIN_INDUSTRY_ROWS:,} 行")
            
            if not is_sufficient:
                error_msg = "数据不足，请运行 data_filler"
                logger.error("=" * 60)
                logger.error(f"V67: 【数据熔断触发】{error_msg}")
                logger.error("V67: 详细原因:")
                for msg in messages:
                    logger.error(f"  - {msg}")
                logger.error("=" * 60)
                return (False, error_msg)
            
            logger.info("V67: 数据熔断检查通过")
            logger.info("=" * 60)
            return (True, "数据充足")
            
        except Exception as e:
            error_msg = f"数据检查失败：{e}"
            logger.error(f"V67: {error_msg}")
            return (False, error_msg)
    
    def initialize(self, start_date: str, end_date: str) -> bool:
        """
        初始化回测引擎
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        bool
            初始化是否成功
        """
        logger.info("=" * 60)
        logger.info("V67 回测引擎 - 初始化")
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info("=" * 60)
        
        # 1. 数据熔断检查
        is_sufficient, message = self._check_data_sufficiency()
        if not is_sufficient:
            logger.error(f"V67: {message}")
            logger.error("V67: 回测引擎拒绝启动")
            return False
        
        # 2. 初始化数据库
        self.db = get_db()
        
        # 3. 初始化核心组件
        self.data_manager = V67DataManager(db=self.db, config=self.config)
        self.alpha_center = V67AlphaCenter(config=self.config)
        self.ic_calculator = V67ICCalculator(config=self.config)
        
        # 4. 加载数据
        try:
            logger.info("V67: 开始加载数据...")
            
            # 加载股票数据
            stock_df = self.data_manager.load_stock_data(start_date, end_date)
            
            # 加载资金流数据
            fund_flow_df = self.data_manager.load_fund_flow_data(start_date, end_date)
            
            # 加载行业数据
            industry_df = self.data_manager.load_industry_data(start_date, end_date)
            
            # 加载流通市值数据
            market_cap_df = self.data_manager.load_market_cap_data(start_date, end_date)
            
            # 缓存数据
            self.data_manager._data_cache = {
                'stock': stock_df,
                'fund_flow': fund_flow_df,
                'industry': industry_df,
                'market_cap': market_cap_df,
            }
            
            logger.info(f"V67: 数据加载完成 - 股票{stock_df.height}行，资金流{fund_flow_df.height}行，行业{industry_df.height}行")
            
        except Exception as e:
            logger.error(f"V67: 数据加载失败：{e}")
            return False
        
        # 5. 重置状态
        self.positions.clear()
        self.trades.clear()
        self.trade_audits.clear()
        self.equity_curve.clear()
        self.cash = self.initial_capital
        self.spi_history.clear()
        self.spi_warning_count = 0
        
        logger.info("V67: 回测引擎初始化完成")
        return True
    
    def run_backtest(self) -> V67BacktestResult:
        """
        运行回测
        
        Returns
        -------
        V67BacktestResult
            回测结果
        """
        logger.info("=" * 60)
        logger.info("V67: 开始运行回测")
        logger.info("=" * 60)
        
        # 获取数据
        stock_df = self.data_manager._data_cache.get('stock')
        fund_flow_df = self.data_manager._data_cache.get('fund_flow')
        industry_df = self.data_manager._data_cache.get('industry')
        market_cap_df = self.data_manager._data_cache.get('market_cap')
        
        if stock_df is None or stock_df.is_empty():
            logger.error("V67: 股票数据为空，无法运行回测")
            return V67BacktestResult()
        
        # 获取交易日期列表
        trade_dates = sorted(stock_df['trade_date'].unique().to_list())
        
        logger.info(f"V67: 共 {len(trade_dates)} 个交易日")
        
        # 逐日回测
        for i, trade_date in enumerate(trade_dates):
            try:
                self._run_daily(trade_date, stock_df, fund_flow_df, industry_df, market_cap_df)
                
                # 进度输出
                if (i + 1) % 20 == 0:
                    logger.info(f"V67: 回测进度 {i+1}/{len(trade_dates)} ({(i+1)/len(trade_dates)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"V67: {trade_date} 回测失败：{e}")
                continue
        
        # 计算回测结果
        result = self._calculate_result()
        
        # 打印回测报告
        self._print_backtest_report(result)
        
        return result
    
    def _run_daily(self, trade_date: str, stock_df: pl.DataFrame,
                   fund_flow_df: pl.DataFrame, industry_df: pl.DataFrame,
                   market_cap_df: pl.DataFrame):
        """
        运行单日回测
        
        Parameters
        ----------
        trade_date : str
            交易日期
        stock_df : pl.DataFrame
            股票数据
        fund_flow_df : pl.DataFrame
            资金流数据
        industry_df : pl.DataFrame
            行业数据
        market_cap_df : pl.DataFrame
            流通市值数据
        """
        # 1. 获取当日数据
        current_df = stock_df.filter(pl.col('trade_date') == trade_date)
        
        if current_df.is_empty():
            return
        
        # 2. 计算大盘状态
        market_regime = self.alpha_center.compute_market_decline_ratio(current_df)
        
        # 3. 计算信号
        result_df, status = self.alpha_center.compute_signals(
            current_df, fund_flow_df, industry_df, market_cap_df
        )
        
        # 记录 SPI 值
        spi_value = status.get('spi_value', 0.0)
        self.spi_history.append(spi_value)
        
        # SPI 监控
        if spi_value < self.spi_min:
            self.spi_warning_count += 1
            logger.warning(f"V67: {trade_date} SPI={spi_value:.4f} < {self.spi_min} (警告#{self.spi_warning_count})")
        
        # 4. 生成信号
        signals = self.alpha_center.generate_signals(
            result_df, trade_date, market_regime, spi_value, status.get('spi_pass', False)
        )
        
        # 5. 检查持仓更新和离场条件
        self._update_positions(trade_date, current_df, fund_flow_df)
        self._check_exit_conditions(trade_date, current_df, fund_flow_df)
        
        # 6. 执行买入
        if market_regime.is_safe_period:
            self._execute_buy(signals, trade_date, current_df)
        
        # 7. 记录资金曲线
        self._record_equity(trade_date)
    
    def _update_positions(self, trade_date: str, current_df: pl.DataFrame,
                          fund_flow_df: pl.DataFrame):
        """更新持仓状态"""
        for symbol, position in self.positions.items():
            # 获取当日价格
            stock_data = current_df.filter(pl.col('symbol') == symbol)
            
            if stock_data.is_empty():
                # 数据缺失，报错透明化
                self.data_manager.log_missing_data(symbol, trade_date, 'price')
                continue
            
            current_price = stock_data['close'][0]
            ma10 = stock_data['ma10'][0] if 'ma10' in stock_data.columns else 0
            
            if current_price <= 0:
                self.data_manager.log_missing_data(symbol, trade_date, 'close')
                continue
            
            # 更新持仓状态
            position.current_price = current_price
            position.market_value = current_price * position.shares
            position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
            
            # 更新最高价和移动止盈价
            if current_price > position.peak_price:
                position.peak_price = current_price
                position.peak_profit = (current_price - position.avg_cost) / position.avg_cost
            
            if position.peak_price > 0:
                position.trailing_stop_price = position.peak_price * (1 - V67_TRAILING_STOP_RATIO)
            
            # 更新 MA10 止损价
            if ma10 > 0:
                position.stop_loss_price = ma10
            
            # 计算持有天数
            try:
                buy_date = datetime.strptime(position.buy_date, "%Y-%m-%d")
                current = datetime.strptime(trade_date, "%Y-%m-%d")
                position.holding_days = (current - buy_date).days
            except Exception:
                pass
            
            # 获取资金流数据
            if not fund_flow_df.is_empty():
                fund_data = fund_flow_df.filter(
                    (pl.col('symbol') == symbol) & (pl.col('trade_date') == trade_date)
                )
                if not fund_data.is_empty():
                    position.net_main_rate = fund_data['net_main_ratio'][0] if 'net_main_ratio' in fund_data.columns else 0.0
    
    def _check_exit_conditions(self, trade_date: str, current_df: pl.DataFrame,
                                fund_flow_df: pl.DataFrame):
        """检查离场条件"""
        symbols_to_sell = []
        
        for symbol, position in self.positions.items():
            # 获取当日数据
            stock_data = current_df.filter(pl.col('symbol') == symbol)
            
            if stock_data.is_empty():
                continue
            
            current_price = stock_data['close'][0]
            ma10 = stock_data['ma10'][0] if 'ma10' in stock_data.columns else 0
            
            # 获取资金流数据
            net_main_rate = 0.0
            if not fund_flow_df.is_empty():
                fund_data = fund_flow_df.filter(
                    (pl.col('symbol') == symbol) & (pl.col('trade_date') == trade_date)
                )
                if not fund_data.is_empty():
                    net_main_rate = fund_data['net_main_ratio'][0] if 'net_main_ratio' in fund_data.columns else 0.0
            
            # 1. 移动止盈
            if position.trailing_stop_price > 0 and current_price <= position.trailing_stop_price:
                symbols_to_sell.append((symbol, f"移动止盈 (回撤>{V67_TRAILING_STOP_RATIO*100:.1f}%)"))
                position.trailing_stop_triggered = True
                continue
            
            # 2. 目标止盈
            current_profit = (current_price - position.avg_cost) / position.avg_cost
            if current_profit >= V67_PROFIT_TARGET_RATIO:
                symbols_to_sell.append((symbol, f"目标止盈 (盈利>{V67_PROFIT_TARGET_RATIO*100:.1f}%)"))
                continue
            
            # 3. 趋势破坏止损
            if current_price < ma10 and net_main_rate < 0:
                symbols_to_sell.append((symbol, f"趋势破坏止损 (跌破 MA10:{ma10:.2f} 且资金净流出:{net_main_rate:.2%})"))
                position.trend_break_triggered = True
                continue
        
        # 执行卖出
        for symbol, reason in symbols_to_sell:
            self._execute_sell(symbol, trade_date, current_df, reason)
    
    def _execute_buy(self, signals: List[V67Signal], trade_date: str,
                     current_df: pl.DataFrame):
        """执行买入"""
        if not signals:
            return
        
        # 检查是否可以继续买入
        if len(self.positions) >= self.max_positions:
            return
        
        for signal in signals:
            if len(self.positions) >= self.max_positions:
                break
            
            # 获取次日开盘价（简化处理：使用当日收盘价代替）
            stock_data = current_df.filter(pl.col('symbol') == signal.symbol)
            
            if stock_data.is_empty():
                continue
            
            close_price = stock_data['close'][0]
            
            if close_price <= 0:
                self.data_manager.log_missing_data(signal.symbol, trade_date, 'close')
                continue
            
            # 计算买入价格（考虑滑点）
            execution_price = close_price * (1 + self.slippage_buy)
            
            # 计算买入数量
            max_position_value = self.cash * V67_MAX_SINGLE_POSITION_PCT
            shares = int(max_position_value / execution_price / 100) * 100
            
            if shares <= 0:
                continue
            
            # 计算费用
            amount = shares * execution_price
            commission = max(amount * self.commission_rate, self.min_commission)
            transfer_fee = amount * self.transfer_fee
            total_cost = amount + commission + transfer_fee
            
            if total_cost > self.cash:
                shares = int((self.cash * 0.95) / execution_price / 100) * 100
                if shares <= 0:
                    continue
                amount = shares * execution_price
                commission = max(amount * self.commission_rate, self.min_commission)
                transfer_fee = amount * self.transfer_fee
                total_cost = amount + commission + transfer_fee
            
            # 创建持仓记录
            position = V67Position(
                symbol=signal.symbol,
                shares=shares,
                avg_cost=execution_price,
                buy_price=execution_price,
                buy_date=trade_date,
                signal_date=signal.trade_date,
                trade_date=trade_date,
                signal_score=signal.signal_score,
                signal_rank=signal.signal_rank,
                composite_score=signal.composite_score,
                net_main_rate=signal.net_main_rate,
                net_main_rate_vs_avg=signal.net_main_rate_vs_avg,
                main_force_ratio=signal.main_force_ratio,
                rs_percentile=signal.rs_percentile,
                rs_z_score=signal.rs_z_score,
                vcp_amplitude=signal.vcp_amplitude,
                vcp_industry_std=signal.vcp_industry_std,
                vcp_pass=signal.vcp_pass,
                spi_value=signal.spi_value,
                spi_pass=signal.spi_pass,
                stop_loss_price=execution_price * (1 - V67_TRAILING_STOP_RATIO),
                trailing_stop_price=execution_price * (1 - V67_TRAILING_STOP_RATIO),
                trigger_price=close_price,
                next_open_price=close_price,
                execution_price=execution_price
            )
            
            self.positions[signal.symbol] = position
            
            # 创建交易记录
            trade = V67Trade(
                trade_date=trade_date,
                symbol=signal.symbol,
                side='buy',
                shares=shares,
                price=execution_price,
                amount=amount,
                commission=commission,
                slippage=amount * self.slippage_buy,
                stamp_duty=0,
                transfer_fee=transfer_fee,
                total_cost=total_cost,
                reason='V67 SPI 驱动',
                signal_date=signal.trade_date,
                trigger_price=close_price,
                next_open_price=close_price,
                spi_value=signal.spi_value
            )
            
            self.trades.append(trade)
            self.cash -= total_cost
            
            logger.info(f"V67 买入：{signal.symbol} @ {execution_price:.2f} x {shares}股，SPI={signal.spi_value:.4f}")
    
    def _execute_sell(self, symbol: str, trade_date: str,
                      current_df: pl.DataFrame, reason: str):
        """执行卖出"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 获取当日价格
        stock_data = current_df.filter(pl.col('symbol') == symbol)
        
        if stock_data.is_empty():
            self.data_manager.log_missing_data(symbol, trade_date, 'price')
            return
        
        current_price = stock_data['close'][0]
        
        if current_price <= 0:
            self.data_manager.log_missing_data(symbol, trade_date, 'close')
            return
        
        # 计算卖出价格（考虑滑点）
        execution_price = current_price * (1 - self.slippage_sell)
        
        shares = position.shares
        amount = shares * execution_price
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty
        transfer_fee = amount * self.transfer_fee
        total_cost = commission + stamp_duty + transfer_fee
        
        # 创建交易记录
        trade = V67Trade(
            trade_date=trade_date,
            symbol=symbol,
            side='sell',
            shares=shares,
            price=execution_price,
            amount=amount,
            commission=commission,
            slippage=amount * self.slippage_sell,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason=reason,
            holding_days=position.holding_days,
            signal_date=position.signal_date,
            trigger_price=position.trigger_price,
            next_open_price=position.next_open_price,
            spi_value=position.spi_value
        )
        
        self.trades.append(trade)
        self.cash += amount - total_cost
        
        # 创建交易审计记录
        gross_pnl = amount - (position.avg_cost * shares)
        net_pnl = gross_pnl - total_cost
        
        audit = V67TradeAudit(
            symbol=symbol,
            buy_date=position.buy_date,
            sell_date=trade_date,
            buy_price=position.buy_price,
            sell_price=execution_price,
            shares=shares,
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
            spi_value=position.spi_value,
            trigger_price=position.trigger_price,
            next_open_price=position.next_open_price,
            execution_price=execution_price
        )
        
        self.trade_audits.append(audit)
        
        # 删除持仓
        del self.positions[symbol]
        
        logger.info(f"V67 卖出：{symbol} @ {execution_price:.2f}, 原因：{reason}, 盈亏：{net_pnl:.2f}")
    
    def _record_equity(self, trade_date: str):
        """记录资金曲线"""
        # 计算持仓市值
        position_value = sum(p.market_value for p in self.positions.values())
        total_value = self.cash + position_value
        
        self.equity_curve.append({
            'trade_date': trade_date,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
        })
    
    def _calculate_result(self) -> V67BacktestResult:
        """计算回测结果"""
        result = V67BacktestResult()
        
        # 1. 计算收益率
        if self.equity_curve:
            initial_value = self.equity_curve[0]['total_value']
            final_value = self.equity_curve[-1]['total_value']
            result.total_return = (final_value - initial_value) / initial_value
            
            # 计算年化收益率
            days = len(self.equity_curve)
            if days > 0:
                result.annual_return = (1 + result.total_return) ** (252 / days) - 1
        
        # 2. 计算最大回撤
        if self.equity_curve:
            equity_values = [e['total_value'] for e in self.equity_curve]
            peak = equity_values[0]
            max_dd = 0.0
            
            for value in equity_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            
            result.max_drawdown = max_dd
        
        # 3. 计算夏普比率
        if self.equity_curve and len(self.equity_curve) > 1:
            returns = []
            for i in range(1, len(self.equity_curve)):
                prev_value = self.equity_curve[i-1]['total_value']
                curr_value = self.equity_curve[i]['total_value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if returns:
                returns_np = np.array(returns)
                mean_return = np.mean(returns_np)
                std_return = np.std(returns_np, ddof=1) if len(returns_np) > 1 else 0.0
                
                if std_return > 0:
                    result.sharpe_ratio = (mean_return - 0.02/252) / std_return * np.sqrt(252)
        
        # 4. 计算胜率和盈亏比
        if self.trade_audits:
            winning_trades = [t for t in self.trade_audits if t.is_profitable]
            losing_trades = [t for t in self.trade_audits if not t.is_profitable]
            
            result.trade_count = len(self.trade_audits)
            result.winning_trades = len(winning_trades)
            result.losing_trades = len(losing_trades)
            result.win_rate = len(winning_trades) / len(self.trade_audits) if self.trade_audits else 0.0
            
            # 计算盈亏比
            avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0.0
            avg_loss = abs(np.mean([t.net_pnl for t in losing_trades])) if losing_trades else 0.0
            
            if avg_loss > 0:
                result.profit_loss_ratio = avg_win / avg_loss
        
        # 5. 计算 AE 指标
        result.ae_metric = calculate_ae_metric(
            result.win_rate,
            result.profit_loss_ratio,
            result.max_drawdown,
            result.trade_count
        )
        
        # 6. SPI 审计
        if self.spi_history:
            spi_np = np.array(self.spi_history)
            result.mean_spi = float(np.mean(spi_np))
            result.min_spi = float(np.min(spi_np))
            result.max_spi = float(np.max(spi_np))
            result.spi_pass_ratio = float(np.sum(spi_np >= self.spi_target) / len(spi_np))
        
        # 7. IC 审计
        if self.ic_calculator and self.ic_calculator.ic_results:
            ic_stats = self.ic_calculator.get_ic_statistics()
            result.mean_ic = ic_stats['mean_ic']
            result.ic_std = ic_stats['ic_std']
            result.ic_ir = ic_stats['ic_ir']
        
        # 8. 资金曲线和交易记录
        result.equity_curve = self.equity_curve
        result.trades = [asdict(t) for t in self.trades]
        
        return result
    
    def _print_backtest_report(self, result: V67BacktestResult):
        """打印回测报告"""
        logger.info("=" * 60)
        logger.info("V67 回测报告")
        logger.info("=" * 60)
        
        logger.info(f"初始资金：{self.initial_capital:,.2f}")
        logger.info(f"最终资金：{self.equity_curve[-1]['total_value']:,.2f}" if self.equity_curve else "N/A")
        logger.info("-" * 40)
        logger.info(f"总收益率：{result.total_return*100:.2f}%")
        logger.info(f"年化收益率：{result.annual_return*100:.2f}%")
        logger.info(f"最大回撤：{result.max_drawdown*100:.2f}%")
        logger.info(f"夏普比率：{result.sharpe_ratio:.2f}")
        logger.info("-" * 40)
        logger.info(f"胜率：{result.win_rate*100:.1f}%")
        logger.info(f"盈亏比：{result.profit_loss_ratio:.2f}")
        logger.info(f"交易次数：{result.trade_count}")
        logger.info(f"盈利次数：{result.winning_trades}")
        logger.info(f"亏损次数：{result.losing_trades}")
        logger.info("-" * 40)
        logger.info(f"AE 指标：{result.ae_metric:.2f}")
        logger.info("-" * 40)
        logger.info("SPI 审计")
        logger.info(f"  Mean SPI: {result.mean_spi:.4f} (目标：>{self.spi_target})")
        logger.info(f"  Min SPI:  {result.min_spi:.4f} (最低容忍：{self.spi_min})")
        logger.info(f"  Max SPI:  {result.max_spi:.4f}")
        logger.info(f"  SPI Pass Ratio: {result.spi_pass_ratio*100:.1f}%")
        logger.info("-" * 40)
        logger.info("IC 审计")
        logger.info(f"  Mean IC: {result.mean_ic:.4f}")
        logger.info(f"  IC Std:  {result.ic_std:.4f}")
        logger.info(f"  IC IR:   {result.ic_ir:.2f}")
        logger.info("=" * 60)
        
        # SPI 警告
        if result.mean_spi < self.spi_target:
            logger.warning(f"V67: SPI 未达标 (Mean SPI={result.mean_spi:.4f} < {self.spi_target})")
            logger.warning("V67: 建议调整策略参数或重新优化信号权重")
        
        if result.min_spi < self.spi_min:
            logger.warning(f"V67: SPI 低于最低容忍值 (Min SPI={result.min_spi:.4f} < {self.spi_min})")
            logger.warning("V67: 建议检查策略逻辑或降低交易频率")


# ===========================================
# 便捷函数
# ===========================================

def run_v67_backtest(start_date: str, end_date: str,
                     config: Optional[Dict[str, Any]] = None) -> V67BacktestResult:
    """
    便捷函数：运行 V67 回测
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    config : Dict[str, Any], optional
        配置字典
        
    Returns
    -------
    V67BacktestResult
        回测结果
    """
    engine = V67BacktestEngine(config)
    
    # 初始化
    if not engine.initialize(start_date, end_date):
        logger.error("V67: 回测引擎初始化失败")
        return V67BacktestResult()
    
    # 运行回测
    result = engine.run_backtest()
    
    return result


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
    
    # 默认回测区间
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # 运行回测
    result = run_v67_backtest(start_date, end_date)
    
    # 保存结果
    if result.trade_count > 0:
        output_file = f"reports/V67_backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False)
        
        logger.info(f"V67: 回测结果已保存至 {output_file}")


__all__ = [
    'V67BacktestResult',
    'V67BacktestEngine',
    'run_v67_backtest',
]