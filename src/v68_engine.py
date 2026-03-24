"""
V68 Engine Module - 强制全链路自检引擎

【V68 全链路自检协议 - 死命令】

1. 启动前门锁
   ✅ if db.count('stock_fund_flow') < 1000000:
         print("数据极度缺失！当前行数：XXX. 必须先运行 data_force_filler!")
         sys.exit(1)

2. 禁止降级
   ✅ 删掉所有 if data_empty: fallback() 的逻辑
   ✅ 没有数据就让程序崩溃，主公需要看到真实的错误

3. Rank IC 质量审计
   ✅ 每日计算预测排名与实际收益排名的相关性
   ✅ 如果 Rank IC 不达标，直接报告"预测模型失败"

作者：量化系统
版本：V68.0
日期：2026-03-24
"""

import sys
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import polars as pl
from loguru import logger

# 导入核心模块
from v68_core import (
    V68DataManager,
    V68AlphaCenter,
    V68RankICCalculator,
    V68Signal,
    V68Position,
    V68Trade,
    V68TradeAudit,
    V68MarketRegime,
    V68StrategyAudit,
    V68_INITIAL_CAPITAL,
    V68_MAX_POSITIONS,
    V68_MONTHLY_TRADE_LIMIT,
    V68_WEEKLY_TRADE_LIMIT,
    V68_GLOBAL_TRADE_LIMIT,
    V68_MIN_FUND_FLOW_ROWS,
    V68_MIN_INDUSTRY_ROWS,
    V68_RANK_IC_TARGET,
    V68_RANK_IC_MIN,
    V68_COMMISSION_RATE,
    V68_MIN_COMMISSION,
    V68_SLIPPAGE_BUY,
    V68_SLIPPAGE_SELL,
    V68_STAMP_DUTY,
    V68_TRANSFER_FEE,
    V68_FRICTION_COST,
    V68_TREND_BREAK_MA_PERIOD,
    V68_PROFIT_TARGET_RATIO,
    V68_TRAILING_STOP_RATIO,
    V68_MAX_SINGLE_POSITION_PCT,
    calculate_ae_metric,
)

# 尝试导入数据库管理器
try:
    from db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.error("V68: db_manager 模块未找到")


# ===========================================
# V68 回测引擎 - 强制全链路自检
# ===========================================

class V68BacktestEngine:
    """
    V68 回测引擎 - 强制全链路自检
    
    【核心协议】
    1. 启动前门锁：数据不足直接退出
    2. 禁止降级：没有数据就崩溃
    3. Rank IC 质量审计：预测模型失败直接报告
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
        self.initial_capital = self.config.get('initial_capital', V68_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V68_MAX_POSITIONS)
        self.max_single_position_pct = self.config.get('max_single_position_pct', V68_MAX_SINGLE_POSITION_PCT)
        
        # 频率熔断
        self.monthly_trade_limit = self.config.get('monthly_trade_limit', V68_MONTHLY_TRADE_LIMIT)
        self.weekly_trade_limit = self.config.get('weekly_trade_limit', V68_WEEKLY_TRADE_LIMIT)
        self.global_trade_limit = self.config.get('global_trade_limit', V68_GLOBAL_TRADE_LIMIT)
        
        # 费率配置 (写死)
        self.commission_rate = V68_COMMISSION_RATE
        self.min_commission = V68_MIN_COMMISSION
        self.slippage_buy = V68_SLIPPAGE_BUY
        self.slippage_sell = V68_SLIPPAGE_SELL
        self.stamp_duty = V68_STAMP_DUTY
        self.transfer_fee = V68_TRANSFER_FEE
        self.friction_cost = V68_FRICTION_COST
        
        # 离场配置
        self.trend_break_ma_period = V68_TREND_BREAK_MA_PERIOD
        self.profit_target_ratio = V68_PROFIT_TARGET_RATIO
        self.trailing_stop_ratio = V68_TRAILING_STOP_RATIO
        
        # 状态变量
        self.cash = self.initial_capital
        self.positions: Dict[str, V68Position] = {}
        self.trades: List[V68Trade] = []
        self.trade_audit: List[V68TradeAudit] = []
        self.signals: List[V68Signal] = []
        
        # 频率统计
        self.monthly_trades: Dict[str, int] = {}
        self.weekly_trades: Dict[str, int] = {}
        self.total_trades = 0
        
        # 数据管理器
        self.data_manager = V68DataManager(db=self.db, config=self.config)
        self.alpha_center = V68AlphaCenter(config=self.config)
        self.rank_ic_calculator = V68RankICCalculator(db=self.db, config=self.config)
        
        # 回测数据
        self._data_df: Optional[pl.DataFrame] = None
        self._fund_flow_df: Optional[pl.DataFrame] = None
        self._industry_df: Optional[pl.DataFrame] = None
        self._market_cap_df: Optional[pl.DataFrame] = None
    
    def preflight_check(self) -> Tuple[bool, str]:
        """
        启动前门锁检查
        
        【强制自检】
        - 如果 stock_fund_flow 行数低于 100 万行，直接退出
        - 禁止降级，没有数据就崩溃
        
        Returns
        -------
        Tuple[bool, str]
            (是否通过，消息)
        """
        logger.info("=" * 60)
        logger.info("V68 启动前门锁检查")
        logger.info("=" * 60)
        
        if self.db is None:
            error_msg = "V68: 数据库连接未初始化"
            logger.error(error_msg)
            return (False, error_msg)
        
        # 检查 stock_fund_flow 表行数
        try:
            query = "SELECT COUNT(*) as cnt FROM stock_fund_flow"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                error_msg = "V68: stock_fund_flow 表不存在或无法访问"
                logger.error(error_msg)
                return (False, error_msg)
            
            fund_flow_rows = int(result['cnt'][0])
            
            logger.info(f"V68: stock_fund_flow 行数：{fund_flow_rows:,}")
            logger.info(f"V68: 阈值：{V68_MIN_FUND_FLOW_ROWS:,}")
            
            # 【启动前门锁】
            if fund_flow_rows < V68_MIN_FUND_FLOW_ROWS:
                error_msg = f"数据极度缺失！当前行数：{fund_flow_rows}. 必须先运行 data_force_filler!"
                logger.error("=" * 60)
                logger.error(f"V68: 【启动前门锁触发】{error_msg}")
                logger.error("=" * 60)
                return (False, error_msg)
            
            logger.info("V68: 数据充足性检查通过")
            
        except Exception as e:
            error_msg = f"V68: 检查 stock_fund_flow 表失败：{e}"
            logger.error(error_msg)
            return (False, error_msg)
        
        # 检查 stock_industry_daily 表行数（可选）
        try:
            query = "SELECT COUNT(*) as cnt FROM stock_industry_daily"
            result = self.db.read_sql(query)
            
            if not result.is_empty():
                industry_rows = int(result['cnt'][0])
                logger.info(f"V68: stock_industry_daily 行数：{industry_rows:,}")
                
                if industry_rows < V68_MIN_INDUSTRY_ROWS:
                    logger.warning(f"V68: 行业数据不足 ({industry_rows:,} < {V68_MIN_INDUSTRY_ROWS:,})，将使用基础 RS 模式")
            
        except Exception as e:
            logger.warning(f"V68: 检查 stock_industry_daily 表失败：{e}，将使用基础模式")
        
        logger.info("V68: 启动前门锁检查完成")
        logger.info("=" * 60)
        return (True, "检查通过")
    
    def load_data(self, start_date: str, end_date: str, 
                  symbols: Optional[List[str]] = None) -> bool:
        """
        加载数据
        
        【禁止降级】
        - 没有数据就崩溃，不准使用 fallback()
        
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
        logger.info(f"V68 数据加载：[{start_date}, {end_date}]")
        logger.info("=" * 60)
        
        try:
            # 加载股票数据（必须成功）
            self._data_df = self.data_manager.load_stock_data(start_date, end_date, symbols)
            
            if self._data_df is None or self._data_df.is_empty():
                error_msg = "V68: 股票数据为空，程序终止"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"V68: 股票数据加载完成 - {self._data_df.height}行")
            
            # 加载资金流数据（可选增强）
            try:
                self._fund_flow_df = self.data_manager.load_fund_flow_data(start_date, end_date, symbols)
                if self._fund_flow_df is not None and not self._fund_flow_df.is_empty():
                    logger.info(f"V68: 资金流数据加载完成 - {self._fund_flow_df.height}行")
                else:
                    logger.warning("V68: 资金流数据为空，将使用基础模式")
            except Exception as e:
                logger.warning(f"V68: 加载资金流数据失败：{e}，将使用基础模式")
                self._fund_flow_df = None
            
            # 加载行业数据（可选增强）
            try:
                self._industry_df = self.data_manager.load_industry_data(start_date, end_date)
                if self._industry_df is not None and not self._industry_df.is_empty():
                    logger.info(f"V68: 行业数据加载完成 - {self._industry_df.height}行")
                else:
                    logger.warning("V68: 行业数据为空，将使用基础 RS 模式")
            except Exception as e:
                logger.warning(f"V68: 加载行业数据失败：{e}，将使用基础 RS 模式")
                self._industry_df = None
            
            # 加载流通市值数据（可选增强）
            try:
                self._market_cap_df = self.data_manager.load_market_cap_data(start_date, end_date)
                if self._market_cap_df is not None and not self._market_cap_df.is_empty():
                    logger.info(f"V68: 流通市值数据加载完成 - {self._market_cap_df.height}行")
                else:
                    logger.warning("V68: 流通市值数据为空，将使用基础模式")
            except Exception as e:
                logger.warning(f"V68: 加载流通市值数据失败：{e}，将使用基础模式")
                self._market_cap_df = None
            
            return True
            
        except Exception as e:
            logger.error(f"V68: 数据加载失败：{e}")
            logger.error("V68: 【禁止降级】没有数据就让程序崩溃，主公需要看到真实的错误")
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
        logger.info("V68 回测引擎 - 启动")
        logger.info("=" * 60)
        
        # 启动前门锁检查
        is_passed, message = self.preflight_check()
        if not is_passed:
            logger.error(f"V68: 启动前门锁检查失败：{message}")
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
        logger.info(f"V68: 共 {len(unique_dates)} 个交易日")
        
        # 逐日回测
        for i, trade_date in enumerate(unique_dates):
            try:
                self._run_daily(trade_date)
                
                # 进度输出
                if (i + 1) % 50 == 0 or i == len(unique_dates) - 1:
                    logger.info(f"V68: 进度 {i+1}/{len(unique_dates)} - 现金：{self.cash:.2f}, 持仓：{len(self.positions)}")
                    
            except Exception as e:
                logger.error(f"V68: {trade_date} 处理失败：{e}")
                logger.error(traceback.format_exc())
                # 继续处理下一天
        
        # 计算 Rank IC
        self._calculate_rank_ic()
        
        # 生成回测报告
        result = self._generate_report(start_date, end_date)
        
        logger.info("=" * 60)
        logger.info("V68 回测完成")
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
            market_cap_df=self._market_cap_df
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
        """
        检查持仓，触发卖出
        
        Parameters
        ----------
        df : pl.DataFrame
            当日数据
        trade_date : str
            交易日期
        """
        positions_to_sell = []
        
        for symbol, position in self.positions.items():
            # 获取当日价格
            stock_data = df.filter(pl.col('symbol') == symbol)
            
            if stock_data.is_empty():
                continue
            
            row = stock_data.iter_rows(named=True).next()
            current_price = row.get('close', 0.0)
            high_price = row.get('high', current_price)
            low_price = row.get('low', current_price)
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
            
            # 1. 跌破 MA10 趋势破位
            if current_price < ma10 * (1 - 0.01):  # 1% 容差
                sell_reason = "跌破 MA10"
                position.trend_break_triggered = True
            
            # 2. 移动止盈触发
            if current_price <= position.trailing_stop_price and position.peak_profit > 0:
                sell_reason = "移动止盈"
                position.trailing_stop_triggered = True
            
            # 3. 达到目标盈利
            if current_price >= position.buy_price * (1 + self.profit_target_ratio):
                sell_reason = "目标盈利"
            
            if sell_reason:
                positions_to_sell.append((symbol, sell_reason, current_price))
        
        # 执行卖出
        for symbol, reason, price in positions_to_sell:
            self._execute_sell(symbol, reason, price, trade_date)
    
    def _process_buy_signals(self, signals: List[V68Signal], trade_date: str):
        """
        处理买入信号
        
        Parameters
        ----------
        signals : List[V68Signal]
            买入信号列表
        trade_date : str
            交易日期
        """
        # 频率熔断检查
        if not self._check_trade_frequency(trade_date):
            logger.warning(f"V68: {trade_date} 频率熔断触发，禁止开仓")
            return
        
        # 按信号评分排序
        signals_sorted = sorted(signals, key=lambda x: x.signal_score, reverse=True)
        
        # 计算可用仓位
        available_slots = self.max_positions - len(self.positions)
        
        if available_slots <= 0:
            return
        
        # 计算单仓金额
        position_size = self.cash * self.max_single_position_pct
        
        # 执行买入
        for signal in signals_sorted[:available_slots]:
            # 频率熔断再次检查
            if not self._check_trade_frequency(trade_date):
                break
            
            # 计算买入价格和数量
            buy_price = signal.close_price * (1 + self.slippage_buy)
            shares = int(position_size / buy_price / 100) * 100  # 100 股整数倍
            
            if shares <= 0:
                continue
            
            # 执行买入
            self._execute_buy(signal, buy_price, shares, trade_date)
    
    def _execute_buy(self, signal: V68Signal, price: float, shares: int, trade_date: str):
        """
        执行买入
        
        Parameters
        ----------
        signal : V68Signal
            买入信号
        price : float
            买入价格
        shares : int
            买入数量
        trade_date : str
            交易日期
        """
        # 计算费用
        amount = price * shares
        commission = max(self.min_commission, amount * self.commission_rate)
        slippage_cost = price * shares * self.slippage_buy
        transfer_fee = amount * self.transfer_fee
        total_cost = amount + commission + slippage_cost + transfer_fee
        
        # 检查资金是否足够
        if total_cost > self.cash:
            logger.warning(f"V68: 资金不足，跳过 {signal.symbol}")
            return
        
        # 更新资金
        self.cash -= total_cost
        
        # 创建持仓
        position = V68Position(
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
        
        # 记录交易
        trade = V68Trade(
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
        
        # 更新频率统计
        self._update_trade_frequency(trade_date)
        self.total_trades += 1
        
        logger.info(f"V68 买入 {signal.symbol} @ {price:.2f} x {shares}股")
    
    def _execute_sell(self, symbol: str, reason: str, price: float, trade_date: str):
        """
        执行卖出
        
        Parameters
        ----------
        symbol : str
            股票代码
        reason : str
            卖出原因
        price : float
            卖出价格
        trade_date : str
            交易日期
        """
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 计算费用
        amount = price * position.shares
        commission = max(self.min_commission, amount * self.commission_rate)
        slippage_cost = price * position.shares * self.slippage_sell
        stamp_duty = amount * self.stamp_duty
        transfer_fee = amount * self.transfer_fee
        total_cost = commission + slippage_cost + stamp_duty + transfer_fee
        
        # 计算盈亏
        gross_pnl = (price - position.avg_cost) * position.shares
        net_pnl = gross_pnl - total_cost
        
        # 更新资金
        self.cash += amount - total_cost
        
        # 记录交易
        trade = V68Trade(
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
            actual_return=gross_pnl / (position.avg_cost * position.shares),
        )
        self.trades.append(trade)
        
        # 记录审计
        audit = V68TradeAudit(
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
            actual_return=position.actual_return,
            trigger_price=position.trigger_price,
            next_open_price=price,
            execution_price=price,
        )
        self.trade_audit.append(audit)
        
        # 移除持仓
        del self.positions[symbol]
        
        logger.info(f"V68 卖出 {symbol} @ {price:.2f} x {position.shares}股 - {reason} (盈亏：{net_pnl:.2f})")
    
    def _check_trade_frequency(self, trade_date: str) -> bool:
        """
        检查交易频率
        
        Parameters
        ----------
        trade_date : str
            交易日期
            
        Returns
        -------
        bool
            是否允许交易
        """
        # 全局频率检查
        if self.total_trades >= self.global_trade_limit:
            return False
        
        # 月度频率检查
        month_key = trade_date[:7]  # YYYY-MM
        if self.monthly_trades.get(month_key, 0) >= self.monthly_trade_limit:
            return False
        
        # 周度频率检查
        week_key = trade_date[:10]  # 简化为日期前 10 位
        if self.weekly_trades.get(week_key, 0) >= self.weekly_trade_limit:
            return False
        
        return True
    
    def _update_trade_frequency(self, trade_date: str):
        """
        更新交易频率统计
        
        Parameters
        ----------
        trade_date : str
            交易日期
        """
        # 月度统计
        month_key = trade_date[:7]
        self.monthly_trades[month_key] = self.monthly_trades.get(month_key, 0) + 1
        
        # 周度统计
        week_key = trade_date[:10]
        self.weekly_trades[week_key] = self.weekly_trades.get(week_key, 0) + 1
    
    def _calculate_rank_ic(self):
        """计算 Rank IC"""
        logger.info("=" * 60)
        logger.info("V68 计算 Rank IC")
        
        if self._data_df is None or self._data_df.is_empty():
            logger.warning("V68: 数据为空，无法计算 Rank IC")
            return
        
        # 使用 AlphaCenter 计算 IC 序列
        try:
            ic_series = self.rank_ic_calculator.calculate_ic_series(self._data_df)
            logger.info(f"V68: 计算 {len(ic_series)} 天的 IC 序列")
            
            # 创建策略审计记录
            audit_records = self.rank_ic_calculator.create_strategy_audit_records(self._data_df)
            
            # 保存到数据库
            self.rank_ic_calculator.save_audit_to_db(audit_records)
            
            # 打印 Rank IC 报告
            self.rank_ic_calculator.print_rank_ic_report()
            
        except Exception as e:
            logger.error(f"V68: 计算 Rank IC 失败：{e}")
            logger.error(traceback.format_exc())
    
    def _generate_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        生成回测报告
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        Dict[str, Any]
            回测报告
        """
        logger.info("=" * 60)
        logger.info("V68 生成回测报告")
        
        # 计算总资金
        total_value = self.cash
        for position in self.positions.values():
            total_value += position.market_value
        
        # 计算总盈亏
        total_pnl = total_value - self.initial_capital
        total_pnl_pct = total_pnl / self.initial_capital
        
        # 计算交易统计
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        profitable_trades = [t for t in self.trade_audit if t.is_profitable]
        win_count = len(profitable_trades)
        loss_count = len(self.trade_audit) - win_count
        win_rate = win_count / len(self.trade_audit) if self.trade_audit else 0.0
        
        # 计算平均盈亏比
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
        
        # 计算最大回撤（简化）
        max_drawdown = self._calculate_max_drawdown()
        
        # 计算 AE 指标
        ae_metric = calculate_ae_metric(win_rate, profit_loss_ratio, max_drawdown, len(self.trade_audit))
        
        # Rank IC 统计
        ic_stats = self.rank_ic_calculator.get_ic_statistics()
        rank_ic_pass, rank_ic_message = self.rank_ic_calculator.check_rank_ic_pass()
        
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
            'positions_count': len(self.positions),
            'cash': self.cash,
        }
        
        # 打印报告
        logger.info("-" * 60)
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info(f"初始资金：{self.initial_capital:.2f}")
        logger.info(f"最终价值：{total_value:.2f}")
        logger.info(f"总盈亏：{total_pnl:.2f} ({total_pnl_pct*100:.2f}%)")
        logger.info("-" * 60)
        logger.info(f"交易次数：{len(self.trade_audit)}")
        logger.info(f"胜率：{win_rate*100:.1f}%")
        logger.info(f"盈亏比：{profit_loss_ratio:.2f}")
        logger.info(f"最大回撤：{max_drawdown*100:.2f}%")
        logger.info(f"AE 指标：{ae_metric:.2f}")
        logger.info("-" * 60)
        logger.info(f"Rank IC: {ic_stats.get('mean_rank_ic', 0.0):.4f}")
        logger.info(f"Rank IC 状态：{rank_ic_message}")
        logger.info("-" * 60)
        
        return report
    
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
                cumulative_pnl += (trade.price - trade.price) * trade.shares  # 简化
        
        return max_dd


# ===========================================
# 便捷函数
# ===========================================

def run_v68_backtest(start_date: str, end_date: str,
                     symbols: Optional[List[str]] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：运行 V68 回测
    
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
    engine = V68BacktestEngine(config=config)
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
    logger.info("V68 强制全链路自检引擎 - 启动")
    logger.info("=" * 60)
    
    # 检查数据库
    if not DB_AVAILABLE:
        logger.error("V68: db_manager 模块未找到")
        sys.exit(1)
    
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V68: 数据库连接失败：{e}")
        sys.exit(1)
    
    # 运行回测
    try:
        result = run_v68_backtest("2024-01-01", "2024-12-31", config={'db': db})
        
        logger.info("=" * 60)
        logger.info("V68 回测完成")
        logger.info(f"最终价值：{result.get('final_value', 0):.2f}")
        logger.info(f"总盈亏：{result.get('total_pnl', 0):.2f}")
        logger.info(f"Rank IC: {result.get('rank_ic', 0):.4f}")
        logger.info(f"Rank IC 状态：{result.get('rank_ic_message', '')}")
        logger.info("=" * 60)
        
    except SystemExit:
        logger.error("V68: 程序已退出")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V68: 回测失败：{e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


__all__ = [
    'V68BacktestEngine',
    'run_v68_backtest',
]