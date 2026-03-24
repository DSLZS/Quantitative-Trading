"""
V63 Engine Module - VCP 波动率收缩回测引擎

【V63 回测引擎核心功能】
1. 主循环：按交易日期迭代执行
2. 大盘择时：每日计算市场广度，危险时强制空仓
3. 信号生成：调用 V63AlphaCenter 生成 VCP 买入信号
4. 成交执行：调用 V63TradeExec 执行真实成交
5. 时间止损：买入 5 天不盈利强制离场
6. 业绩统计：生成回测报告

作者：量化系统
版本：V63.0
日期：2026-03-23
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import polars as pl
from loguru import logger

from v63_core import (
    # 常量
    V63_INITIAL_CAPITAL,
    V63_MAX_POSITIONS,
    V63_WARMUP_PERIOD,
    V63_MIN_SAMPLE_SIZE,
    V63_FRICTION_COST,
    V63_HARD_STOP_LOSS_RATIO,
    V63_TRAILING_STOP_RATIO,
    V63_PROFIT_TARGET_RATIO,
    V63_MAX_SINGLE_POSITION_PCT,
    V63_WASH_SALE_WINDOW,
    V63_TIME_STOP_DAYS,
    V63_MONTHLY_TRADE_LIMIT,
    V63_MARKET_BREADTH_THRESHOLD,
    
    # 函数
    validate_factors,
    
    # 数据类
    V63Position,
    V63Trade,
    V63TradeAudit,
    V63Signal,
    V63WashSaleRecord,
    V63MarketRegime,
    
    # 核心类
    V63DataManager,
    V63AlphaCenter,
    V63TradeExec,
)


@dataclass
class V63BacktestMetrics:
    """V63 回测业绩指标"""
    # 基础指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_buy_trades: int = 0
    total_sell_trades: int = 0
    
    # 资金曲线
    initial_capital: float = V63_INITIAL_CAPITAL
    final_capital: float = V63_INITIAL_CAPITAL
    peak_capital: float = V63_INITIAL_CAPITAL
    
    # 持仓统计
    avg_holding_days: float = 0.0
    max_positions_held: int = 0
    avg_position_size: float = 0.0
    
    # 费用统计
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_stamp_duty: float = 0.0
    total_fees: float = 0.0
    
    # 回测区间
    start_date: str = ""
    end_date: str = ""
    total_days: int = 0
    trading_days: int = 0
    
    # VCP 信号统计
    total_vcp_signals: int = 0
    vcp_win_rate: float = 0.0
    avg_vcp_amplitude: float = 0.0
    avg_vcp_contraction_days: float = 0.0
    
    # 大盘择时统计
    forced_empty_days: int = 0
    market_breadth_avg: float = 0.0
    
    # 时间止损统计
    time_stop_count: int = 0
    time_stop_ratio: float = 0.0
    
    # 频率控制统计
    monthly_trade_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class V63DailyRecord:
    """V63 每日记录"""
    trade_date: str
    cash: float
    position_value: float
    total_value: float
    daily_return: float
    position_count: int
    buy_count: int
    sell_count: int
    buy_signals: int = 0
    wash_sale_count: int = 0
    market_breadth: float = 0.0
    is_safe_period: bool = True
    forced_empty: bool = False


class V63BacktestEngine:
    """
    V63 回测引擎
    
    【核心流程】
    1. 初始化：加载数据、初始化组件
    2. 主循环：按交易日期迭代
    3. 大盘择时：计算市场广度，危险时强制空仓
    4. 信号生成：AlphaCenter 生成 VCP 买入信号
    5. 成交执行：TradeExec 执行买卖
    6. 时间止损：检查 5 天不盈利离场
    7. 业绩统计：生成报告
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        
        # 回测参数
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', V63_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V63_MAX_POSITIONS)
        
        # 初始化组件
        self.data_manager = V63DataManager(db=db, config=self.config)
        self.alpha_center = V63AlphaCenter(config=self.config)
        self.trade_exec = V63TradeExec(config=self.config)
        
        # 数据缓存
        self.stock_data: Optional[pl.DataFrame] = None
        self.index_data: Optional[pl.DataFrame] = None
        self.industry_data: Optional[pl.DataFrame] = None
        self.signal_data: Optional[pl.DataFrame] = None
        
        # 回测记录
        self.daily_records: List[V63DailyRecord] = []
        self.trade_audits: List[V63TradeAudit] = []
        
        # 价格缓存（用于获取次日开盘价）
        self._price_cache: Dict[str, pl.DataFrame] = {}
        
        # 市场广度缓存
        self._market_breadth_cache: Dict[str, V63MarketRegime] = {}
    
    def run_backtest(self) -> V63BacktestMetrics:
        """
        运行回测
        
        Returns
        -------
        V63BacktestMetrics
            回测业绩指标
        """
        logger.info("=" * 60)
        logger.info("V63 VCP 波动率收缩回测启动")
        logger.info(f"回测区间：[{self.start_date}, {self.end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.2f}")
        logger.info(f"最大持仓：{self.max_positions}只")
        logger.info(f"月度交易上限：{V63_MONTHLY_TRADE_LIMIT}次")
        logger.info(f"大盘熔断阈值：{V63_MARKET_BREADTH_THRESHOLD*100:.0f}%")
        logger.info("=" * 60)
        
        try:
            # 1. 加载数据
            self._load_data()
            
            # 2. 计算信号
            self._compute_signals()
            
            # 3. 获取交易日期列表
            trade_dates = self._get_trade_dates()
            
            if not trade_dates:
                logger.error("V63: 未找到交易日期")
                return self._create_empty_metrics()
            
            logger.info(f"V63: 共 {len(trade_dates)} 个交易日")
            
            # 4. 主循环
            self._main_loop(trade_dates)
            
            # 5. 生成业绩报告
            metrics = self._generate_metrics()
            
            # 6. 打印报告
            self._print_report(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"V63 回测失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(self):
        """加载数据"""
        logger.info("V63: 正在加载数据...")
        
        # 加载股票数据（预加载逻辑在 DataManager 内部处理）
        self.stock_data = self.data_manager.load_stock_data(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 加载指数数据
        self.index_data = self.data_manager.load_index_data(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 加载行业数据
        self.industry_data = self.data_manager.load_industry_data(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 构建价格缓存
        self._build_price_cache()
        
        logger.info(f"V63: 数据加载完成，{self.stock_data.height}行，{self.stock_data['symbol'].n_unique()}只股票")
    
    def _build_price_cache(self):
        """构建价格缓存"""
        symbols = self.stock_data['symbol'].unique().to_list()
        
        for symbol in symbols:
            symbol_df = self.stock_data.filter(pl.col('symbol') == symbol)
            self._price_cache[symbol] = symbol_df.sort('trade_date')
    
    def _get_next_open(self, symbol: str, trade_date: str) -> float:
        """获取次日开盘价"""
        if symbol not in self._price_cache:
            return 0.0
        
        df = self._price_cache[symbol]
        
        try:
            # 找到当前日期的索引
            current_idx = df.filter(pl.col('trade_date') == trade_date).height
            
            if current_idx >= df.height:
                # 已经是最后一天，使用当日开盘价
                row = df.filter(pl.col('trade_date') == trade_date)
                return row['open'][0] if row.height > 0 else 0.0
            
            # 获取下一行的开盘价
            next_row = df.slice(current_idx, 1)
            return next_row['open'][0] if next_row.height > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _get_current_price(self, symbol: str, trade_date: str) -> float:
        """获取当日收盘价"""
        if symbol not in self._price_cache:
            return 0.0
        
        df = self._price_cache[symbol]
        row = df.filter(pl.col('trade_date') == trade_date)
        return row['close'][0] if row.height > 0 else 0.0
    
    def _compute_signals(self):
        """计算所有信号"""
        logger.info("V63: 正在计算信号...")
        
        # 使用 AlphaCenter 计算因子和信号
        self.signal_data, status = self.alpha_center.compute_signals(
            df=self.stock_data,
            index_data=self.index_data
        )
        
        logger.info(f"V63: 信号计算完成，因子：{status['factors_computed']}")
    
    def _get_trade_dates(self) -> List[str]:
        """获取交易日期列表（仅回测区间内）"""
        if self.signal_data is None:
            return []
        
        dates = self.signal_data['trade_date'].unique().to_list()
        
        # 过滤回测区间
        filtered_dates = [
            d for d in dates
            if d >= self.start_date and d <= self.end_date
        ]
        
        return sorted(filtered_dates)
    
    def _compute_market_regime(self, trade_date: str) -> V63MarketRegime:
        """
        计算当日市场状态（大盘择时）
        """
        if trade_date in self._market_breadth_cache:
            return self._market_breadth_cache[trade_date]
        
        # 过滤当日数据
        current_df = self.stock_data.filter(pl.col('trade_date') == trade_date)
        
        if current_df.is_empty():
            return V63MarketRegime(trade_date=trade_date, is_safe_period=True)
        
        # 计算 MA20
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        df_with_ma20 = current_df.with_columns(ma20.alias('ma20'))
        
        # 计算 Close > MA20 的股票占比
        above_ma20_count = df_with_ma20.filter(pl.col('close') > pl.col('ma20')).height
        total_count = df_with_ma20.height
        
        if total_count == 0:
            return V63MarketRegime(trade_date=trade_date, is_safe_period=True)
        
        market_breadth = above_ma20_count / total_count
        is_safe = market_breadth >= V63_MARKET_BREADTH_THRESHOLD
        
        regime = V63MarketRegime(
            trade_date=trade_date,
            market_breadth=market_breadth,
            is_safe_period=is_safe,
            forced_empty=not is_safe,
            regime_reason=f"市场广度={market_breadth*100:.1f}%, 阈值={V63_MARKET_BREADTH_THRESHOLD*100:.0f}%"
        )
        
        self._market_breadth_cache[trade_date] = regime
        
        if not is_safe:
            logger.warning(f"V63: {trade_date} 大盘危险 ({regime.regime_reason})，强制空仓！")
        
        return regime
    
    def _main_loop(self, trade_dates: List[str]):
        """
        回测主循环
        
        Parameters
        ----------
        trade_dates : List[str]
            交易日期列表
        """
        logger.info("V63: 开始主循环...")
        
        for i, trade_date in enumerate(trade_dates):
            try:
                # 1. 计算大盘状态（V63 核心：大盘择时）
                market_regime = self._compute_market_regime(trade_date)
                
                # 2. 更新持仓状态
                self._update_positions(trade_date)
                
                # 3. 检查离场条件（止损止盈 + 时间止损）
                self._check_exit_conditions(trade_date)
                
                # 4. 生成买入信号（受大盘状态限制）
                buy_signals = self._generate_buy_signals(trade_date, market_regime)
                
                # 5. 执行买入（受频率限制）
                executed_buys = self._execute_buys(buy_signals, trade_date)
                
                # 6. 记录每日数据
                daily_record = self._record_daily_data(
                    trade_date=trade_date,
                    buy_signals=len(buy_signals),
                    buy_count=executed_buys,
                    market_regime=market_regime
                )
                self.daily_records.append(daily_record)
                
                # 进度日志
                if (i + 1) % 20 == 0 or i == len(trade_dates) - 1:
                    current_value = self.trade_exec.get_portfolio_value()
                    logger.info(f"V63 进度：{i+1}/{len(trade_dates)} 日期:{trade_date} 总值:{current_value:,.2f} 持仓:{self.trade_exec.get_position_count()} 安全:{market_regime.is_safe_period}")
                
            except Exception as e:
                logger.error(f"V63 主循环第 {i+1} 天 ({trade_date}) 失败：{e}")
                logger.error(traceback.format_exc())
                # 继续执行，不中断
    
    def _update_positions(self, trade_date: str):
        """更新持仓状态"""
        market_data = {}
        
        for symbol in self.trade_exec.positions.keys():
            close_price = self._get_current_price(symbol, trade_date)
            if close_price > 0:
                market_data[symbol] = {'close': close_price}
        
        self.trade_exec.update_positions(market_data, trade_date)
    
    def _check_exit_conditions(self, trade_date: str):
        """检查离场条件（含时间止损）"""
        symbols_to_sell = []
        
        for symbol, position in self.trade_exec.positions.items():
            current_price = self._get_current_price(symbol, trade_date)
            
            if current_price <= 0:
                continue
            
            # 检查离场条件
            result = self.trade_exec.check_exit_conditions(symbol, current_price, trade_date)
            
            if result is not None and result[0]:
                symbols_to_sell.append((symbol, current_price, result[1]))
        
        # 执行卖出
        for symbol, price, reason in symbols_to_sell:
            self.trade_exec.execute_sell(symbol, price, trade_date, reason)
            
            # 记录审计
            self._record_trade_audit(symbol, trade_date, reason)
    
    def _generate_buy_signals(self, trade_date: str, market_regime: V63MarketRegime) -> List[V63Signal]:
        """生成买入信号"""
        if self.signal_data is None:
            return []
        
        # 大盘危险，强制空仓（死命令！）
        if not market_regime.is_safe_period:
            logger.warning(f"V63: {trade_date} 大盘熔断，禁止开仓！")
            return []
        
        # 使用 AlphaCenter 生成信号
        signals = self.alpha_center.generate_signals(
            self.signal_data, 
            trade_date,
            market_regime
        )
        
        # 过滤洗售限制
        filtered_signals = []
        for signal in signals:
            if not self.trade_exec.check_wash_sale(signal.symbol, trade_date):
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _execute_buys(self, signals: List[V63Signal], trade_date: str) -> int:
        """
        执行买入
        
        Parameters
        ----------
        signals : List[V63Signal]
            买入信号列表
        trade_date : str
            交易日期
        
        Returns
        -------
        int
            实际成交数量
        """
        executed = 0
        
        for signal in signals:
            # 检查仓位限制
            if not self.trade_exec.can_buy_more():
                break
            
            # 检查频率限制
            if not self.trade_exec._check_trade_limit(trade_date):
                logger.warning(f"V63: {trade_date} 交易频率已达上限，停止开仓")
                break
            
            # 获取次日开盘价
            next_open = self._get_next_open(signal.symbol, trade_date)
            
            if next_open <= 0:
                next_open = signal.close_price
            
            # 执行买入（使用 Close 作为 Trigger 价格）
            trigger_price = signal.close_price
            capital = self.trade_exec.cash
            
            trade = self.trade_exec.execute_buy(
                signal=signal,
                next_open=next_open,
                trigger_price=trigger_price,
                capital=capital
            )
            
            if trade is not None:
                executed += 1
        
        return executed
    
    def _record_daily_data(self, trade_date: str, buy_signals: int, buy_count: int,
                           market_regime: V63MarketRegime) -> V63DailyRecord:
        """记录每日数据"""
        portfolio_value = self.trade_exec.get_portfolio_value()
        cash = self.trade_exec.cash
        position_value = portfolio_value - cash
        position_count = self.trade_exec.get_position_count()
        
        # 计算日收益率
        if self.daily_records:
            prev_value = self.daily_records[-1].total_value
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            daily_return = 0
        
        # 计算当日卖出数量
        sell_count = sum(1 for t in self.trade_exec.trades if t.trade_date == trade_date and t.side == 'sell')
        
        # 洗售计数
        wash_sale_count = sum(1 for w in self.trade_exec.wash_sale_records if w.blocked_buy_date == trade_date)
        
        return V63DailyRecord(
            trade_date=trade_date,
            cash=cash,
            position_value=position_value,
            total_value=portfolio_value,
            daily_return=daily_return,
            position_count=position_count,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_signals=buy_signals,
            wash_sale_count=wash_sale_count,
            market_breadth=market_regime.market_breadth,
            is_safe_period=market_regime.is_safe_period,
            forced_empty=market_regime.forced_empty
        )
    
    def _record_trade_audit(self, symbol: str, sell_date: str, reason: str):
        """记录交易审计"""
        # 查找对应的买入记录
        buy_trade = None
        for trade in self.trade_exec.trades:
            if trade.symbol == symbol and trade.side == 'buy' and trade.trade_date < sell_date:
                buy_trade = trade
        
        if buy_trade is None:
            return
        
        # 查找卖出记录
        sell_trade = None
        for trade in self.trade_exec.trades:
            if trade.symbol == symbol and trade.side == 'sell' and trade.trade_date == sell_date:
                sell_trade = trade
                break
        
        if sell_trade is None:
            return
        
        # 从持仓获取 VCP 状态
        position = self.trade_exec.positions.get(symbol)
        
        audit = V63TradeAudit(
            symbol=symbol,
            buy_date=buy_trade.trade_date,
            sell_date=sell_date,
            buy_price=buy_trade.price,
            sell_price=sell_trade.price,
            shares=buy_trade.shares,
            gross_pnl=(sell_trade.price - buy_trade.price) * buy_trade.shares,
            total_fees=buy_trade.total_cost - buy_trade.amount + sell_trade.total_cost,
            net_pnl=(sell_trade.price - buy_trade.price) * buy_trade.shares - (buy_trade.total_cost - buy_trade.amount + sell_trade.total_cost),
            holding_days=sell_trade.holding_days,
            is_profitable=(sell_trade.price - buy_trade.price) * buy_trade.shares > 0,
            sell_reason=reason,
            trigger_price=buy_trade.trigger_price,
            next_open_price=buy_trade.next_open_price,
            execution_price=buy_trade.price,
            # VCP 状态
            is_vcp_entry=buy_trade.reason == 'VCP 波动率收缩突破',
            vcp_amplitude=0.0,
            vcp_contraction_days=0,
            volume_dry_days=0,
            breakout_confirmed=False,
            ma_trend_aligned=False
        )
        
        self.trade_audits.append(audit)
    
    def _generate_metrics(self) -> V63BacktestMetrics:
        """生成业绩指标"""
        logger.info("V63: 正在生成业绩指标...")
        
        metrics = V63BacktestMetrics()
        
        # 基础数据
        metrics.initial_capital = self.initial_capital
        metrics.start_date = self.start_date
        metrics.end_date = self.end_date
        
        if not self.daily_records:
            return metrics
        
        # 最终资金
        metrics.final_capital = self.daily_records[-1].total_value
        metrics.peak_capital = max(r.total_value for r in self.daily_records)
        
        # 总收益率
        metrics.total_return = (metrics.final_capital - metrics.initial_capital) / metrics.initial_capital
        
        # 计算年化收益率
        total_days = len(self.daily_records)
        metrics.total_days = total_days
        metrics.trading_days = total_days
        
        if total_days > 0:
            years = total_days / 252
            if years > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1
        
        # 最大回撤
        peak = metrics.initial_capital
        max_dd = 0
        for record in self.daily_records:
            if record.total_value > peak:
                peak = record.total_value
            dd = (peak - record.total_value) / peak
            if dd > max_dd:
                max_dd = dd
        metrics.max_drawdown = max_dd
        
        # 交易统计
        all_trades = self.trade_exec.trades
        buy_trades = [t for t in all_trades if t.side == 'buy']
        sell_trades = [t for t in all_trades if t.side == 'sell']
        
        metrics.total_trades = len(sell_trades)
        metrics.total_buy_trades = len(buy_trades)
        metrics.total_sell_trades = len(sell_trades)
        
        # 胜率
        winning = [t for t in self.trade_audits if t.is_profitable]
        losing = [t for t in self.trade_audits if not t.is_profitable]
        
        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # 盈亏比
        if metrics.losing_trades > 0:
            avg_win = sum(t.net_pnl for t in winning) / metrics.winning_trades if metrics.winning_trades > 0 else 0
            avg_loss = abs(sum(t.net_pnl for t in losing) / metrics.losing_trades)
            metrics.profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 夏普比率
        if len(self.daily_records) > 1:
            returns = [r.daily_return for r in self.daily_records]
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            if std_return > 0:
                metrics.sharpe_ratio = avg_return / std_return * (252 ** 0.5)
        
        # 持仓统计
        if self.trade_audits:
            metrics.avg_holding_days = sum(t.holding_days for t in self.trade_audits) / len(self.trade_audits)
        
        metrics.max_positions_held = max(r.position_count for r in self.daily_records) if self.daily_records else 0
        
        # 费用统计
        metrics.total_commission = sum(t.commission for t in all_trades)
        metrics.total_slippage = sum(t.slippage for t in all_trades)
        metrics.total_stamp_duty = sum(t.stamp_duty for t in all_trades)
        metrics.total_fees = metrics.total_commission + metrics.total_slippage + metrics.total_stamp_duty
        
        # VCP 信号统计
        vcp_trades = [t for t in self.trade_audits if t.is_vcp_entry]
        metrics.total_vcp_signals = len(vcp_trades)
        
        if vcp_trades:
            vcp_winning = [t for t in vcp_trades if t.is_profitable]
            metrics.vcp_win_rate = len(vcp_winning) / len(vcp_trades)
        
        # 大盘择时统计
        forced_empty_days = sum(1 for r in self.daily_records if r.forced_empty)
        metrics.forced_empty_days = forced_empty_days
        metrics.market_breadth_avg = sum(r.market_breadth for r in self.daily_records) / len(self.daily_records) if self.daily_records else 0
        
        # 时间止损统计
        time_stop_trades = [t for t in self.trade_audits if '时间止损' in t.sell_reason]
        metrics.time_stop_count = len(time_stop_trades)
        metrics.time_stop_ratio = len(time_stop_trades) / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # 月度交易统计
        metrics.monthly_trade_counts = dict(self.trade_exec.monthly_trades)
        
        return metrics
    
    def _create_empty_metrics(self) -> V63BacktestMetrics:
        """创建空指标"""
        metrics = V63BacktestMetrics()
        metrics.start_date = self.start_date
        metrics.end_date = self.end_date
        metrics.initial_capital = self.initial_capital
        metrics.final_capital = self.initial_capital
        return metrics
    
    def _print_report(self, metrics: V63BacktestMetrics):
        """打印回测报告"""
        logger.info("=" * 60)
        logger.info("V63 VCP 波动率收缩回测报告")
        logger.info("=" * 60)
        
        logger.info(f"回测区间：[{metrics.start_date}, {metrics.end_date}] ({metrics.trading_days}天)")
        logger.info(f"初始资金：{metrics.initial_capital:,.2f}")
        logger.info(f"最终资金：{metrics.final_capital:,.2f}")
        logger.info(f"总收益率：{metrics.total_return*100:.2f}%")
        logger.info(f"年化收益：{metrics.annualized_return*100:.2f}%")
        logger.info(f"最大回撤：{metrics.max_drawdown*100:.2f}%")
        logger.info(f"夏普比率：{metrics.sharpe_ratio:.2f}")
        logger.info("-" * 40)
        logger.info(f"总交易数：{metrics.total_trades}")
        logger.info(f"盈利次数：{metrics.winning_trades}")
        logger.info(f"亏损次数：{metrics.losing_trades}")
        logger.info(f"胜率：{metrics.win_rate*100:.2f}%")
        logger.info(f"盈亏比：{metrics.profit_loss_ratio:.2f}")
        logger.info("-" * 40)
        logger.info(f"总佣金：{metrics.total_commission:.2f}")
        logger.info(f"总滑点：{metrics.total_slippage:.2f}")
        logger.info(f"总费用：{metrics.total_fees:.2f}")
        logger.info("-" * 40)
        logger.info(f"VCP 信号数：{metrics.total_vcp_signals}")
        logger.info(f"VCP 胜率：{metrics.vcp_win_rate*100:.2f}%")
        logger.info("-" * 40)
        logger.info(f"强制空仓天数：{metrics.forced_empty_days}")
        logger.info(f"平均市场广度：{metrics.market_breadth_avg*100:.1f}%")
        logger.info("-" * 40)
        logger.info(f"时间止损次数：{metrics.time_stop_count} ({metrics.time_stop_ratio*100:.1f}%)")
        logger.info("-" * 40)
        logger.info("月度交易统计:")
        for month, count in sorted(metrics.monthly_trade_counts.items()):
            limit_flag = "⚠️超限" if count >= V63_MONTHLY_TRADE_LIMIT else ""
            logger.info(f"  {month}: {count}次 {limit_flag}")
        logger.info("=" * 60)
    
    def get_trade_history(self) -> List[V63Trade]:
        """获取交易历史"""
        return self.trade_exec.trades
    
    def get_audit_history(self) -> List[V63TradeAudit]:
        """获取审计历史"""
        return self.trade_audits
    
    def get_daily_records(self) -> List[V63DailyRecord]:
        """获取每日记录"""
        return self.daily_records


def run_v63_backtest(start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     initial_capital: float = V63_INITIAL_CAPITAL,
                     max_positions: int = V63_MAX_POSITIONS,
                     db=None) -> V63BacktestMetrics:
    """
    便捷函数：运行 V63 回测
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    initial_capital : float
        初始资金
    max_positions : int
        最大持仓数
    db : optional
        数据库连接
    
    Returns
    -------
    V63BacktestMetrics
        回测业绩指标
    """
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'max_positions': max_positions,
        'warmup_period': V63_WARMUP_PERIOD,
        'min_sample_size': V63_MIN_SAMPLE_SIZE,
    }
    
    engine = V63BacktestEngine(db=db, config=config)
    return engine.run_backtest()


__all__ = [
    'V63BacktestMetrics',
    'V63DailyRecord',
    'V63BacktestEngine',
    'run_v63_backtest',
]