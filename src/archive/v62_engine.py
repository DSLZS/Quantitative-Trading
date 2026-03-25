"""
V62 Engine Module - RS-Pullback 回测引擎

【V62 回测引擎核心功能】
1. 主循环：按交易日期迭代执行
2. 信号生成：调用 V62AlphaCenter 生成买入信号
3. 成交执行：调用 V62TradeExec 执行真实成交
4. 持仓管理：检查止损止盈条件
5. 业绩统计：生成回测报告

作者：量化系统
版本：V62.0
日期：2026-03-23
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import polars as pl
from loguru import logger

from v62_core import (
    # 常量
    V62_INITIAL_CAPITAL,
    V62_MAX_POSITIONS,
    V62_WARMUP_PERIOD,
    V62_MIN_SAMPLE_SIZE,
    V62_FRICTION_COST,
    V62_HARD_STOP_LOSS_RATIO,
    V62_TRAILING_STOP_RATIO,
    V62_PROFIT_TARGET_RATIO,
    V62_MAX_SINGLE_POSITION_PCT,
    V62_WASH_SALE_WINDOW,
    
    # 函数
    validate_factors,
    
    # 数据类
    V62Position,
    V62Trade,
    V62TradeAudit,
    V62Signal,
    V62WashSaleRecord,
    
    # 核心类
    V62DataManager,
    V62AlphaCenter,
    V62TradeExec,
)


@dataclass
class V62BacktestMetrics:
    """V62 回测业绩指标"""
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
    initial_capital: float = V62_INITIAL_CAPITAL
    final_capital: float = V62_INITIAL_CAPITAL
    peak_capital: float = V62_INITIAL_CAPITAL
    
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
    
    # Pullback 信号统计
    total_pullback_signals: int = 0
    pullback_win_rate: float = 0.0
    avg_pullback_depth: float = 0.0
    avg_pullback_days: float = 0.0


@dataclass
class V62DailyRecord:
    """V62 每日记录"""
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


class V62BacktestEngine:
    """
    V62 回测引擎
    
    【核心流程】
    1. 初始化：加载数据、初始化组件
    2. 主循环：按交易日期迭代
    3. 信号生成：AlphaCenter 生成买入信号
    4. 成交执行：TradeExec 执行买卖
    5. 业绩统计：生成报告
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        
        # 回测参数
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', V62_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V62_MAX_POSITIONS)
        
        # 初始化组件
        self.data_manager = V62DataManager(db=db, config=self.config)
        self.alpha_center = V62AlphaCenter(config=self.config)
        self.trade_exec = V62TradeExec(config=self.config)
        
        # 数据缓存
        self.stock_data: Optional[pl.DataFrame] = None
        self.index_data: Optional[pl.DataFrame] = None
        self.industry_data: Optional[pl.DataFrame] = None
        self.signal_data: Optional[pl.DataFrame] = None
        
        # 回测记录
        self.daily_records: List[V62DailyRecord] = []
        self.trade_audits: List[V62TradeAudit] = []
        
        # 价格缓存（用于获取次日开盘价）
        self._price_cache: Dict[str, pl.DataFrame] = {}
    
    def run_backtest(self) -> V62BacktestMetrics:
        """
        运行回测
        
        Returns
        -------
        V62BacktestMetrics
            回测业绩指标
        """
        logger.info("=" * 60)
        logger.info("V62 RS-Pullback 回测启动")
        logger.info(f"回测区间：[{self.start_date}, {self.end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.2f}")
        logger.info(f"最大持仓：{self.max_positions}只")
        logger.info("=" * 60)
        
        try:
            # 1. 加载数据
            self._load_data()
            
            # 2. 计算信号
            self._compute_signals()
            
            # 3. 获取交易日期列表
            trade_dates = self._get_trade_dates()
            
            if not trade_dates:
                logger.error("V62: 未找到交易日期")
                return self._create_empty_metrics()
            
            logger.info(f"V62: 共 {len(trade_dates)} 个交易日")
            
            # 4. 主循环
            self._main_loop(trade_dates)
            
            # 5. 生成业绩报告
            metrics = self._generate_metrics()
            
            # 6. 打印报告
            self._print_report(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"V62 回测失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(self):
        """加载数据"""
        logger.info("V62: 正在加载数据...")
        
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
        
        # 构建价格缓存（用于获取次日开盘价）
        self._build_price_cache()
        
        logger.info(f"V62: 数据加载完成，{self.stock_data.height}行，{self.stock_data['symbol'].n_unique()}只股票")
    
    def _build_price_cache(self):
        """构建价格缓存"""
        # 按 symbol 分组缓存数据
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
        logger.info("V62: 正在计算信号...")
        
        # 使用 AlphaCenter 计算因子和信号
        self.signal_data, status = self.alpha_center.compute_signals(
            df=self.stock_data,
            index_data=self.index_data
        )
        
        logger.info(f"V62: 信号计算完成，因子：{status['factors_computed']}")
    
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
    
    def _main_loop(self, trade_dates: List[str]):
        """
        回测主循环
        
        Parameters
        ----------
        trade_dates : List[str]
            交易日期列表
        """
        logger.info("V62: 开始主循环...")
        
        prev_value = self.initial_capital
        
        for i, trade_date in enumerate(trade_dates):
            try:
                # 1. 更新持仓状态
                self._update_positions(trade_date)
                
                # 2. 检查离场条件（止损止盈）
                self._check_exit_conditions(trade_date)
                
                # 3. 生成买入信号
                buy_signals = self._generate_buy_signals(trade_date)
                
                # 4. 执行买入
                executed_buys = self._execute_buys(buy_signals, trade_date)
                
                # 5. 记录每日数据
                daily_record = self._record_daily_data(
                    trade_date=trade_date,
                    buy_signals=len(buy_signals),
                    buy_count=executed_buys
                )
                self.daily_records.append(daily_record)
                
                # 进度日志
                if (i + 1) % 20 == 0 or i == len(trade_dates) - 1:
                    current_value = self.trade_exec.get_portfolio_value()
                    logger.info(f"V62 进度：{i+1}/{len(trade_dates)} 日期:{trade_date} 总值:{current_value:,.2f} 持仓:{self.trade_exec.get_position_count()}")
                
            except Exception as e:
                logger.error(f"V62 主循环第 {i+1} 天 ({trade_date}) 失败：{e}")
                logger.error(traceback.format_exc())
                # 继续执行，不中断
    
    def _update_positions(self, trade_date: str):
        """更新持仓状态"""
        # 构建当日市场数据
        market_data = {}
        
        for symbol in self.trade_exec.positions.keys():
            close_price = self._get_current_price(symbol, trade_date)
            if close_price > 0:
                market_data[symbol] = {'close': close_price}
        
        self.trade_exec.update_positions(market_data, trade_date)
    
    def _check_exit_conditions(self, trade_date: str):
        """检查离场条件"""
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
    
    def _generate_buy_signals(self, trade_date: str) -> List[V62Signal]:
        """生成买入信号"""
        if self.signal_data is None:
            return []
        
        # 使用 AlphaCenter 生成信号
        signals = self.alpha_center.generate_signals(self.signal_data, trade_date)
        
        # 过滤洗售限制
        filtered_signals = []
        for signal in signals:
            if not self.trade_exec.check_wash_sale(signal.symbol, trade_date):
                filtered_signals.append(signal)
        
        return filtered_signals
    
    def _execute_buys(self, signals: List[V62Signal], trade_date: str) -> int:
        """
        执行买入
        
        Parameters
        ----------
        signals : List[V62Signal]
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
            
            # 获取次日开盘价
            next_open = self._get_next_open(signal.symbol, trade_date)
            
            if next_open <= 0:
                # 无法获取次日开盘价，使用当日收盘价
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
    
    def _record_daily_data(self, trade_date: str, buy_signals: int, buy_count: int) -> V62DailyRecord:
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
        
        return V62DailyRecord(
            trade_date=trade_date,
            cash=cash,
            position_value=position_value,
            total_value=portfolio_value,
            daily_return=daily_return,
            position_count=position_count,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_signals=buy_signals,
            wash_sale_count=wash_sale_count
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
        
        # 从 sell_history 获取 Pullback 状态（如果存在）
        # 注意：需要在 TradeExec 中保存更多信息
        
        audit = V62TradeAudit(
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
            # Pullback 状态 - 从买入记录中获取
            is_pullback_entry=True,  # 所有买入都是 Pullback 信号触发的
            pullback_days=0,  # 需要从原始信号获取
            pullback_depth=0.0,
            volume_shrunk=False,
            rsrs_zscore=0.0,
            rs_rank=9999
        )
        
        self.trade_audits.append(audit)
    
    def _generate_metrics(self) -> V62BacktestMetrics:
        """生成业绩指标"""
        logger.info("V62: 正在生成业绩指标...")
        
        metrics = V62BacktestMetrics()
        
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
        
        # Pullback 信号统计
        pullback_trades = [t for t in self.trade_audits if t.is_pullback_entry]
        metrics.total_pullback_signals = len(pullback_trades)
        
        if pullback_trades:
            pb_winning = [t for t in pullback_trades if t.is_profitable]
            metrics.pullback_win_rate = len(pb_winning) / len(pullback_trades)
        
        return metrics
    
    def _create_empty_metrics(self) -> V62BacktestMetrics:
        """创建空指标"""
        metrics = V62BacktestMetrics()
        metrics.start_date = self.start_date
        metrics.end_date = self.end_date
        metrics.initial_capital = self.initial_capital
        metrics.final_capital = self.initial_capital
        return metrics
    
    def _print_report(self, metrics: V62BacktestMetrics):
        """打印回测报告"""
        logger.info("=" * 60)
        logger.info("V62 RS-Pullback 回测报告")
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
        logger.info(f"Pullback 信号数：{metrics.total_pullback_signals}")
        logger.info(f"Pullback 胜率：{metrics.pullback_win_rate*100:.2f}%")
        logger.info("=" * 60)
    
    def get_trade_history(self) -> List[V62Trade]:
        """获取交易历史"""
        return self.trade_exec.trades
    
    def get_audit_history(self) -> List[V62TradeAudit]:
        """获取审计历史"""
        return self.trade_audits
    
    def get_daily_records(self) -> List[V62DailyRecord]:
        """获取每日记录"""
        return self.daily_records


def run_v62_backtest(start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     initial_capital: float = V62_INITIAL_CAPITAL,
                     max_positions: int = V62_MAX_POSITIONS,
                     db=None) -> V62BacktestMetrics:
    """
    便捷函数：运行 V62 回测
    
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
    V62BacktestMetrics
        回测业绩指标
    """
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'max_positions': max_positions,
        'warmup_period': V62_WARMUP_PERIOD,
        'min_sample_size': V62_MIN_SAMPLE_SIZE,
    }
    
    engine = V62BacktestEngine(db=db, config=config)
    return engine.run_backtest()


__all__ = [
    'V62BacktestMetrics',
    'V62DailyRecord',
    'V62BacktestEngine',
    'run_v62_backtest',
]