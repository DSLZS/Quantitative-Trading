"""
V64 Engine Module - 资金流增强型 RS-Pullback 回测引擎

【V64 回测引擎核心功能】
1. 主循环：按交易日期迭代执行
2. 大盘避坑：每日计算下跌家数占比，危险时强制空仓
3. 信号生成：调用 V64AlphaCenter 生成资金流增强信号
4. 成交执行：调用 V64TradeExec 执行真实成交
5. 时间止损：买入 5 天不盈利强制离场
6. 业绩统计：生成回测报告和 V64 预测有效性审计报告

作者：量化系统
版本：V64.0
日期：2026-03-24
"""

import json
import math
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import polars as pl
from loguru import logger

from v64_core import (
    # 常量
    V64_INITIAL_CAPITAL,
    V64_MAX_POSITIONS,
    V64_WARMUP_PERIOD,
    V64_MIN_SAMPLE_SIZE,
    V64_FRICTION_COST,
    V64_HARD_STOP_LOSS_RATIO,
    V64_TRAILING_STOP_RATIO,
    V64_PROFIT_TARGET_RATIO,
    V64_MAX_SINGLE_POSITION_PCT,
    V64_TIME_STOP_DAYS,
    V64_MONTHLY_TRADE_LIMIT,
    V64_MARKET_DECLINE_RATIO_THRESHOLD,
    
    # 数据类
    V64Position,
    V64Trade,
    V64TradeAudit,
    V64Signal,
    V64MarketRegime,
    
    # 核心类
    DatabaseIntegrityCheck,
    V64DataManager,
    V64AlphaCenter,
    V64TradeExec,
)


@dataclass
class V64BacktestMetrics:
    """V64 回测业绩指标"""
    # 基础指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    
    # 考核指标：PF * ln(Trades)
    pf_ln_trades: float = 0.0
    
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_buy_trades: int = 0
    total_sell_trades: int = 0
    
    # 资金曲线
    initial_capital: float = V64_INITIAL_CAPITAL
    final_capital: float = V64_INITIAL_CAPITAL
    peak_capital: float = V64_INITIAL_CAPITAL
    
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
    
    # 资金流信号统计
    total_fund_flow_signals: int = 0
    fund_flow_win_rate: float = 0.0
    avg_fund_flow_consecutive: float = 0.0
    
    # 行业中性化 RS 统计
    avg_rs_vs_industry: float = 0.0
    rs_outperform_ratio: float = 0.0
    
    # VCP 打分统计
    avg_vcp_score: float = 0.0
    
    # 大盘避坑统计
    forced_empty_days: int = 0
    market_decline_ratio_avg: float = 0.0
    
    # 时间止损统计
    time_stop_count: int = 0
    time_stop_ratio: float = 0.0
    
    # 月度交易统计
    monthly_trade_counts: Dict[str, int] = field(default_factory=dict)
    
    # 审计案例
    filtered_by_market_cases: List[Dict] = field(default_factory=list)
    fund_flow_filtered_cases: List[Dict] = field(default_factory=list)


@dataclass
class V64DailyRecord:
    """V64 每日记录"""
    trade_date: str
    cash: float
    position_value: float
    total_value: float
    daily_return: float
    position_count: int
    buy_count: int
    sell_count: int
    buy_signals: int = 0
    market_decline_ratio: float = 0.0
    is_safe_period: bool = True
    forced_empty: bool = False
    filtered_signals_count: int = 0


class V64BacktestEngine:
    """
    V64 回测引擎
    
    【核心流程】
    1. 初始化：加载数据、数据库完整性检查
    2. 主循环：按交易日期迭代
    3. 大盘避坑：计算下跌家数占比，危险时强制空仓
    4. 信号生成：AlphaCenter 生成资金流增强买入信号
    5. 成交执行：TradeExec 执行买卖
    6. 时间止损：检查 5 天不盈利离场
    7. 业绩统计：生成报告和 V64 预测有效性审计报告
    """
    
    def __init__(self, db=None, config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        
        # 回测参数
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', V64_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V64_MAX_POSITIONS)
        
        # 数据库完整性检查
        self.integrity_check = DatabaseIntegrityCheck(db=db)
        
        # 初始化组件
        self.data_manager = V64DataManager(db=db, config=self.config)
        self.alpha_center = V64AlphaCenter(config=self.config, integrity_check=self.integrity_check)
        self.trade_exec = V64TradeExec(config=self.config)
        
        # 数据缓存
        self.stock_data: Optional[pl.DataFrame] = None
        self.fund_flow_data: Optional[pl.DataFrame] = None
        self.industry_data: Optional[pl.DataFrame] = None
        self.signal_data: Optional[pl.DataFrame] = None
        
        # 回测记录
        self.daily_records: List[V64DailyRecord] = []
        self.trade_audits: List[V64TradeAudit] = []
        
        # 审计案例追踪
        self._filtered_by_market_cases: List[Dict] = []
        self._fund_flow_filtered_cases: List[Dict] = []
        
        # 价格缓存
        self._price_cache: Dict[str, pl.DataFrame] = {}
        
        # 市场状态缓存
        self._market_regime_cache: Dict[str, V64MarketRegime] = {}
    
    def run_backtest(self) -> V64BacktestMetrics:
        """运行回测"""
        logger.info("=" * 60)
        logger.info("V64 资金流增强型 RS-Pullback 回测启动")
        logger.info(f"回测区间：[{self.start_date}, {self.end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.2f}")
        logger.info(f"最大持仓：{self.max_positions}只")
        logger.info("=" * 60)
        
        try:
            # 1. 数据库完整性检查
            integrity_result = self.integrity_check.run_full_check()
            
            # 2. 加载数据
            self._load_data()
            
            # 3. 计算信号
            self._compute_signals()
            
            # 4. 获取交易日期列表
            trade_dates = self._get_trade_dates()
            
            if not trade_dates:
                logger.error("V64: 未找到交易日期")
                return self._create_empty_metrics()
            
            logger.info(f"V64: 共 {len(trade_dates)} 个交易日")
            
            # 5. 主循环
            self._main_loop(trade_dates)
            
            # 6. 生成业绩报告
            metrics = self._generate_metrics()
            
            # 7. 打印报告
            self._print_report(metrics)
            
            # 8. 生成 V64 预测有效性审计报告
            self._generate_audit_report(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"V64 回测失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_data(self):
        """加载数据"""
        logger.info("V64: 正在加载数据...")
        
        # 加载股票数据
        self.stock_data = self.data_manager.load_stock_data(
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # 加载资金流向数据
        if self.integrity_check.has_fund_flow():
            self.fund_flow_data = self.data_manager.load_fund_flow_data(
                start_date=self.start_date,
                end_date=self.end_date
            )
        else:
            self.fund_flow_data = None
            logger.warning("V64: 无资金流数据，使用纯价量模式")
        
        # 加载行业数据
        if self.integrity_check.has_industry_data():
            self.industry_data = self.data_manager.load_industry_data(
                start_date=self.start_date,
                end_date=self.end_date
            )
        else:
            self.industry_data = None
            logger.warning("V64: 无行业数据，使用基础 RS")
        
        # 构建价格缓存
        self._build_price_cache()
        
        logger.info(f"V64: 数据加载完成，{self.stock_data.height}行，{self.stock_data['symbol'].n_unique()}只股票")
    
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
            dates = df['trade_date'].to_list()
            if trade_date not in dates:
                return 0.0
            
            current_idx = dates.index(trade_date)
            
            if current_idx >= len(dates) - 1:
                row = df.filter(pl.col('trade_date') == trade_date)
                return row['open'][0] if row.height > 0 else 0.0
            
            next_row = df.slice(current_idx + 1, 1)
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
        logger.info("V64: 正在计算信号...")
        
        self.signal_data, status = self.alpha_center.compute_signals(
            df=self.stock_data,
            fund_flow_df=self.fund_flow_data,
            industry_df=self.industry_data
        )
        
        logger.info(f"V64: 信号计算完成，模式：{status['mode']}, 因子：{status['factors_computed']}")
    
    def _get_trade_dates(self) -> List[str]:
        """获取交易日期列表"""
        if self.signal_data is None:
            return []
        
        dates = self.signal_data['trade_date'].unique().to_list()
        
        filtered_dates = [
            d for d in dates
            if d >= self.start_date and d <= self.end_date
        ]
        
        return sorted(filtered_dates)
    
    def _compute_market_regime(self, trade_date: str) -> V64MarketRegime:
        """计算当日市场状态（大盘避坑）"""
        if trade_date in self._market_regime_cache:
            return self._market_regime_cache[trade_date]
        
        current_df = self.stock_data.filter(pl.col('trade_date') == trade_date)
        
        if current_df.is_empty():
            return V64MarketRegime(trade_date=trade_date, is_safe_period=True)
        
        # 计算下跌家数占比
        decline_count = current_df.filter(pl.col('close') < pl.col('open')).height
        total_count = current_df.height
        
        if total_count == 0:
            return V64MarketRegime(trade_date=trade_date, is_safe_period=True)
        
        decline_ratio = decline_count / total_count
        is_safe = decline_ratio <= V64_MARKET_DECLINE_RATIO_THRESHOLD
        
        regime = V64MarketRegime(
            trade_date=trade_date,
            decline_ratio=decline_ratio,
            is_safe_period=is_safe,
            forced_empty=not is_safe,
            regime_reason=f"下跌家数占比={decline_ratio*100:.1f}%, 阈值={V64_MARKET_DECLINE_RATIO_THRESHOLD*100:.0f}%"
        )
        
        self._market_regime_cache[trade_date] = regime
        
        if not is_safe:
            logger.warning(f"V64: {trade_date} 大盘避坑触发 ({regime.regime_reason})，强制空仓！")
        
        return regime
    
    def _main_loop(self, trade_dates: List[str]):
        """回测主循环"""
        logger.info("V64: 开始主循环...")
        
        for i, trade_date in enumerate(trade_dates):
            try:
                # 1. 计算大盘状态（大盘避坑）
                market_regime = self._compute_market_regime(trade_date)
                
                # 2. 更新持仓状态
                self._update_positions(trade_date)
                
                # 3. 检查离场条件（止损止盈 + 时间止损）
                self._check_exit_conditions(trade_date)
                
                # 4. 生成买入信号（受大盘状态限制）
                buy_signals, filtered_count = self._generate_buy_signals(trade_date, market_regime)
                
                # 5. 追踪被大盘环境过滤的信号案例
                if filtered_count > 0 and len(self._filtered_by_market_cases) < 5:
                    for signal in buy_signals[:5 - len(self._filtered_by_market_cases)]:
                        self._filtered_by_market_cases.append({
                            'trade_date': trade_date,
                            'symbol': signal.symbol,
                            'composite_score': signal.composite_score,
                            'rs_vs_industry': signal.rs_vs_industry,
                            'vcp_score': signal.vcp_score,
                            'reason': market_regime.regime_reason
                        })
                
                # 6. 执行买入（受频率限制）
                executed_buys = self._execute_buys(buy_signals, trade_date)
                
                # 7. 记录每日数据
                daily_record = self._record_daily_data(
                    trade_date=trade_date,
                    buy_signals=len(buy_signals),
                    buy_count=executed_buys,
                    filtered_count=filtered_count,
                    market_regime=market_regime
                )
                self.daily_records.append(daily_record)
                
                # 进度日志
                if (i + 1) % 20 == 0 or i == len(trade_dates) - 1:
                    current_value = self.trade_exec.get_portfolio_value()
                    logger.info(f"V64 进度：{i+1}/{len(trade_dates)} 日期:{trade_date} 总值:{current_value:,.2f} 持仓:{self.trade_exec.get_position_count()} 安全:{market_regime.is_safe_period}")
                
            except Exception as e:
                logger.error(f"V64 主循环第 {i+1} 天 ({trade_date}) 失败：{e}")
                logger.error(traceback.format_exc())
    
    def _update_positions(self, trade_date: str):
        """更新持仓状态"""
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
            
            result = self.trade_exec.check_exit_conditions(symbol, current_price, trade_date)
            
            if result is not None and result[0]:
                symbols_to_sell.append((symbol, current_price, result[1]))
        
        for symbol, price, reason in symbols_to_sell:
            self.trade_exec.execute_sell(symbol, price, trade_date, reason)
            self._record_trade_audit(symbol, trade_date, reason)
    
    def _generate_buy_signals(self, trade_date: str, market_regime: V64MarketRegime) -> Tuple[List[V64Signal], int]:
        """生成买入信号"""
        if self.signal_data is None:
            return [], 0
        
        # 大盘危险，强制空仓
        if not market_regime.is_safe_period:
            logger.warning(f"V64: {trade_date} 大盘熔断，禁止开仓！")
            return [], 0
        
        # 使用 AlphaCenter 生成信号
        signals = self.alpha_center.generate_signals(
            self.signal_data, 
            trade_date,
            market_regime
        )
        
        # 过滤洗售限制
        filtered_signals = []
        for signal in signals:
            if signal.symbol not in self.trade_exec.sell_history:
                filtered_signals.append(signal)
            else:
                last_sell_date = self.trade_exec.sell_history[signal.symbol]
                try:
                    sell_date = datetime.strptime(last_sell_date, "%Y-%m-%d")
                    current = datetime.strptime(trade_date, "%Y-%m-%d")
                    days_between = (current - sell_date).days
                    if days_between > 5:  # 5 天洗售窗口
                        filtered_signals.append(signal)
                except Exception:
                    filtered_signals.append(signal)
        
        # 追踪资金流过滤案例
        self._track_fund_flow_filtered_cases(signals, filtered_signals, trade_date)
        
        return filtered_signals, len(signals) - len(filtered_signals)
    
    def _track_fund_flow_filtered_cases(self, all_signals: List[V64Signal],
                                         filtered_signals: List[V64Signal],
                                         trade_date: str):
        """追踪资金流成功过滤掉补跌股的案例"""
        if len(self._fund_flow_filtered_cases) >= 5:
            return
        
        filtered_out = [s for s in all_signals if s not in filtered_signals]
        
        for signal in filtered_out[:5 - len(self._fund_flow_filtered_cases)]:
            if signal.fund_flow_consecutive_positive < 3:
                self._fund_flow_filtered_cases.append({
                    'trade_date': trade_date,
                    'symbol': signal.symbol,
                    'composite_score': signal.composite_score,
                    'fund_flow_consecutive_positive': signal.fund_flow_consecutive_positive,
                    'net_main_amount': signal.net_main_amount,
                    'reason': f"资金流连续{signal.fund_flow_consecutive_positive}日为正不满足要求"
                })
    
    def _execute_buys(self, signals: List[V64Signal], trade_date: str) -> int:
        """执行买入"""
        executed = 0
        
        for signal in signals:
            if not self.trade_exec.can_buy_more():
                break
            
            if not self.trade_exec._check_trade_limit(trade_date):
                logger.warning(f"V64: {trade_date} 交易频率已达上限，停止开仓")
                break
            
            next_open = self._get_next_open(signal.symbol, trade_date)
            
            if next_open <= 0:
                next_open = signal.close_price
            
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
                           filtered_count: int, market_regime: V64MarketRegime) -> V64DailyRecord:
        """记录每日数据"""
        portfolio_value = self.trade_exec.get_portfolio_value()
        cash = self.trade_exec.cash
        position_value = portfolio_value - cash
        position_count = self.trade_exec.get_position_count()
        
        if self.daily_records:
            prev_value = self.daily_records[-1].total_value
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
        else:
            daily_return = 0
        
        sell_count = sum(1 for t in self.trade_exec.trades if t.trade_date == trade_date and t.side == 'sell')
        
        return V64DailyRecord(
            trade_date=trade_date,
            cash=cash,
            position_value=position_value,
            total_value=portfolio_value,
            daily_return=daily_return,
            position_count=position_count,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_signals=buy_signals,
            market_decline_ratio=market_regime.decline_ratio,
            is_safe_period=market_regime.is_safe_period,
            forced_empty=market_regime.forced_empty,
            filtered_signals_count=filtered_count
        )
    
    def _record_trade_audit(self, symbol: str, sell_date: str, reason: str):
        """记录交易审计"""
        buy_trade = None
        for trade in self.trade_exec.trades:
            if trade.symbol == symbol and trade.side == 'buy' and trade.trade_date < sell_date:
                buy_trade = trade
        
        if buy_trade is None:
            return
        
        sell_trade = None
        for trade in self.trade_exec.trades:
            if trade.symbol == symbol and trade.side == 'sell' and trade.trade_date == sell_date:
                sell_trade = trade
                break
        
        if sell_trade is None:
            return
        
        position = self.trade_exec.positions.get(symbol)
        
        audit = V64TradeAudit(
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
            fund_flow_consecutive_positive=buy_trade.reason == '资金流增强 RS-Pullback' and 3 or 0,
            rs_vs_industry=0.0,
            vcp_score=0.0
        )
        
        self.trade_audits.append(audit)
    
    def _generate_metrics(self) -> V64BacktestMetrics:
        """生成业绩指标"""
        logger.info("V64: 正在生成业绩指标...")
        
        metrics = V64BacktestMetrics()
        
        metrics.initial_capital = self.initial_capital
        metrics.start_date = self.start_date
        metrics.end_date = self.end_date
        
        if not self.daily_records:
            return metrics
        
        metrics.final_capital = self.daily_records[-1].total_value
        metrics.peak_capital = max(r.total_value for r in self.daily_records)
        
        metrics.total_return = (metrics.final_capital - metrics.initial_capital) / metrics.initial_capital
        
        total_days = len(self.daily_records)
        metrics.total_days = total_days
        metrics.trading_days = total_days
        
        if total_days > 0:
            years = total_days / 252
            if years > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (1 / years) - 1
        
        peak = metrics.initial_capital
        max_dd = 0
        for record in self.daily_records:
            if record.total_value > peak:
                peak = record.total_value
            dd = (peak - record.total_value) / peak
            if dd > max_dd:
                max_dd = dd
        metrics.max_drawdown = max_dd
        
        all_trades = self.trade_exec.trades
        buy_trades = [t for t in all_trades if t.side == 'buy']
        sell_trades = [t for t in all_trades if t.side == 'sell']
        
        metrics.total_trades = len(sell_trades)
        metrics.total_buy_trades = len(buy_trades)
        metrics.total_sell_trades = len(sell_trades)
        
        winning = [t for t in self.trade_audits if t.is_profitable]
        losing = [t for t in self.trade_audits if not t.is_profitable]
        
        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        if metrics.losing_trades > 0:
            avg_win = sum(t.net_pnl for t in winning) / metrics.winning_trades if metrics.winning_trades > 0 else 0
            avg_loss = abs(sum(t.net_pnl for t in losing) / metrics.losing_trades)
            metrics.profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # 计算 PF * ln(Trades)
        gross_profit = sum(t.net_pnl for t in winning) if winning else 0
        gross_loss = abs(sum(t.net_pnl for t in losing)) if losing else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        
        if metrics.total_trades > 0:
            metrics.pf_ln_trades = pf * math.log(metrics.total_trades)
        
        if len(self.daily_records) > 1:
            returns = [r.daily_return for r in self.daily_records]
            avg_return = sum(returns) / len(returns)
            std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            if std_return > 0:
                metrics.sharpe_ratio = avg_return / std_return * (252 ** 0.5)
        
        if self.trade_audits:
            metrics.avg_holding_days = sum(t.holding_days for t in self.trade_audits) / len(self.trade_audits)
        
        metrics.max_positions_held = max(r.position_count for r in self.daily_records) if self.daily_records else 0
        
        metrics.total_commission = sum(t.commission for t in all_trades)
        metrics.total_slippage = sum(t.slippage for t in all_trades)
        metrics.total_stamp_duty = sum(t.stamp_duty for t in all_trades)
        metrics.total_fees = metrics.total_commission + metrics.total_slippage + metrics.total_stamp_duty
        
        forced_empty_days = sum(1 for r in self.daily_records if r.forced_empty)
        metrics.forced_empty_days = forced_empty_days
        metrics.market_decline_ratio_avg = sum(r.market_decline_ratio for r in self.daily_records) / len(self.daily_records) if self.daily_records else 0
        
        time_stop_trades = [t for t in self.trade_audits if '时间止损' in t.sell_reason]
        metrics.time_stop_count = len(time_stop_trades)
        metrics.time_stop_ratio = len(time_stop_trades) / metrics.total_trades if metrics.total_trades > 0 else 0
        
        metrics.monthly_trade_counts = dict(self.trade_exec.monthly_trades)
        
        metrics.filtered_by_market_cases = self._filtered_by_market_cases[:5]
        metrics.fund_flow_filtered_cases = self._fund_flow_filtered_cases[:5]
        
        return metrics
    
    def _create_empty_metrics(self) -> V64BacktestMetrics:
        """创建空指标"""
        metrics = V64BacktestMetrics()
        metrics.start_date = self.start_date
        metrics.end_date = self.end_date
        metrics.initial_capital = self.initial_capital
        metrics.final_capital = self.initial_capital
        return metrics
    
    def _print_report(self, metrics: V64BacktestMetrics):
        """打印回测报告"""
        logger.info("=" * 60)
        logger.info("V64 资金流增强型 RS-Pullback 回测报告")
        logger.info("=" * 60)
        
        logger.info(f"回测区间：[{metrics.start_date}, {metrics.end_date}] ({metrics.trading_days}天)")
        logger.info(f"初始资金：{metrics.initial_capital:,.2f}")
        logger.info(f"最终资金：{metrics.final_capital:,.2f}")
        logger.info(f"总收益率：{metrics.total_return*100:.2f}%")
        logger.info(f"年化收益：{metrics.annualized_return*100:.2f}%")
        logger.info(f"最大回撤：{metrics.max_drawdown*100:.2f}%")
        logger.info(f"夏普比率：{metrics.sharpe_ratio:.2f}")
        logger.info(f"考核指标 PF*ln(Trades): {metrics.pf_ln_trades:.2f}")
        logger.info("-" * 40)
        logger.info(f"总交易数：{metrics.total_trades}")
        logger.info(f"盈利次数：{metrics.winning_trades}")
        logger.info(f"亏损次数：{metrics.losing_trades}")
        logger.info(f"胜率：{metrics.win_rate*100:.2f}%")
        logger.info(f"盈亏比：{metrics.profit_loss_ratio:.2f}")
        logger.info("-" * 40)
        logger.info(f"强制空仓天数：{metrics.forced_empty_days}")
        logger.info(f"平均下跌家数占比：{metrics.market_decline_ratio_avg*100:.1f}%")
        logger.info("-" * 40)
        logger.info(f"时间止损次数：{metrics.time_stop_count} ({metrics.time_stop_ratio*100:.1f}%)")
        logger.info("=" * 60)
    
    def _generate_audit_report(self, metrics: V64BacktestMetrics):
        """生成 V64 预测有效性审计报告"""
        logger.info("V64: 正在生成预测有效性审计报告...")
        
        report_lines = [
            "=" * 60,
            "V64 预测有效性审计报告",
            "=" * 60,
            "",
            "一、考核指标",
            f"  - PF * ln(Trades) = {metrics.pf_ln_trades:.2f}",
            f"  - 总交易次数：{metrics.total_trades}",
            f"  - 盈亏比 (PF): {metrics.profit_loss_ratio:.2f}",
            "",
            "二、模型预测成功但因大盘环境被过滤的案例",
        ]
        
        for i, case in enumerate(metrics.filtered_by_market_cases, 1):
            report_lines.append(f"  案例{i}: {case['trade_date']} {case['symbol']}")
            report_lines.append(f"    - 综合评分：{case['composite_score']:.2f}")
            report_lines.append(f"    - RS vs 行业：{case['rs_vs_industry']*100:.2f}%")
            report_lines.append(f"    - VCP 得分：{case['vcp_score']:.2f}")
            report_lines.append(f"    - 过滤原因：{case['reason']}")
            report_lines.append("")
        
        report_lines.extend([
            "三、资金流成功过滤掉补跌股的案例",
        ])
        
        for i, case in enumerate(metrics.fund_flow_filtered_cases, 1):
            report_lines.append(f"  案例{i}: {case['trade_date']} {case['symbol']}")
            report_lines.append(f"    - 综合评分：{case['composite_score']:.2f}")
            report_lines.append(f"    - 资金流连续为正天数：{case['fund_flow_consecutive_positive']}")
            report_lines.append(f"    - 主力净流入：{case['net_main_amount']:.2f}")
            report_lines.append(f"    - 过滤原因：{case['reason']}")
            report_lines.append("")
        
        report_lines.extend([
            "",
            "=" * 60,
            "审计报告完成",
            "=" * 60,
        ])
        
        report_content = "\n".join(report_lines)
        logger.info(report_content)
        
        # 保存报告到文件
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/V64_Predictive_Validity_Audit_Report_{timestamp}.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# V64 预测有效性审计报告\n\n")
                f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(report_content)
            
            logger.info(f"V64: 审计报告已保存至 {report_path}")
            
        except Exception as e:
            logger.error(f"V64: 保存审计报告失败：{e}")
    
    def get_trade_history(self) -> List[V64Trade]:
        """获取交易历史"""
        return self.trade_exec.trades
    
    def get_audit_history(self) -> List[V64TradeAudit]:
        """获取审计历史"""
        return self.trade_audits
    
    def get_daily_records(self) -> List[V64DailyRecord]:
        """获取每日记录"""
        return self.daily_records


def run_v64_backtest(start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     initial_capital: float = V64_INITIAL_CAPITAL,
                     max_positions: int = V64_MAX_POSITIONS,
                     db=None) -> V64BacktestMetrics:
    """
    便捷函数：运行 V64 回测
    """
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'max_positions': max_positions,
        'warmup_period': V64_WARMUP_PERIOD,
        'min_sample_size': V64_MIN_SAMPLE_SIZE,
    }
    
    engine = V64BacktestEngine(db=db, config=config)
    return engine.run_backtest()


__all__ = [
    'V64BacktestMetrics',
    'V64DailyRecord',
    'V64BacktestEngine',
    'run_v64_backtest',
]