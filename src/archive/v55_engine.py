"""
V55 Engine Module - 真实成交价与精选股系统

【V55 核心改进 - 从"乱割肉"到"精选股"】

1. 审计通报：严禁虚假成交价（最高优先级）
   ✅ 强制规则：Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)
   ✅ 严禁在止损时获得任何价格优势

2. 频率熔断（Global_Trade_Limit）
   ✅ 每周最多开仓 3 只
   ✅ 全场最多持仓 5 只

3. 选股升级：波动率挤压（Volatility Squeeze）
   ✅ 只有当股价处于低位横盘、即将突破时才入场
   ✅ 避开在高位震荡区入场被反复洗盘的情况

4. 止损逻辑优化
   ✅ 硬止损放宽至 2.5 * ATR，给股票呼吸空间
   ✅ 时间止损：买入后 5 天不盈利则减仓 50%

5. 行业对冲硬约束
   ✅ 持有 2 只同行业股票时，第 3 只同行业股票严禁买入

作者：量化系统
版本：V55.0
日期：2026-03-21
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import polars as pl
from loguru import logger

from v55_core import (
    V55FactorEngine, V55RiskManager, V55Position, V55Trade,
    V55TradeAudit, V55MarketRegime, V55DrawdownState,
    V55_INITIAL_CAPITAL, V55_MAX_POSITIONS, V55_MAINTAIN_TOP_N,
    V55_ENTRY_TOP_N, V55_USE_T1_EXECUTION,
    V55_MA60_FILTER, V55_VOLUME_FILTER_ENABLED,
    V55_MAX_DRAWDOWN_TARGET, V55_ANNUAL_RETURN_TARGET,
    V55_PROFIT_LOSS_RATIO_TARGET, V55_GLOBAL_TRADE_LIMIT,
    V55_DATABASE_TABLES, V55IndustryLoader,
    V55_VOLATILITY_SQUEEZE_ENABLED, V55_TIME_STOP_ENABLED,
    V55_TIME_STOP_DAYS, V55_TIME_STOP_REDUCE_RATIO,
    V55_WEEKLY_TRADE_LIMIT,
)


class V55BacktestEngine:
    """
    V55 回测引擎 - 真实成交价与精选股系统
    
    【核心功能】
    1. T/T+1 严格隔离：信号日 (T) 使用收盘价计算，交易日 (T+1) 使用开盘价成交
    2. 真实成交价：止损单成交价 = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)
    3. 频率熔断：每周最多开仓 3 只，全场最多持仓 5 只
    4. 波动率挤压选股：低位横盘 + 突破信号才入场
    5. 行业对冲硬约束：最多持有 2 只同行业股票
    6. 时间止损：买入后 5 天不盈利则减仓 50%
    """
    
    def __init__(self, initial_capital: float = V55_INITIAL_CAPITAL, db=None):
        self.initial_capital = initial_capital
        self.db = db
        self.risk_manager = V55RiskManager(initial_capital=initial_capital)
        self.factor_engine = V55FactorEngine()
        self.industry_loader = V55IndustryLoader(db=db)
        
        self.portfolio_values: List[Dict[str, Any]] = []
        self.daily_trades: List[V55Trade] = []
        self.trade_log: List[V55TradeAudit] = []
        self.current_date: Optional[str] = None
        self.previous_date: Optional[str] = None
        self.factor_cache: Dict[str, pl.DataFrame] = {}
        self.price_cache: Dict[str, pl.DataFrame] = {}
        
        self.start_date: str = ""
        self.end_date: str = ""
        self.backtest_stats: Dict[str, Any] = {}
        
        # 将 portfolio_values 引用添加到 risk_manager
        self.risk_manager.portfolio_values = self.portfolio_values
    
    def run_backtest(self, price_df: pl.DataFrame, start_date: str, end_date: str,
                     index_df: Optional[pl.DataFrame] = None,
                     industry_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """运行回测"""
        try:
            logger.info("=" * 60)
            logger.info("V55 回测引擎启动 - 真实成交价与精选股系统")
            logger.info("=" * 60)
            
            self.start_date = start_date
            self.end_date = end_date
            
            trade_dates = sorted(price_df['trade_date'].unique().to_list())
            trade_dates = [d for d in trade_dates if start_date <= d <= end_date]
            
            if not trade_dates:
                logger.error("No trade dates found")
                return self._generate_empty_stats()
            
            logger.info(f"回测区间：{start_date} 至 {end_date}")
            logger.info(f"交易日数量：{len(trade_dates)}")
            
            # 构建次日价格映射（用于真实成交价计算）
            next_day_price_map = self._build_next_day_price_map(price_df)
            
            for i, trade_date in enumerate(trade_dates):
                # 获取次日价格数据（用于真实成交价）
                next_day = self._get_next_trade_day(trade_date, trade_dates)
                next_day_prices = None
                if next_day and next_day in next_day_price_map:
                    next_day_prices = next_day_price_map[next_day]
                
                self._run_trading_day(trade_date, price_df, index_df, industry_df, next_day_prices)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"进度：{i + 1}/{len(trade_dates)} 交易日")
            
            logger.info("=" * 60)
            logger.info("V55 回测完成")
            logger.info("=" * 60)
            
            return self._generate_final_stats()
            
        except Exception as e:
            logger.error(f"V55 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            return self._generate_error_stats(e)
    
    def _build_next_day_price_map(self, price_df: pl.DataFrame) -> Dict[str, pl.DataFrame]:
        """
        构建次日价格映射表
        
        用于真实成交价计算：Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)
        """
        try:
            next_day_map = {}
            
            # 按 trade_date 分组
            for date in price_df['trade_date'].unique().to_list():
                day_prices = price_df.filter(pl.col('trade_date') == date)
                next_day_map[date] = day_prices
            
            return next_day_map
            
        except Exception as e:
            logger.error(f"Error building next_day_price_map: {e}")
            return {}
    
    def _get_next_trade_day(self, current_date: str, trade_dates: List[str]) -> Optional[str]:
        """获取下一个交易日"""
        try:
            if current_date in trade_dates:
                idx = trade_dates.index(current_date)
                if idx + 1 < len(trade_dates):
                    return trade_dates[idx + 1]
        except:
            pass
        return None
    
    def _run_trading_day(self, trade_date: str, price_df: pl.DataFrame,
                         index_df: Optional[pl.DataFrame],
                         industry_df: Optional[pl.DataFrame],
                         next_day_prices: Optional[pl.DataFrame] = None) -> None:
        """
        V55 运行单个交易日 - 严格 T/T+1 隔离
        
        流程：
        1. T 日：计算因子，产生信号，缓存
        2. T+1 日：使用 T 日信号，以开盘价执行交易
        3. 真实成交价：止损单使用 min(Trigger_Price, Next_Open_Price) * (1 - Slippage)
        """
        try:
            self.previous_date = self.current_date
            self.current_date = trade_date
            
            self.risk_manager.reset_daily_counters(trade_date)
            
            day_prices = price_df.filter(pl.col('trade_date') == trade_date)
            
            if day_prices.is_empty():
                return
            
            # V55: 计算当日因子
            current_factors, factor_status = self.factor_engine.compute_all_factors(
                day_prices, industry_df, self.db, self.start_date, self.end_date
            )
            self.factor_cache[trade_date] = current_factors
            self.risk_manager.factor_status = factor_status
            
            # V55 关键：在因子计算之后、交易执行之前调用 handle_exits
            if self.previous_date:
                prev_factors = self.factor_cache.get(self.previous_date)
                if prev_factors is not None:
                    # 先执行退出信号（包含真实成交价逻辑）
                    self._handle_exits(trade_date, day_prices, prev_factors, current_factors, next_day_prices)
                    # 然后执行入场信号
                    self._execute_signals(trade_date, day_prices, prev_factors, current_factors)
            
            # V55: 处理时间止损减仓
            if V55_TIME_STOP_ENABLED:
                self._handle_time_stop_reductions(trade_date, day_prices)
            
            self._update_market_regime(trade_date, index_df, day_prices)
            
            self._record_portfolio_state(trade_date, day_prices)
            
            if len(self.factor_cache) > 30:
                oldest_date = min(self.factor_cache.keys())
                del self.factor_cache[oldest_date]
            
        except Exception as e:
            logger.error(f"Error running trading day {trade_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _handle_exits(self, trade_date: str, price_df: pl.DataFrame,
                      signal_factors: pl.DataFrame,
                      current_factors: pl.DataFrame,
                      next_day_prices: Optional[pl.DataFrame] = None) -> None:
        """
        V55 处理退出信号 - 真实成交价逻辑
        
        核心规则：Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)
        """
        try:
            if not V55_USE_T1_EXECUTION:
                return
            
            # V55 真实成交价：传入次日价格数据
            exit_signals = self.risk_manager.check_exits(
                self.risk_manager.positions, trade_date, price_df,
                current_factors if current_factors is not None else signal_factors,
                next_day_prices
            )
            
            # exit_signals: [(symbol, reason, trigger_price, next_open_price), ...]
            for symbol, reason, trigger_price, next_open_price in exit_signals:
                open_price = self._get_open_price(price_df, symbol)
                if open_price and open_price > 0:
                    # V55 核心：传入 trigger_price 和 next_open_price 用于真实成交价计算
                    self.risk_manager.execute_sell(
                        trade_date, symbol, open_price, 
                        reason=reason,
                        trigger_price=trigger_price,
                        next_open_price=next_open_price
                    )
            
        except Exception as e:
            logger.error(f"Error handling exits on {trade_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _handle_time_stop_reductions(self, trade_date: str, price_df: pl.DataFrame) -> None:
        """
        V55 处理时间止损减仓 - 买入后 5 天不盈利则减仓 50%
        """
        try:
            for symbol, pos in list(self.risk_manager.positions.items()):
                # 检查是否需要减仓
                if (pos.time_stop_triggered and 
                    not pos.time_stop_reduced and
                    pos.holding_days >= V55_TIME_STOP_DAYS):
                    
                    open_price = self._get_open_price(price_df, symbol)
                    if open_price and open_price > 0:
                        self.risk_manager.execute_time_stop_reduce(trade_date, symbol, open_price)
                        pos.time_stop_reduced = True
                        
        except Exception as e:
            logger.error(f"Error handling time_stop_reductions: {e}")
    
    def _execute_signals(self, trade_date: str, price_df: pl.DataFrame,
                         signal_factors: pl.DataFrame,
                         current_factors: pl.DataFrame) -> None:
        """
        V55 执行信号 - 严格 T/T+1 隔离，包含波动率挤压过滤和行业对冲
        """
        try:
            if not V55_USE_T1_EXECUTION:
                return
            
            # 检查是否可以开新仓（频率熔断）
            if not self.risk_manager.can_open_new_position():
                return
            
            # V55 最大持仓限制
            if len(self.risk_manager.positions) >= V55_MAX_POSITIONS:
                return
            
            if current_factors is None:
                return
            
            # 选择入场候选（波动率挤压过滤 + 趋势过滤 + 成交量萎缩过滤）
            candidates = self._select_entry_candidates_with_filters(current_factors, price_df)
            
            for candidate in candidates:
                symbol = candidate['symbol']
                open_price = self._get_open_price(price_df, symbol)
                if not open_price or open_price <= 0:
                    continue
                
                atr = candidate.get('atr') or 0.01
                if atr is None or atr <= 0:
                    atr = 0.01
                
                signal_date = self.previous_date or trade_date
                industry_name = candidate.get('industry_name', '')
                
                total_assets = self.risk_manager.get_total_portfolio_value(trade_date)
                
                # V55 波动率适配头寸计算
                shares, target_amount, position_tier = self.risk_manager.calculate_position_size(
                    symbol, atr, open_price, total_assets
                )
                
                if shares < 100:
                    continue
                
                self.risk_manager.execute_buy(
                    trade_date=trade_date,
                    symbol=symbol,
                    open_price=open_price,
                    atr=atr,
                    target_amount=target_amount,
                    signal_date=signal_date,
                    signal_score=candidate.get('signal_score', 0),
                    signal_rank=candidate.get('rank', 999),
                    composite_score=candidate.get('composite_score', 0),
                    composite_percentile=candidate.get('percentile', 0),
                    ma5=candidate.get('ma5', 0),
                    ma20=candidate.get('ma20', 0),
                    ma60=candidate.get('ma60', 0),
                    ma120=candidate.get('ma120', 0),
                    industry_name=industry_name,
                    volatility_ratio=candidate.get('volatility_ratio', 0),
                    volume_shrunk=candidate.get('volume_shrunk', False),
                    volatility_squeeze_low=candidate.get('volatility_squeeze_low', False),
                    squeeze_breakout=candidate.get('squeeze_breakout', False),
                    reason="v55_entry_signal"
                )
                
                if len(self.risk_manager.positions) >= V55_MAX_POSITIONS:
                    break
                    
        except Exception as e:
            logger.error(f"Error executing signals on {trade_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _select_entry_candidates_with_filters(self, factors_df: pl.DataFrame,
                                                price_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """
        V55 选择入场候选股票 - 包含波动率挤压过滤、趋势过滤和成交量萎缩过滤
        
        核心逻辑：
        1. 按 composite_rank 排序
        2. 波动率挤压过滤：低位横盘 + 突破信号
        3. 趋势过滤：股价必须在 60 日均线之上
        4. 成交量萎缩过滤：防止买入"僵尸股"
        5. 行业对冲：最多持有 2 只同行业股票
        """
        try:
            required_cols = ['symbol', 'composite_rank', 'close']
            for col in required_cols:
                if col not in factors_df.columns:
                    logger.warning(f"Missing required column: {col}")
                    return []
            
            if 'composite_rank' not in factors_df.columns:
                logger.warning(f"composite_rank column missing. Available columns: {factors_df.columns}")
                return []
            
            # 过滤并排序
            candidates_df = factors_df.filter(
                (pl.col('close') > 0) & 
                (pl.col('composite_rank').is_not_null())
            ).sort('composite_rank')
            
            if candidates_df.is_empty():
                logger.warning(f"No candidates found. factors_df shape: {factors_df.shape}")
                return []
            
            # 获取可用空间
            available_slots = V55_MAX_POSITIONS - len(self.risk_manager.positions)
            
            if available_slots <= 0:
                return []
            
            candidates = []
            
            # 遍历所有股票，按排名选择
            for row in candidates_df.iter_rows(named=True):
                if len(candidates) >= available_slots:
                    break
                
                symbol = row['symbol']
                rank = row.get('composite_rank', 999) or 999
                
                # V55: 只选择 Top N 排名的股票
                if rank > V55_ENTRY_TOP_N:
                    continue
                
                # V55 趋势过滤：股价在 60 日均线之上
                if V55_MA60_FILTER:
                    price_above_ma60 = row.get('price_above_ma60', None)
                    if price_above_ma60 is None:
                        ma60_val = row.get('ma60', 0) or 0
                        close_val = row.get('close', 0) or 0
                        price_above_ma60 = close_val > ma60_val if ma60_val > 0 else True
                    if not price_above_ma60:
                        ma60_val = row.get('ma60', 0) or 0
                        if ma60_val > 0:
                            logger.debug(f"TREND FILTER: {symbol} - Price below MA60, skipped")
                            continue
                
                # V55 成交量萎缩过滤
                if V55_VOLUME_FILTER_ENABLED:
                    volume_filter_pass = row.get('volume_filter_pass', True)
                    is_volume_shrunk = row.get('is_volume_shrunk', False)
                    vol_ma20 = row.get('vol_ma20', 0) or 0
                    if vol_ma20 > 0:
                        if not volume_filter_pass or is_volume_shrunk:
                            logger.debug(f"VOLUME FILTER: {symbol} - Volume shrunk, skipped")
                            continue
                
                # 获取必要数据
                atr_value = row.get('atr_20') or 0.01
                if atr_value is None or atr_value <= 0:
                    atr_value = 0.01
                
                ma60 = row.get('ma60', 0) or 0
                ma20 = row.get('ma20', 0) or 0
                ma5 = row.get('ma5', 0) or 0
                ma120 = row.get('ma120', 0) or 0
                
                # 计算波动率比率
                close_price = row.get('close', 1)
                volatility_ratio = atr_value / close_price if close_price > 0 else 0
                
                # 获取行业
                industry = ''
                if 'industry_name' in factors_df.columns:
                    industry = row.get('industry_name', '') or ''
                if not industry:
                    industry = self.industry_loader.get_industry_for_symbol(symbol)
                
                # V55 行业对冲硬约束检查
                industry_ok, industry_reason = self.risk_manager.check_industry_constraint(industry)
                if not industry_ok:
                    logger.debug(f"INDUSTRY CONSTRAINT: {symbol} - {industry_reason}")
                    continue
                
                # V55 波动率挤压标记
                volatility_squeeze_low = row.get('volatility_squeeze_low', False) or False
                squeeze_breakout = row.get('squeeze_breakout', False) or False
                
                candidates.append({
                    'symbol': symbol,
                    'signal_score': row.get('composite_score', 0) or 0,
                    'rank': rank,
                    'composite_score': row.get('composite_score', 0) or 0,
                    'percentile': row.get('composite_percentile', 1) or 1,
                    'atr': atr_value,
                    'ma5': ma5,
                    'ma20': ma20,
                    'ma60': ma60,
                    'ma120': ma120,
                    'industry_name': industry,
                    'volatility_ratio': volatility_ratio,
                    'volume_shrunk': row.get('is_volume_shrunk', False) or False,
                    'volatility_squeeze_low': volatility_squeeze_low,
                    'squeeze_breakout': squeeze_breakout,
                    'price_above_ma60': price_above_ma60 if V55_MA60_FILTER else True,
                })
            
            if candidates:
                logger.info(f"Selected {len(candidates)} candidates: {[c['symbol'] for c in candidates]}")
            else:
                logger.warning(f"No candidates selected. Check filters (MA60, volume, industry)")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error selecting candidates with filters: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _get_open_price(self, price_df: pl.DataFrame, symbol: str) -> Optional[float]:
        """获取开盘价"""
        try:
            row = price_df.filter((pl.col('symbol') == symbol)).select('open').row(0)
            if row:
                return float(row[0])
        except:
            pass
        return None
    
    def _update_market_regime(self, trade_date: str, index_df: Optional[pl.DataFrame],
                               price_df: pl.DataFrame) -> None:
        """更新大盘状态"""
        try:
            if index_df is not None and not index_df.is_empty():
                try:
                    index_row = index_df.filter(pl.col('trade_date') == trade_date).row(0)
                    if index_row:
                        index_close = float(index_row[1]) if len(index_row) > 1 else 0
                        self.risk_manager.market_regime = V55MarketRegime(
                            trade_date=trade_date,
                            index_close=index_close,
                            is_risk_period=False
                        )
                        self.risk_manager.is_risk_period = False
                        return
                except:
                    pass
            
            self.risk_manager.market_regime = V55MarketRegime(
                trade_date=trade_date,
                is_risk_period=False
            )
            self.risk_manager.is_risk_period = False
            
        except Exception as e:
            logger.error(f"Error updating market regime: {e}")
    
    def _record_portfolio_state(self, trade_date: str, price_df: pl.DataFrame) -> None:
        """记录组合状态"""
        try:
            for symbol, pos in self.risk_manager.positions.items():
                try:
                    row = price_df.filter((pl.col('symbol') == symbol)).select('close').row(0)
                    if row:
                        pos.current_price = float(row[0])
                        pos.market_value = pos.shares * pos.current_price
                        pos.unrealized_pnl = (pos.current_price - pos.avg_cost) * pos.shares
                except:
                    pass
            
            total_value = self.risk_manager.get_total_portfolio_value(trade_date)
            
            if not self.portfolio_values:
                initial_value = self.initial_capital
            else:
                initial_value = self.initial_capital
            
            cumulative_return = (total_value - initial_value) / initial_value if initial_value > 0 else 0
            
            self.portfolio_values.append({
                'trade_date': trade_date,
                'total_value': total_value,
                'cash': self.risk_manager.cash,
                'market_value': total_value - self.risk_manager.cash,
                'cumulative_return': cumulative_return,
                'daily_drawdown': self.risk_manager.drawdown_state.daily_drawdown,
                'weekly_drawdown': self.risk_manager.drawdown_state.weekly_drawdown,
                'num_positions': len(self.risk_manager.positions),
            })
            
        except Exception as e:
            logger.error(f"Error recording portfolio state: {e}")
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.portfolio_values) < 2:
            return 0.0
        
        max_dd = 0.0
        peak = self.portfolio_values[0]['total_value']
        
        for pv in self.portfolio_values:
            v = pv['total_value']
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _generate_empty_stats(self) -> Dict[str, Any]:
        return {'error': 'No data to backtest'}
    
    def _generate_error_stats(self, error: Exception) -> Dict[str, Any]:
        return {
            'error': str(error),
            'traceback': traceback.format_exc(),
            'status': 'failed'
        }
    
    def _generate_final_stats(self) -> Dict[str, Any]:
        """生成最终统计"""
        try:
            if not self.portfolio_values:
                return self._generate_empty_stats()
            
            pv_df = pl.DataFrame(self.portfolio_values)
            
            initial_value = self.initial_capital
            final_value = self.portfolio_values[-1]['total_value'] if self.portfolio_values else initial_value
            total_return = (final_value - initial_value) / initial_value if initial_value > 0 else 0
            
            # 正确的回撤计算
            values = pv_df['total_value'].to_list()
            if len(values) > 1:
                max_dd = 0.0
                peak = values[0]
                for v in values:
                    if v > peak:
                        peak = v
                    dd = (peak - v) / peak if peak > 0 else 0
                    if dd > max_dd:
                        max_dd = dd
            else:
                max_dd = 0.0
            
            # 计算最大单日亏损
            max_single_day_loss = 0.0
            for i in range(1, len(values)):
                daily_return = (values[i] - values[i-1]) / values[i-1] if values[i-1] > 0 else 0
                if daily_return < 0 and abs(daily_return) > max_single_day_loss:
                    max_single_day_loss = abs(daily_return)
            
            trade_log = self.risk_manager.trade_log
            profitable_trades = [t for t in trade_log if t.is_profitable]
            losing_trades = [t for t in trade_log if not t.is_profitable]
            
            avg_win = sum(t.net_pnl for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0
            avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
            
            profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            num_trades = len(trade_log)
            
            trade_dates = pv_df['trade_date'].to_list()
            if len(trade_dates) >= 2:
                start = datetime.strptime(trade_dates[0], "%Y-%m-%d")
                end = datetime.strptime(trade_dates[-1], "%Y-%m-%d")
                days = (end - start).days
                years = days / 365.25 if days > 0 else 1
                annual_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else total_return
            else:
                annual_return = total_return
            
            # V55 审计：交易次数检查
            trade_count_exceeded = num_trades > V55_GLOBAL_TRADE_LIMIT
            
            # V55 对赌条款检查
            mdd_exceeded = max_dd > V55_MAX_DRAWDOWN_TARGET
            
            # V55 三级防御体系统计
            defense_stats = self.risk_manager.get_three_level_defense_stats()
            
            # 统计卖出原因
            hard_stop_count = defense_stats['hard_stop_count']
            trailing_profit_count = defense_stats['trailing_profit_count']
            ma20_exit_count = defense_stats['ma20_exit_count']
            rank_drop_count = defense_stats['rank_drop_count']
            stop_loss_count = defense_stats['initial_stop_count']
            time_stop_count = defense_stats['time_stop_count']
            
            # 统计头寸层级
            standard_tier_count = sum(1 for t in trade_log if t.position_tier == "standard")
            reduced_tier_count = sum(1 for t in trade_log if t.position_tier == "reduced")
            
            # 统计成交量萎缩过滤
            volume_shrunk_count = sum(1 for t in trade_log if t.volume_shrunk_at_entry)
            
            # 统计波动率挤压低位入场
            squeeze_low_count = sum(1 for t in trade_log if t.volatility_squeeze_low)
            
            # V55 真实成交价审计
            stop_audit_records = self.risk_manager.get_stop_audit_records()
            stop_audit_valid = all(r.get('is_valid', True) for r in stop_audit_records)
            
            stats = {
                'status': 'success',
                'version': 'V55.0',
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_dd,
                'max_single_day_loss': max_single_day_loss,
                'num_trades': num_trades,
                'profitable_trades': len(profitable_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(profitable_trades) / num_trades if num_trades > 0 else 0,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_loss_ratio': profit_loss_ratio,
                'portfolio_values': self.portfolio_values,
                'trades': [
                    {
                        'trade_date': t.trade_date,
                        'symbol': t.symbol,
                        'side': t.side,
                        'shares': t.shares,
                        'price': t.price,
                        'execution_price': t.execution_price,
                        'reason': t.reason,
                        'signal_date': t.signal_date,
                        't_plus_1': t.t_plus_1,
                        # V55 真实成交价审计字段
                        'trigger_price': t.trigger_price,
                        'next_open_price': t.next_open_price,
                        'min_trigger_open': t.min_trigger_open,
                        'slippage_applied': t.slippage_applied,
                    }
                    for t in self.risk_manager.trades
                ],
                'trade_log': [
                    {
                        'symbol': t.symbol,
                        'buy_date': t.buy_date,
                        'sell_date': t.sell_date,
                        'buy_price': t.buy_price,
                        'sell_price': t.sell_price,
                        'net_pnl': t.net_pnl,
                        'holding_days': t.holding_days,
                        'sell_reason': t.sell_reason,
                        'position_tier': t.position_tier,
                        # V55 真实成交价审计字段
                        'trigger_price': t.trigger_price,
                        'next_open_price': t.next_open_price,
                        'execution_price': t.execution_price,
                        'slippage_applied': t.slippage_applied,
                    }
                    for t in trade_log
                ],
                'wash_sale_stats': self.risk_manager.get_wash_sale_stats(),
                'blacklist_stats': self.risk_manager.get_blacklist_stats(),
                'drawdown_stats': self.risk_manager.get_drawdown_stats(),
                'position_sizing_stats': self.risk_manager.get_position_sizing_stats(),
                'trend_filter_stats': self.risk_manager.get_trend_filter_stats(),
                'position_buffer_stats': self.risk_manager.get_position_buffer_stats(),
                'three_level_defense_stats': defense_stats,
                'frequency_fuse_stats': getattr(self.risk_manager, 'get_frequency 熔断_stats', lambda: {})(),
                'industry_constraint_stats': self.risk_manager.get_industry_constraint_stats(),
                'volatility_squeeze_stats': self.risk_manager.get_volatility_squeeze_stats(),
                'stop_audit_records': stop_audit_records,
                'v55_features': {
                    't_t1_isolation': V55_USE_T1_EXECUTION,
                    'trend_filter_enabled': V55_MA60_FILTER,
                    'ma_filter': 'MA60',
                    'volume_filter_enabled': V55_VOLUME_FILTER_ENABLED,
                    'volatility_squeeze_enabled': V55_VOLATILITY_SQUEEZE_ENABLED,
                    'time_stop_enabled': V55_TIME_STOP_ENABLED,
                    'time_stop_days': V55_TIME_STOP_DAYS,
                    'time_stop_reduce_ratio': V55_TIME_STOP_REDUCE_RATIO,
                    'frequency_fuse': {
                        'weekly_limit': V55_WEEKLY_TRADE_LIMIT,
                        'max_positions': V55_MAX_POSITIONS,
                    },
                    'industry_constraint': {
                        'max_same_industry': 2,
                    },
                    'entry_top_n': V55_ENTRY_TOP_N,
                    'maintain_top_n': V55_MAINTAIN_TOP_N,
                    'volatility_adaptive_positioning': True,
                    'real_execution_price': True,  # V55 核心：真实成交价
                    'hard_stop_count': hard_stop_count,
                    'trailing_profit_count': trailing_profit_count,
                    'ma20_exit_count': ma20_exit_count,
                    'rank_drop_count': rank_drop_count,
                    'stop_loss_count': stop_loss_count,
                    'time_stop_count': time_stop_count,
                    'standard_tier_trades': standard_tier_count,
                    'reduced_tier_trades': reduced_tier_count,
                    'volume_shrunk_trades': volume_shrunk_count,
                    'squeeze_low_trades': squeeze_low_count,
                },
                'performance_targets': {
                    'annual_return_target': V55_ANNUAL_RETURN_TARGET,
                    'max_drawdown_target': V55_MAX_DRAWDOWN_TARGET,
                    'profit_loss_ratio_target': V55_PROFIT_LOSS_RATIO_TARGET,
                    'max_trades_threshold': V55_GLOBAL_TRADE_LIMIT,
                },
                'audit_result': {
                    'trade_count_exceeded': trade_count_exceeded,
                    'trade_count_status': 'FAILED' if trade_count_exceeded else 'PASSED',
                    'mdd_exceeded': mdd_exceeded,
                    'mdd_status': 'FAILED' if mdd_exceeded else 'PASSED',
                    'three_level_defense_audit': {
                        'hard_stop_triggered': hard_stop_count > 0,
                        'trailing_profit_triggered': trailing_profit_count > 0,
                        'ma20_exit_triggered': ma20_exit_count > 0,
                        'time_stop_triggered': time_stop_count > 0,
                        'all_zero_check': hard_stop_count == 0 and trailing_profit_count == 0 and ma20_exit_count == 0,
                    },
                    'real_execution_price_audit': {
                        'total_stop_records': len(stop_audit_records),
                        'valid_records': sum(1 for r in stop_audit_records if r.get('is_valid', True)),
                        'invalid_records': sum(1 for r in stop_audit_records if not r.get('is_valid', True)),
                        'all_valid': stop_audit_valid,
                    },
                },
            }
            
            stats['target_met'] = {
                'annual_return': annual_return >= V55_ANNUAL_RETURN_TARGET,
                'max_drawdown': max_dd <= V55_MAX_DRAWDOWN_TARGET,
                'profit_loss_ratio': profit_loss_ratio >= V55_PROFIT_LOSS_RATIO_TARGET,
                'trade_count': not trade_count_exceeded,
                'mdd_constraint': not mdd_exceeded,
            }
            
            logger.info(f"V55 回测结果:")
            logger.info(f"  总收益：{total_return:.2%}")
            logger.info(f"  年化收益：{annual_return:.2%}")
            logger.info(f"  最大回撤：{max_dd:.2%}")
            logger.info(f"  最大单日亏损：{max_single_day_loss:.2%}")
            logger.info(f"  盈亏比：{profit_loss_ratio:.2f}")
            logger.info(f"  交易次数：{num_trades}")
            logger.info(f"  交易次数审计：{'FAILED' if trade_count_exceeded else 'PASSED'}")
            logger.info(f"  MDD 约束审计：{'FAILED' if mdd_exceeded else 'PASSED'}")
            logger.info(f"  真实成交价审计：{'PASSED' if stop_audit_valid else 'FAILED'}")
            logger.info(f"  三级防御统计:")
            logger.info(f"    - 硬止损触发：{hard_stop_count}")
            logger.info(f"    - 动态止盈触发：{trailing_profit_count}")
            logger.info(f"    - MA20 离场触发：{ma20_exit_count}")
            logger.info(f"    - 位次缓冲离场：{rank_drop_count}")
            logger.info(f"    - 初始止损触发：{stop_loss_count}")
            logger.info(f"    - 时间止损触发：{time_stop_count}")
            logger.info(f"  标准头寸交易：{standard_tier_count}")
            logger.info(f"  降头寸交易：{reduced_tier_count}")
            logger.info(f"  成交量萎缩过滤：{volume_shrunk_count}")
            logger.info(f"  波动率挤压低位入场：{squeeze_low_count}")
            
            # V55 对赌条款：若 MDD > 8% 或交易次数 > 45 次，道歉并承认失败
            if mdd_exceeded or trade_count_exceeded:
                logger.error("=" * 60)
                logger.error("❌ V55 对赌条款触发 - 系统风控设计失败")
                logger.error("=" * 60)
                if mdd_exceeded:
                    logger.error(f"❌ MDD 失败：最大回撤 {max_dd:.2%} > 目标 {V55_MAX_DRAWDOWN_TARGET:.0%}")
                if trade_count_exceeded:
                    logger.error(f"❌ 交易次数失败：交易次数 {num_trades} > 限制 {V55_GLOBAL_TRADE_LIMIT}")
                
                # 查找失败代码行号
                logger.error("失败代码行号参考:")
                logger.error("  - v55_core.py: V55RiskManager.check_exits() - 止损逻辑")
                logger.error("  - v55_core.py: V55RiskManager.can_open_new_position() - 频率熔断")
                logger.error("  - v55_core.py: V55RiskManager.check_industry_constraint() - 行业对冲")
                logger.error("  - v55_engine.py: V55BacktestEngine._handle_exits() - 真实成交价执行")
            
            # V55 逻辑统计审计
            if hard_stop_count == 0 and trailing_profit_count == 0 and ma20_exit_count == 0:
                logger.error("❌ V55 AUDIT FAILED: All three-level defense triggers are 0!")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating final stats: {e}")
            logger.error(traceback.format_exc())
            return self._generate_error_stats(e)


def run_v55_backtest(price_df: pl.DataFrame, start_date: str, end_date: str,
                     index_df: Optional[pl.DataFrame] = None,
                     industry_df: Optional[pl.DataFrame] = None,
                     initial_capital: float = V55_INITIAL_CAPITAL,
                     db=None) -> Dict[str, Any]:
    """V55 回测入口函数"""
    engine = V55BacktestEngine(initial_capital=initial_capital, db=db)
    return engine.run_backtest(price_df, start_date, end_date, index_df, industry_df)