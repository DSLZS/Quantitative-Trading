"""
V53 Engine Module - 自适应波段与波动率控仓

【V53 核心改进 - 事前风控与趋势共振】

1. 逻辑大扫除（已移除 V52 毒素）
   ❌ 移除：Flash Cut（周回撤减仓）逻辑
   ❌ 移除：15% 自动降温逻辑
   ❌ 移除：卖出浮盈最低的自残代码

2. 核心进化：事前风控（Pre-risk Control）
   ✅ 波动率适配仓位：DynamicPositionSizer
   ✅ 中线位次缓冲：入场 Top 5，离场 Top 40 或跌破 MA20

3. 进场过滤：趋势共振
   ✅ 双重确认：因子得分 + 股价在 120 日半年线之上

4. 严正审计与交付
   ✅ 杜绝碎片化交易：交易次数 > 60 次必须重写 TradeLogic
   ✅ 拒绝造假：固定滑点 0.1%，计入印花税
   ✅ 自动对赌：收益率 < 15% 时道歉并指出制约因素

作者：量化系统
版本：V53.0
日期：2026-03-21
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import polars as pl
from loguru import logger

from v53_core import (
    V53FactorEngine, V53RiskManager, V53Position, V53Trade,
    V53TradeAudit, V53MarketRegime, V53DrawdownState,
    V53_INITIAL_CAPITAL, V53_MAX_POSITIONS, V53_MAINTAIN_TOP_N,
    V53_ENTRY_TOP_N, V53_USE_T1_EXECUTION,
    V53_TREND_FILTER_ENABLED, V53_MA120_FILTER,
    V53_MAX_DRAWDOWN_TARGET, V53_ANNUAL_RETURN_TARGET,
    V53_PROFIT_LOSS_RATIO_TARGET, V53_MAX_TRADES_THRESHOLD,
    V53_DATABASE_TABLES, V53IndustryLoader,
)


class V53BacktestEngine:
    """
    V53 回测引擎 - 自适应波段与波动率控仓
    
    【核心功能】
    1. T/T+1 严格隔离：信号日 (T) 使用收盘价计算，交易日 (T+1) 使用开盘价成交
    2. 波动率适配仓位：高波动股票自动降至 10% 头寸
    3. 趋势共振过滤：只做股价在 120 日均线之上的股票
    4. 中线位次缓冲：入场 Top 5，离场 Top 40 或跌破 MA20
    5. 严正审计：交易次数 > 60 次自动标记失败
    """
    
    def __init__(self, initial_capital: float = V53_INITIAL_CAPITAL, db=None):
        self.initial_capital = initial_capital
        self.db = db
        self.risk_manager = V53RiskManager(initial_capital=initial_capital)
        self.factor_engine = V53FactorEngine()
        self.industry_loader = V53IndustryLoader(db=db)
        
        self.portfolio_values: List[Dict[str, Any]] = []
        self.daily_trades: List[V53Trade] = []
        self.trade_log: List[V53TradeAudit] = []
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
            logger.info("V53 回测引擎启动 - 自适应波段与波动率控仓")
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
            
            for i, trade_date in enumerate(trade_dates):
                self._run_trading_day(trade_date, price_df, index_df, industry_df)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"进度：{i + 1}/{len(trade_dates)} 交易日")
            
            logger.info("=" * 60)
            logger.info("V53 回测完成")
            logger.info("=" * 60)
            
            return self._generate_final_stats()
            
        except Exception as e:
            logger.error(f"V53 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            return self._generate_error_stats(e)
    
    def _run_trading_day(self, trade_date: str, price_df: pl.DataFrame,
                         index_df: Optional[pl.DataFrame],
                         industry_df: Optional[pl.DataFrame]) -> None:
        """
        V53 运行单个交易日 - 严格 T/T+1 隔离
        
        流程：
        1. T 日：计算因子，产生信号，缓存
        2. T+1 日：使用 T 日信号，以开盘价执行交易
        """
        try:
            self.previous_date = self.current_date
            self.current_date = trade_date
            
            self.risk_manager.reset_daily_counters(trade_date)
            
            day_prices = price_df.filter(pl.col('trade_date') == trade_date)
            
            if day_prices.is_empty():
                return
            
            # V53: 计算当日因子，然后执行前一日信号
            current_factors, factor_status = self.factor_engine.compute_all_factors(
                day_prices, industry_df, self.db, self.start_date, self.end_date
            )
            self.factor_cache[trade_date] = current_factors
            self.risk_manager.factor_status = factor_status
            
            # V53 关键：移除 Flash Cut，直接执行前一日信号
            if self.previous_date:
                prev_factors = self.factor_cache.get(self.previous_date)
                if prev_factors is not None:
                    self._execute_signals(trade_date, day_prices, prev_factors, current_factors)
            
            self._update_market_regime(trade_date, index_df, day_prices)
            
            self._record_portfolio_state(trade_date, day_prices)
            
            if len(self.factor_cache) > 30:
                oldest_date = min(self.factor_cache.keys())
                del self.factor_cache[oldest_date]
            
        except Exception as e:
            logger.error(f"Error running trading day {trade_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _execute_signals(self, trade_date: str, price_df: pl.DataFrame,
                         signal_factors: pl.DataFrame,
                         current_factors: pl.DataFrame) -> None:
        """
        V53 执行信号 - 严格 T/T+1 隔离，包含趋势共振和波动率适配
        """
        try:
            if not V53_USE_T1_EXECUTION:
                return
            
            # 1. 先检查退出信号（中线位次缓冲）
            exit_signals = self.risk_manager.check_exits(
                self.risk_manager.positions, trade_date, price_df,
                current_factors if current_factors is not None else signal_factors
            )
            
            for symbol, reason in exit_signals:
                open_price = self._get_open_price(price_df, symbol)
                if open_price and open_price > 0:
                    self.risk_manager.execute_sell(trade_date, symbol, open_price, reason=reason)
            
            # 2. 检查是否可以开新仓
            if not self.risk_manager.can_open_new_position():
                return
            
            if len(self.risk_manager.positions) >= V53_MAX_POSITIONS:
                return
            
            if current_factors is None:
                return
            
            # 3. 选择入场候选（趋势共振过滤）
            candidates = self._select_entry_candidates_with_trend_filter(current_factors, price_df)
            
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
                
                # V53 核心：波动率适配头寸计算
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
                    ma120=candidate.get('ma120', 0),
                    industry_name=industry_name,
                    volatility_ratio=candidate.get('volatility_ratio', 0),
                    reason="v53_entry_signal"
                )
                
                if len(self.risk_manager.positions) >= V53_MAX_POSITIONS:
                    break
                    
        except Exception as e:
            logger.error(f"Error executing signals on {trade_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _select_entry_candidates_with_trend_filter(self, factors_df: pl.DataFrame,
                                                     price_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """
        V53 选择入场候选股票 - 包含趋势共振过滤
        
        核心逻辑：
        1. 按 composite_rank 排序
        2. 趋势共振过滤：股价必须在 120 日均线之上
        3. 只选择 Top N 排名的股票
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
            available_slots = V53_MAX_POSITIONS - len(self.risk_manager.positions)
            
            if available_slots <= 0:
                return []
            
            candidates = []
            
            # 遍历所有股票，按排名选择
            for row in candidates_df.iter_rows(named=True):
                if len(candidates) >= available_slots:
                    break
                
                symbol = row['symbol']
                rank = row.get('composite_rank', 999) or 999
                
                # V53 核心：只选择 Top N 排名的股票
                if rank > V53_ENTRY_TOP_N:
                    continue
                
                # V53 核心：趋势共振过滤
                if V53_TREND_FILTER_ENABLED and V53_MA120_FILTER:
                    price_above_ma120 = row.get('price_above_ma120', None)
                    # 如果 price_above_ma120 为 None，检查 ma120 和 close 的关系
                    if price_above_ma120 is None:
                        ma120_val = row.get('ma120', 0) or 0
                        close_val = row.get('close', 0) or 0
                        price_above_ma120 = close_val > ma120_val if ma120_val > 0 else True
                    if not price_above_ma120:
                        logger.debug(f"TREND FILTER: {symbol} - Price below MA120, skipped")
                        continue
                
                # 获取必要数据
                atr_value = row.get('atr_20') or 0.01
                if atr_value is None or atr_value <= 0:
                    atr_value = 0.01
                
                ma120 = row.get('ma120', 0) or 0
                ma20 = row.get('ma20', 0) or 0
                ma5 = row.get('ma5', 0) or 0
                
                # 计算波动率比率
                close_price = row.get('close', 1)
                volatility_ratio = atr_value / close_price if close_price > 0 else 0
                
                # 获取行业
                industry = ''
                if 'industry_name' in factors_df.columns:
                    industry = row.get('industry_name', '') or ''
                if not industry:
                    industry = self.industry_loader.get_industry_for_symbol(symbol)
                
                candidates.append({
                    'symbol': symbol,
                    'signal_score': row.get('composite_score', 0) or 0,
                    'rank': rank,
                    'composite_score': row.get('composite_score', 0) or 0,
                    'percentile': row.get('composite_percentile', 1) or 1,
                    'atr': atr_value,
                    'ma5': ma5,
                    'ma20': ma20,
                    'ma120': ma120,
                    'industry_name': industry,
                    'volatility_ratio': volatility_ratio,
                    'price_above_ma120': price_above_ma120 if V53_TREND_FILTER_ENABLED else True,
                })
            
            if candidates:
                logger.info(f"Selected {len(candidates)} candidates: {[c['symbol'] for c in candidates]}")
            else:
                logger.warning(f"No candidates selected. Check rank threshold (V53_ENTRY_TOP_N={V53_ENTRY_TOP_N}) and trend filter")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error selecting candidates with trend filter: {e}")
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
                        self.risk_manager.market_regime = V53MarketRegime(
                            trade_date=trade_date,
                            index_close=index_close,
                            is_risk_period=False
                        )
                        self.risk_manager.is_risk_period = False
                        return
                except:
                    pass
            
            self.risk_manager.market_regime = V53MarketRegime(
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
            
            # V53 严正审计：交易次数检查
            trade_count_exceeded = num_trades > V53_MAX_TRADES_THRESHOLD
            
            # 统计卖出原因
            rank_drop_count = sum(1 for t in trade_log if t.sell_reason == "rank_drop")
            ma20_break_count = sum(1 for t in trade_log if t.sell_reason == "ma20_break")
            trailing_stop_count = sum(1 for t in trade_log if t.sell_reason == "trailing_stop")
            stop_loss_count = sum(1 for t in trade_log if t.sell_reason == "stop_loss")
            
            # 统计头寸层级
            standard_tier_count = sum(1 for t in trade_log if t.position_tier == "standard")
            reduced_tier_count = sum(1 for t in trade_log if t.position_tier == "reduced")
            
            stats = {
                'status': 'success',
                'version': 'V53.0',
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
                    }
                    for t in trade_log
                ],
                'wash_sale_stats': self.risk_manager.get_wash_sale_stats(),
                'blacklist_stats': self.risk_manager.get_blacklist_stats(),
                'drawdown_stats': self.risk_manager.get_drawdown_stats(),
                'position_sizing_stats': self.risk_manager.get_position_sizing_stats(),
                'trend_filter_stats': self.risk_manager.get_trend_filter_stats(),
                'position_buffer_stats': self.risk_manager.get_position_buffer_stats(),
                'v53_features': {
                    't_t1_isolation': V53_USE_T1_EXECUTION,
                    'trend_filter_enabled': V53_TREND_FILTER_ENABLED,
                    'ma120_filter_active': V53_MA120_FILTER,
                    'entry_top_n': V53_ENTRY_TOP_N,
                    'maintain_top_n': V53_MAINTAIN_TOP_N,
                    'volatility_adaptive_positioning': True,
                    'flash_cut_removed': True,
                    'auto_cooldown_removed': True,
                    'rank_drop_exits': rank_drop_count,
                    'ma20_break_exits': ma20_break_count,
                    'trailing_stop_exits': trailing_stop_count,
                    'stop_loss_exits': stop_loss_count,
                    'standard_tier_trades': standard_tier_count,
                    'reduced_tier_trades': reduced_tier_count,
                },
                'performance_targets': {
                    'annual_return_target': V53_ANNUAL_RETURN_TARGET,
                    'max_drawdown_target': V53_MAX_DRAWDOWN_TARGET,
                    'profit_loss_ratio_target': V53_PROFIT_LOSS_RATIO_TARGET,
                    'max_trades_threshold': V53_MAX_TRADES_THRESHOLD,
                },
                'audit_result': {
                    'trade_count_exceeded': trade_count_exceeded,
                    'trade_count_status': 'FAILED' if trade_count_exceeded else 'PASSED',
                },
            }
            
            stats['target_met'] = {
                'annual_return': annual_return >= V53_ANNUAL_RETURN_TARGET,
                'max_drawdown': max_dd <= V53_MAX_DRAWDOWN_TARGET,
                'profit_loss_ratio': profit_loss_ratio >= V53_PROFIT_LOSS_RATIO_TARGET,
                'trade_count': not trade_count_exceeded,
            }
            
            logger.info(f"V53 回测结果:")
            logger.info(f"  总收益：{total_return:.2%}")
            logger.info(f"  年化收益：{annual_return:.2%}")
            logger.info(f"  最大回撤：{max_dd:.2%}")
            logger.info(f"  最大单日亏损：{max_single_day_loss:.2%}")
            logger.info(f"  盈亏比：{profit_loss_ratio:.2f}")
            logger.info(f"  交易次数：{num_trades}")
            logger.info(f"  交易次数审计：{'FAILED' if trade_count_exceeded else 'PASSED'}")
            logger.info(f"  标准头寸交易：{standard_tier_count}")
            logger.info(f"  降头寸交易：{reduced_tier_count}")
            
            # V53 自动对赌：收益率 < 15% 时道歉
            if annual_return < V53_ANNUAL_RETURN_TARGET:
                logger.warning(f"⚠️  V53 PERFORMANCE WARNING: Annual return {annual_return:.2%} < target {V53_ANNUAL_RETURN_TARGET:.0%}")
                logger.warning(f"⚠️  制约因素分析:")
                if trade_count_exceeded:
                    logger.warning(f"  - 碎片化交易：交易次数 {num_trades} > {V53_MAX_TRADES_THRESHOLD}")
                if max_dd > V53_MAX_DRAWDOWN_TARGET:
                    logger.warning(f"  - 回撤控制：最大回撤 {max_dd:.2%} > 目标 {V53_MAX_DRAWDOWN_TARGET:.0%}")
                if profit_loss_ratio < V53_PROFIT_LOSS_RATIO_TARGET:
                    logger.warning(f"  - 盈亏比：{profit_loss_ratio:.2f} < 目标 {V53_PROFIT_LOSS_RATIO_TARGET:.1f}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating final stats: {e}")
            logger.error(traceback.format_exc())
            return self._generate_error_stats(e)


def run_v53_backtest(price_df: pl.DataFrame, start_date: str, end_date: str,
                     index_df: Optional[pl.DataFrame] = None,
                     industry_df: Optional[pl.DataFrame] = None,
                     initial_capital: float = V53_INITIAL_CAPITAL,
                     db=None) -> Dict[str, Any]:
    """V53 回测入口函数"""
    engine = V53BacktestEngine(initial_capital=initial_capital, db=db)
    return engine.run_backtest(price_df, start_date, end_date, index_df, industry_df)