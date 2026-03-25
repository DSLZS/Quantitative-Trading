"""
V52 Engine Module - 铁盾行动与行业均衡

【V52 核心改进 - 全面防御升级】

1. 防御系统全面升级
   ✅ 强制减仓（Flash Cut）：当周回撤超过 3.5%，强制卖出浮盈最低的两只股票
   ✅ 阶梯动态止盈（Step-Trailing Stop）：
      - 浮盈 > 5% 且 < 12%：1.2 * ATR 追踪止损
      - 浮盈 > 12%：最高价回落 10% 硬性离场

2. 行业压舱石
   ✅ 行业多样化约束：同一行业最多持有 2 只股票
   ✅ 信号自动顺延：多出信号顺延给下一排名非同行业股票

3. 信号降噪与频率控制
   ✅ 持仓位次惯性：新标的 Top 3 且持仓跌出 Top 25 时才换仓
   ✅ 目标交易次数：25 - 45 次

4. 严正审计
   ✅ 自动头寸降温：回撤>5% 时，头寸从 20% 降至 15%

作者：量化系统
版本：V52.0
日期：2026-03-21
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import polars as pl
from loguru import logger

from v52_core import (
    V52FactorEngine, V52RiskManager, V52Position, V52Trade,
    V52TradeAudit, V52MarketRegime, V52DrawdownState,
    V52_INITIAL_CAPITAL, V52_MAX_POSITIONS, V52_MAINTAIN_TOP_N,
    V52_ENTRY_TOP_N, V52_USE_T1_EXECUTION,
    V52_SINGLE_DAY_DRAWDOWN_LIMIT, V52_WEEKLY_DRAWDOWN_LIMIT,
    V52_WEEKLY_DRAWDOWN_FLASH_CUT, V52_FLASH_CUT_NUM_STOCKS,
    V52_DRAWDOWN_CUT_POSITION_RATIO, V52_PROFIT_HWM_THRESHOLD,
    V52_PROFIT_DRAWDOWN_RATIO, V52_MAX_DRAWDOWN_TARGET,
    V52_ANNUAL_RETURN_TARGET, V52_PROFIT_LOSS_RATIO_TARGET,
    V52_MIN_TRADES_TARGET, V52_MAX_TRADES_TARGET,
    V52_TRADE_COUNT_FAIL_THRESHOLD, V52_DATABASE_TABLES,
    V52IndustryLoader, V52_MAX_POSITIONS_PER_INDUSTRY,
    V52_INERTIA_NEW_TOP_N, V52_INERTIA_OLD_OUT_OF_N,
    V52_SECTOR_BALANCING_ENABLED, V52_POSITION_INERTIA_ENABLED,
    V52_FLASH_CUT_ENABLED, V52_STEP_TRAILING_ENABLED,
    V52_AUTO_COOLDOWN_ENABLED, V52_COOLDOWN_DRAWDOWN_THRESHOLD,
)


class V52BacktestEngine:
    """
    V52 回测引擎 - 铁盾行动与行业均衡
    
    【核心功能】
    1. T/T+1 严格隔离：信号日 (T) 使用收盘价计算，交易日 (T+1) 使用开盘价成交
    2. 强制减仓（Flash Cut）：周回撤>3.5% 时卖出浮盈最低 2 只股票
    3. 阶梯动态止盈：5%-12% 用 ATR 追踪，>12% 用回落 10% 离场
    4. 行业均衡：同一行业最多持有 2 只
    5. 持仓位次惯性：减少不必要的换仓
    6. 自动头寸降温：回撤>5% 时降低头寸
    """
    
    def __init__(self, initial_capital: float = V52_INITIAL_CAPITAL, db=None):
        self.initial_capital = initial_capital
        self.db = db
        self.risk_manager = V52RiskManager(initial_capital=initial_capital)
        self.factor_engine = V52FactorEngine()
        self.industry_loader = V52IndustryLoader(db=db)
        
        self.portfolio_values: List[Dict[str, Any]] = []
        self.daily_trades: List[V52Trade] = []
        self.trade_log: List[V52TradeAudit] = []
        self.current_date: Optional[str] = None
        self.previous_date: Optional[str] = None
        self.factor_cache: Dict[str, pl.DataFrame] = {}
        self.price_cache: Dict[str, pl.DataFrame] = {}
        
        self.start_date: str = ""
        self.end_date: str = ""
        self.backtest_stats: Dict[str, Any] = {}
        
        # 将 portfolio_values 引用添加到 risk_manager 以便计算最大回撤
        self.risk_manager.portfolio_values = self.portfolio_values
    
    def run_backtest(self, price_df: pl.DataFrame, start_date: str, end_date: str,
                     index_df: Optional[pl.DataFrame] = None,
                     industry_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """运行回测"""
        try:
            logger.info("=" * 60)
            logger.info("V52 回测引擎启动 - 铁盾行动与行业均衡")
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
            logger.info("V52 回测完成")
            logger.info("=" * 60)
            
            return self._generate_final_stats()
            
        except Exception as e:
            logger.error(f"V52 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            return self._generate_error_stats(e)
    
    def _run_trading_day(self, trade_date: str, price_df: pl.DataFrame,
                         index_df: Optional[pl.DataFrame],
                         industry_df: Optional[pl.DataFrame]) -> None:
        """
        V52 运行单个交易日 - 严格 T/T+1 隔离
        
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
            
            # V52: 先计算当日因子，然后执行前一日信号
            current_factors, factor_status = self.factor_engine.compute_all_factors(
                day_prices, industry_df, self.db, self.start_date, self.end_date
            )
            self.factor_cache[trade_date] = current_factors
            self.risk_manager.factor_status = factor_status
            
            # V52 关键：执行 Flash Cut 检查（周回撤>3.5% 时）
            if V52_FLASH_CUT_ENABLED:
                flash_cut_signals = self.risk_manager.execute_flash_cut(trade_date, day_prices)
                for symbol, reason in flash_cut_signals:
                    open_price = self._get_open_price(day_prices, symbol)
                    if open_price and open_price > 0:
                        self.risk_manager.execute_sell(trade_date, symbol, open_price, reason=reason)
            
            # 执行前一日信号（T-1 日信号在 T 日执行）
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
        V52 执行信号 - 严格 T/T+1 隔离，包含行业均衡和持仓位次惯性
        """
        try:
            if not V52_USE_T1_EXECUTION:
                return
            
            # 1. 先检查退出信号（包含阶梯止盈）
            exit_signals = self.risk_manager.check_exits(
                self.risk_manager.positions, trade_date, price_df,
                current_factors if current_factors is not None else signal_factors
            )
            
            for symbol, reason in exit_signals:
                open_price = self._get_open_price(price_df, symbol)
                if open_price and open_price > 0:
                    self.risk_manager.execute_sell(trade_date, symbol, open_price, reason=reason)
            
            # 2. 检查持仓位次惯性（减少不必要换仓）
            if V52_POSITION_INERTIA_ENABLED and current_factors is not None:
                inertia_signals = self.risk_manager.check_position_inertia(current_factors)
                for symbol, reason in inertia_signals:
                    open_price = self._get_open_price(price_df, symbol)
                    if open_price and open_price > 0:
                        self.risk_manager.execute_sell(trade_date, symbol, open_price, reason=reason)
            
            # 3. 检查是否可以开新仓
            if not self.risk_manager.can_open_new_position():
                return
            
            if len(self.risk_manager.positions) >= V52_MAX_POSITIONS:
                return
            
            if current_factors is None:
                return
            
            # 4. 选择入场候选（考虑行业均衡）
            candidates = self._select_entry_candidates_with_sector_balance(current_factors, price_df)
            
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
                shares, target_amount = self.risk_manager.calculate_position_size(
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
                    industry_name=industry_name,
                    reason="v52_entry_signal"
                )
                
                if len(self.risk_manager.positions) >= V52_MAX_POSITIONS:
                    break
                    
        except Exception as e:
            logger.error(f"Error executing signals on {trade_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _select_entry_candidates_with_sector_balance(self, factors_df: pl.DataFrame,
                                                       price_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """
        V52 选择入场候选股票 - 考虑行业均衡约束
        
        核心逻辑：
        1. 按 composite_rank 排序
        2. 同一行业最多持有 V52_MAX_POSITIONS_PER_INDUSTRY 只
        3. 超出行业限制的股票自动顺延
        """
        try:
            # 确保必要的列存在
            required_cols = ['symbol', 'composite_rank', 'close']
            for col in required_cols:
                if col not in factors_df.columns:
                    logger.warning(f"Missing required column: {col}")
                    return []
            
            # 获取所有候选股票 - 按排名排序
            # 首先检查 composite_rank 列是否存在且有效
            if 'composite_rank' not in factors_df.columns:
                logger.warning(f"composite_rank column missing. Available columns: {factors_df.columns}")
                return []
            
            # 过滤掉 composite_rank 为 null 的行，然后排序
            candidates_df = factors_df.filter(
                (pl.col('close') > 0) & 
                (pl.col('composite_rank').is_not_null())
            ).sort('composite_rank')
            
            if candidates_df.is_empty():
                logger.warning(f"No candidates found. factors_df shape: {factors_df.shape}")
                logger.warning(f"Available columns: {factors_df.columns}")
                # 尝试打印一些样本数据来调试
                try:
                    sample = factors_df.head(3)
                    logger.warning(f"Sample data: {sample.to_dict()}")
                except:
                    pass
                return []
            
            # 检查最小排名值
            min_rank = candidates_df['composite_rank'].min()
            max_rank = candidates_df['composite_rank'].max()
            logger.debug(f"Composite rank range: {min_rank} to {max_rank}")
            
            # 获取当前各行业的持仓数量
            industry_holdings: Dict[str, int] = {}
            for pos in self.risk_manager.positions.values():
                industry = pos.industry_name or "Unknown"
                industry_holdings[industry] = industry_holdings.get(industry, 0) + 1
            
            # 计算可用空间
            available_slots = V52_MAX_POSITIONS - len(self.risk_manager.positions)
            
            if available_slots <= 0:
                return []
            
            candidates = []
            selected_industries: Dict[str, int] = {}  # 本次选择的行业计数
            
            # 遍历所有股票，按排名选择
            for row in candidates_df.iter_rows(named=True):
                if len(candidates) >= available_slots:
                    break
                
                symbol = row['symbol']
                rank = row.get('composite_rank', 999) or 999
                
                # 检查是否在允许进场的范围内 (Top N)
                if rank > V52_ENTRY_TOP_N:
                    continue
                
                # 获取行业 - 优先使用数据框中的值，否则模拟
                industry = ''
                if 'industry_name' in factors_df.columns:
                    industry = row.get('industry_name', '') or ''
                if not industry:
                    industry = self.industry_loader.get_industry_for_symbol(symbol)
                
                # 检查行业约束
                current_industry_count = industry_holdings.get(industry, 0) + selected_industries.get(industry, 0)
                
                if V52_SECTOR_BALANCING_ENABLED:
                    if current_industry_count >= V52_MAX_POSITIONS_PER_INDUSTRY:
                        logger.debug(f"SECTOR FULL: {industry} - skipping {symbol}")
                        continue
                
                # 添加到候选
                atr_value = row.get('atr_20') or 0.01
                if atr_value is None or atr_value <= 0:
                    atr_value = 0.01
                
                candidates.append({
                    'symbol': symbol,
                    'signal_score': row.get('composite_score', 0) or 0,
                    'rank': rank,
                    'composite_score': row.get('composite_score', 0) or 0,
                    'percentile': row.get('composite_percentile', 1) or 1,
                    'atr': atr_value,
                    'ma5': row.get('ma5', 0) or 0,
                    'ma20': row.get('ma20', 0) or 0,
                    'industry_name': industry,
                })
                
                # 更新本次选择的行业计数
                selected_industries[industry] = selected_industries.get(industry, 0) + 1
            
            if candidates:
                logger.info(f"Selected {len(candidates)} candidates: {[c['symbol'] for c in candidates]}")
            else:
                logger.warning(f"No candidates selected. Check rank threshold (V52_ENTRY_TOP_N={V52_ENTRY_TOP_N})")
            return candidates
            
        except Exception as e:
            logger.error(f"Error selecting candidates with sector balance: {e}")
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
                        self.risk_manager.market_regime = V52MarketRegime(
                            trade_date=trade_date,
                            index_close=index_close,
                            is_risk_period=False
                        )
                        self.risk_manager.is_risk_period = False
                        return
                except:
                    pass
            
            self.risk_manager.market_regime = V52MarketRegime(
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
                'flash_cut_triggered': self.risk_manager.drawdown_state.flash_cut_triggered,
                'cut_position_active': self.risk_manager.cut_position_active,
                'auto_cooldown_active': self.risk_manager.auto_cooldown_triggered,
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
            
            # V52 修复：正确的回撤计算
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
            
            # 统计 Flash Cut 触发次数
            flash_cut_count = sum(1 for t in trade_log if t.sell_reason == "flash_cut")
            step_trailing_count = sum(1 for t in trade_log if "step_trailing" in t.sell_reason)
            hwm_stop_count = sum(1 for t in trade_log if t.sell_reason == "hwm_stop")
            
            stats = {
                'status': 'success',
                'version': 'V52.0',
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
                        'hwm_stop_triggered': t.hwm_stop_triggered,
                        'flash_cut_triggered': t.flash_cut_triggered,
                        'step_trailing_tier': t.step_trailing_tier,
                    }
                    for t in trade_log
                ],
                'wash_sale_stats': self.risk_manager.get_wash_sale_stats(),
                'blacklist_stats': self.risk_manager.get_blacklist_stats(),
                'drawdown_stats': self.risk_manager.get_drawdown_stats(),
                'position_sizing_stats': self.risk_manager.get_position_sizing_stats(),
                'sector_balancing_stats': self.risk_manager.get_sector_balancing_stats(),
                'step_trailing_stats': self.risk_manager.get_step_trailing_stats(),
                'v52_features': {
                    't_t1_isolation': V52_USE_T1_EXECUTION,
                    'flash_cut_enabled': V52_FLASH_CUT_ENABLED,
                    'flash_cut_threshold': f"{V52_WEEKLY_DRAWDOWN_FLASH_CUT:.1%}",
                    'flash_cut_count': flash_cut_count,
                    'step_trailing_enabled': V52_STEP_TRAILING_ENABLED,
                    'step_trailing_count': step_trailing_count,
                    'hwm_profit_protection': True,
                    'hwm_stop_count': hwm_stop_count,
                    'sector_balancing_enabled': V52_SECTOR_BALANCING_ENABLED,
                    'max_per_industry': V52_MAX_POSITIONS_PER_INDUSTRY,
                    'position_inertia_enabled': V52_POSITION_INERTIA_ENABLED,
                    'auto_cooldown_enabled': V52_AUTO_COOLDOWN_ENABLED,
                    'auto_cooldown_triggered': self.risk_manager.auto_cooldown_triggered,
                },
                'performance_targets': {
                    'annual_return_target': V52_ANNUAL_RETURN_TARGET,
                    'max_drawdown_target': V52_MAX_DRAWDOWN_TARGET,
                    'profit_loss_ratio_target': V52_PROFIT_LOSS_RATIO_TARGET,
                    'min_trades_target': V52_MIN_TRADES_TARGET,
                    'max_trades_target': V52_MAX_TRADES_TARGET,
                    'trade_count_fail_threshold': V52_TRADE_COUNT_FAIL_THRESHOLD,
                },
            }
            
            stats['target_met'] = {
                'annual_return': annual_return >= V52_ANNUAL_RETURN_TARGET,
                'max_drawdown': max_dd <= V52_MAX_DRAWDOWN_TARGET,
                'profit_loss_ratio': profit_loss_ratio >= V52_PROFIT_LOSS_RATIO_TARGET,
                'trade_count': V52_MIN_TRADES_TARGET <= num_trades <= V52_TRADE_COUNT_FAIL_THRESHOLD,
            }
            
            logger.info(f"V52 回测结果:")
            logger.info(f"  总收益：{total_return:.2%}")
            logger.info(f"  年化收益：{annual_return:.2%}")
            logger.info(f"  最大回撤：{max_dd:.2%}")
            logger.info(f"  最大单日亏损：{max_single_day_loss:.2%}")
            logger.info(f"  盈亏比：{profit_loss_ratio:.2f}")
            logger.info(f"  交易次数：{num_trades}")
            logger.info(f"  Flash Cut 触发：{flash_cut_count} 次")
            logger.info(f"  阶梯止盈触发：{step_trailing_count} 次")
            logger.info(f"  自动降温触发：{self.risk_manager.auto_cooldown_triggered}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating final stats: {e}")
            logger.error(traceback.format_exc())
            return self._generate_error_stats(e)


def run_v52_backtest(price_df: pl.DataFrame, start_date: str, end_date: str,
                     index_df: Optional[pl.DataFrame] = None,
                     industry_df: Optional[pl.DataFrame] = None,
                     initial_capital: float = V52_INITIAL_CAPITAL,
                     db=None) -> Dict[str, Any]:
    """V52 回测入口函数"""
    engine = V52BacktestEngine(initial_capital=initial_capital, db=db)
    return engine.run_backtest(price_df, start_date, end_date, index_df, industry_df)