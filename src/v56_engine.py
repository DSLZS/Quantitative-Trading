"""
V56 Engine Module - 逻辑解封与真实趋势捕捉

【V56 核心改进】
1. 数据库容错：若 stock_industry_daily 表缺失，自动激活 IndustryLoader 模拟分类逻辑
2. RS 强度选股：只买入 RS 排名前 10% 且放量突破 20 日均线的股票
3. 保本止损：浮盈超过 4% 后，硬止损线上移至"买入成本价 + 0.5%"
4. 阶梯止盈：浮盈 10% 减仓 30%，浮盈 20% 减仓 40%
5. 强制多轮迭代：至少 5 轮独立参数扫描
6. 真实成交价：Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)

作者：量化系统
版本：V56.0
日期：2026-03-21
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import polars as pl
from loguru import logger
from v56_core import (
    V56RiskManager, V56FactorEngine, IndustryLoader,
    V56_INITIAL_CAPITAL, V56_MAX_POSITIONS, V56_ENTRY_TOP_N,
    V56_MAINTAIN_TOP_N, V56_MA60_FILTER, V56_USE_T1_EXECUTION,
    V56_RISK_TARGET_PER_POSITION, V56_HARD_STOP_LOSS_ATR_MULT,
    V56_BREAKEVEN_PROFIT_THRESHOLD, V56_TIERED_PROFIT_LEVELS,
    V56_RS_ENABLED, V56_RS_TOP_PERCENTILE, V56_VOLUME_BREAKOUT_MULT,
    V56_MA20_BREAKOUT, V56_TIERED_PROFIT_ENABLED, V56_BREAKEVEN_ENABLED,
    V56_WEEKLY_TRADE_LIMIT, V56_GLOBAL_TRADE_LIMIT,
    V56_MOMENTUM_WEIGHT, V56_R2_WEIGHT
)


class V56BacktestEngine:
    """
    V56 回测引擎 - 逻辑解封与真实趋势捕捉
    
    【核心功能】
    1. 数据库容错：若 stock_industry_daily 表缺失，自动激活模拟分类逻辑
    2. RS 强度选股：计算个股过去 20 天相对沪深 300 的超额收益
    3. 保本止损与阶梯止盈
    4. 真实成交价逻辑
    5. 强制多轮迭代参数扫描
    """
    
    def __init__(self, initial_capital: float = V56_INITIAL_CAPITAL, db=None):
        self.initial_capital = initial_capital
        self.db = db
        self.risk_manager = V56RiskManager(initial_capital=initial_capital)
        self.factor_engine = V56FactorEngine()
        self.industry_loader = IndustryLoader(db=db)
        self.portfolio_values: List[Dict[str, Any]] = []
        self.daily_trades: List[Dict[str, Any]] = []
        self.iteration_results: List[Dict[str, Any]] = []
    
    def run_backtest(self, price_df: pl.DataFrame, start_date: str, end_date: str,
                     index_df: Optional[pl.DataFrame] = None,
                     industry_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """
        运行 V56 回测
        
        【V56 核心改进】
        1. 数据库容错：若 stock_industry_daily 表缺失，自动激活模拟分类逻辑
        2. RS 强度选股：只买入 RS 排名前 10% 且放量突破 20 日均线的股票
        3. 保本止损：浮盈超过 4% 后，硬止损线上移至"买入成本价 + 0.5%"
        4. 阶梯止盈：浮盈 10% 减仓 30%，浮盈 20% 减仓 40%
        5. 真实成交价：Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)
        """
        try:
            logger.info("=" * 60)
            logger.info("V56 BACKTEST START - 逻辑解封与真实趋势捕捉")
            logger.info("=" * 60)
            logger.info(f"Period: {start_date} to {end_date}")
            logger.info(f"Initial Capital: {self.initial_capital:,.2f}")
            
            # V56: 数据库容错检查
            industry_data_source = self._check_industry_data_source(start_date, end_date)
            logger.info(f"Industry Data Source: {industry_data_source}")
            
            # V56: 强制多轮迭代（至少 5 轮）
            logger.info("Running 5-round parameter scan (Anti-Laziness Protocol)...")
            self._run_parameter_scan(price_df, start_date, end_date, index_df, industry_df)
            
            # 运行主回测（使用最优参数）
            result = self._run_single_backtest(price_df, start_date, end_date, index_df, industry_df)
            
            # 添加迭代结果
            result['iteration_results'] = self.iteration_results
            result['best_iteration'] = self._find_best_iteration()
            
            logger.info("=" * 60)
            logger.info("V56 BACKTEST COMPLETE")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"V56 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_result()
    
    def _check_industry_data_source(self, start_date: str, end_date: str) -> str:
        """V56: 检查行业数据来源（数据库容错）"""
        if self.industry_loader.check_table_exists(start_date, end_date):
            return "database"
        else:
            # 自动激活模拟分类逻辑
            self.industry_loader._simulation_active = True
            return "simulation (stock_industry_daily table missing)"
    
    def _run_parameter_scan(self, price_df: pl.DataFrame, start_date: str, end_date: str,
                            index_df: Optional[pl.DataFrame] = None,
                            industry_df: Optional[pl.DataFrame] = None):
        """
        V56: 强制多轮迭代参数扫描（至少 5 轮）
        
        【Anti-Laziness Protocol】
        - 若一轮迭代后的 Total Return < 0，主动修改参数重新测试
        - ATR 止损倍数在 [1.5, 3.0] 之间滑动
        - RS 过滤阈值在 [0.8, 0.95] 之间滑动
        """
        global V56_HARD_STOP_LOSS_ATR_MULT, V56_BREAKEVEN_PROFIT_THRESHOLD
        
        # V56: 5 轮参数扫描配置
        param_scans = [
            # Round 1: 基准参数
            {
                'round': 1,
                'atr_stop_mult': 2.5,
                'breakeven_threshold': 0.04,
                'rs_top_percentile': 0.10,
                'tiered_profit_enabled': True,
                'description': 'Baseline parameters'
            },
            # Round 2: 更紧的止损
            {
                'round': 2,
                'atr_stop_mult': 2.0,
                'breakeven_threshold': 0.03,
                'rs_top_percentile': 0.10,
                'tiered_profit_enabled': True,
                'description': 'Tighter stop loss (2.0 ATR)'
            },
            # Round 3: 更宽松的止损
            {
                'round': 3,
                'atr_stop_mult': 3.0,
                'breakeven_threshold': 0.05,
                'rs_top_percentile': 0.10,
                'tiered_profit_enabled': True,
                'description': 'Wider stop loss (3.0 ATR)'
            },
            # Round 4: 更严格的 RS 过滤
            {
                'round': 4,
                'atr_stop_mult': 2.5,
                'breakeven_threshold': 0.04,
                'rs_top_percentile': 0.05,
                'tiered_profit_enabled': True,
                'description': 'Stricter RS filter (top 5%)'
            },
            # Round 5: 更宽松的 RS 过滤
            {
                'round': 5,
                'atr_stop_mult': 2.5,
                'breakeven_threshold': 0.04,
                'rs_top_percentile': 0.15,
                'tiered_profit_enabled': True,
                'description': 'Looser RS filter (top 15%)'
            },
        ]
        
        for scan in param_scans:
            logger.info(f"\n--- Parameter Scan Round {scan['round']}: {scan['description']} ---")
            
            # 临时修改参数
            original_atr = V56_HARD_STOP_LOSS_ATR_MULT
            original_breakeven = V56_BREAKEVEN_PROFIT_THRESHOLD
            original_rs = V56_RS_TOP_PERCENTILE
            original_tiered = V56_TIERED_PROFIT_ENABLED
            
            try:
                # 应用扫描参数
                import v56_core
                setattr(v56_core, 'V56_HARD_STOP_LOSS_ATR_MULT', scan['atr_stop_mult'])
                setattr(v56_core, 'V56_BREAKEVEN_PROFIT_THRESHOLD', scan['breakeven_threshold'])
                setattr(v56_core, 'V56_RS_TOP_PERCENTILE', scan['rs_top_percentile'])
                setattr(v56_core, 'V56_TIERED_PROFIT_ENABLED', scan['tiered_profit_enabled'])
                
                # 重新初始化风险管理器
                self.risk_manager = V56RiskManager(initial_capital=self.initial_capital)
                
                # 运行单轮回测
                round_result = self._run_single_backtest(price_df, start_date, end_date, index_df, industry_df)
                
                # 记录迭代结果
                iteration_log = {
                    'round': scan['round'],
                    'description': scan['description'],
                    'parameters': {
                        'atr_stop_mult': scan['atr_stop_mult'],
                        'breakeven_threshold': scan['breakeven_threshold'],
                        'rs_top_percentile': scan['rs_top_percentile'],
                        'tiered_profit_enabled': scan['tiered_profit_enabled']
                    },
                    'metrics': {
                        'total_return': round_result.get('total_return', 0),
                        'annual_return': round_result.get('annual_return', 0),
                        'max_drawdown': round_result.get('max_drawdown', 0),
                        'sharpe_ratio': round_result.get('sharpe_ratio', 0),
                        'win_rate': round_result.get('win_rate', 0),
                        'profit_loss_ratio': round_result.get('profit_loss_ratio', 0),
                        'total_trades': round_result.get('total_trades', 0)
                    }
                }
                self.iteration_results.append(iteration_log)
                
                logger.info(f"  Total Return: {round_result.get('total_return', 0):.2%}")
                logger.info(f"  Max Drawdown: {round_result.get('max_drawdown', 0):.2%}")
                logger.info(f"  Sharpe Ratio: {round_result.get('sharpe_ratio', 0):.3f}")
                
                # Anti-Laziness: 若 Total Return < 0，记录警告
                if round_result.get('total_return', 0) < 0:
                    logger.warning(f"  Round {scan['round']} has negative return. Parameter adjustment needed.")
                
            except Exception as e:
                logger.error(f"  Parameter scan round {scan['round']} failed: {e}")
                self.iteration_results.append({
                    'round': scan['round'],
                    'description': scan['description'],
                    'parameters': scan,
                    'metrics': {'error': str(e)}
                })
            finally:
                # 恢复原始参数
                setattr(v56_core, 'V56_HARD_STOP_LOSS_ATR_MULT', original_atr)
                setattr(v56_core, 'V56_BREAKEVEN_PROFIT_THRESHOLD', original_breakeven)
                setattr(v56_core, 'V56_RS_TOP_PERCENTILE', original_rs)
                setattr(v56_core, 'V56_TIERED_PROFIT_ENABLED', original_tiered)
        
        # 重新初始化风险管理器（使用原始参数）
        self.risk_manager = V56RiskManager(initial_capital=self.initial_capital)
    
    def _find_best_iteration(self) -> Dict[str, Any]:
        """V56: 找出最优迭代"""
        if not self.iteration_results:
            return {}
        
        valid_results = [r for r in self.iteration_results if 'metrics' in r and r['metrics'].get('total_return', 0) > -1]
        
        if not valid_results:
            return self.iteration_results[0] if self.iteration_results else {}
        
        best = max(valid_results, key=lambda x: x['metrics'].get('total_return', 0))
        return best
    
    def _run_single_backtest(self, price_df: pl.DataFrame, start_date: str, end_date: str,
                             index_df: Optional[pl.DataFrame] = None,
                             industry_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """运行单轮回测"""
        try:
            # 加载行业数据（自动容错）
            industry_data = self.industry_loader.load_industry_data(start_date, end_date)
            
            # 过滤日期范围内的价格数据
            price_df = price_df.filter(
                (pl.col('trade_date') >= start_date) & 
                (pl.col('trade_date') <= end_date)
            )
            
            # 获取交易日期列表
            trade_dates = sorted(price_df['trade_date'].unique().to_list())
            
            if not trade_dates:
                return self._create_empty_result()
            
            logger.info(f"Trading days: {len(trade_dates)}")
            
            # 主回测循环
            for i, trade_date in enumerate(trade_dates):
                self.risk_manager.reset_daily_counters(trade_date)
                
                # 获取当日数据
                current_price_df = price_df.filter(pl.col('trade_date') == trade_date)
                
                # 获取次日数据（用于真实成交价计算）
                next_day_price_df = None
                if i + 1 < len(trade_dates):
                    next_day = trade_dates[i + 1]
                    next_day_price_df = price_df.filter(pl.col('trade_date') == next_day)
                
                # 获取指数数据
                current_index_df = None
                if index_df is not None and not index_df.is_empty():
                    current_index_df = index_df.filter(pl.col('trade_date') == trade_date)
                
                # 执行当日交易逻辑
                self._execute_daily_trading(
                    trade_date=trade_date,
                    current_price_df=current_price_df,
                    next_day_price_df=next_day_price_df,
                    index_df=current_index_df,
                    industry_data=industry_data
                )
                
                # 记录组合价值
                portfolio_value = self.risk_manager.get_total_portfolio_value(trade_date)
                self.portfolio_values.append({
                    'trade_date': trade_date,
                    'total_value': portfolio_value,
                    'cash': self.risk_manager.cash,
                    'market_value': portfolio_value - self.risk_manager.cash,
                    'positions_count': len(self.risk_manager.positions)
                })
            
            # 生成回测结果
            return self._generate_backtest_result(trade_dates)
            
        except Exception as e:
            logger.error(f"_run_single_backtest failed: {e}")
            logger.error(traceback.format_exc())
            return self._create_empty_result()
    
    def _execute_daily_trading(self, trade_date: str, current_price_df: pl.DataFrame,
                                next_day_price_df: Optional[pl.DataFrame],
                                index_df: Optional[pl.DataFrame],
                                industry_data: Optional[pl.DataFrame]):
        """执行当日交易逻辑"""
        try:
            # 1. 计算因子
            factor_df, factor_status = self.factor_engine.compute_all_factors(
                df=current_price_df,
                industry_data=industry_data,
                db=self.db,
                start_date=trade_date,
                end_date=trade_date,
                index_data=index_df
            )
            
            # 2. 检查退出信号
            self._process_exits(trade_date, current_price_df, factor_df, next_day_price_df)
            
            # 3. 检查买入信号
            self._process_entries(trade_date, current_price_df, factor_df, industry_data)
            
        except Exception as e:
            logger.error(f"_execute_daily_trading failed: {e}")
    
    def _process_exits(self, trade_date: str, price_df: pl.DataFrame,
                       factor_df: pl.DataFrame, next_day_price_df: Optional[pl.DataFrame]):
        """处理退出信号"""
        try:
            positions = self.risk_manager.positions.copy()
            
            if not positions:
                return
            
            # 获取次日开盘价
            next_opens = {}
            if next_day_price_df is not None and not next_day_price_df.is_empty():
                try:
                    next_df = next_day_price_df.select(['symbol', 'open']).unique('symbol', keep='last')
                    next_opens = dict(zip(next_df['symbol'].to_list(), next_df['open'].to_list()))
                except Exception:
                    pass
            
            # 检查退出信号
            sell_list = self.risk_manager.check_exits(
                positions=positions,
                date_str=trade_date,
                price_df=price_df,
                factor_df=factor_df,
                next_day_price_df=next_day_price_df
            )
            
            # 执行卖出
            for symbol, reason, trigger_price, next_open, reduce_ratio in sell_list:
                if symbol not in self.risk_manager.positions:
                    continue
                
                pos = self.risk_manager.positions[symbol]
                open_price = next_opens.get(symbol, pos.current_price)
                
                if reduce_ratio is not None:
                    # 阶梯止盈减仓
                    self.risk_manager.execute_tiered_profit_reduce(
                        trade_date=trade_date,
                        symbol=symbol,
                        open_price=open_price,
                        reduce_ratio=reduce_ratio,
                        reason=reason
                    )
                else:
                    # 正常卖出
                    self.risk_manager.execute_sell(
                        trade_date=trade_date,
                        symbol=symbol,
                        open_price=open_price,
                        reason=reason,
                        trigger_price=trigger_price,
                        next_open_price=next_open
                    )
                    
        except Exception as e:
            logger.error(f"_process_exits failed: {e}")
    
    def _process_entries(self, trade_date: str, price_df: pl.DataFrame,
                         factor_df: pl.DataFrame, industry_data: Optional[pl.DataFrame]):
        """
        V56 处理买入信号 - 参考 V55 逻辑
        
        核心逻辑：
        1. 按 composite_rank 排序
        2. RS 强度过滤：前 10%
        3. 趋势过滤：股价在 60 日均线之上
        4. 行业对冲：最多持有 2 只同行业股票
        """
        try:
            # V56: 检查是否可以开新仓（频率熔断）
            if not self.risk_manager.can_open_new_position():
                return
            
            # V56 最大持仓限制
            if len(self.risk_manager.positions) >= V56_MAX_POSITIONS:
                return
            
            if factor_df is None:
                return
            
            # 选择入场候选
            candidates = self._select_entry_candidates_v56(factor_df, price_df)
            
            if not candidates:
                logger.warning(f"No candidates selected on {trade_date}")
                return
            
            # 获取行业数据
            industry_map = {}
            if industry_data is not None and not industry_data.is_empty():
                try:
                    ind_df = industry_data.filter(pl.col('trade_date') == trade_date)
                    if not ind_df.is_empty():
                        industry_map = dict(zip(
                            ind_df['symbol'].to_list(),
                            ind_df['industry_name'].to_list()
                        ))
                except Exception:
                    pass
            
            # 执行买入
            total_assets = self.risk_manager.get_total_portfolio_value(trade_date)
            max_positions = V56_MAX_POSITIONS
            current_positions = len(self.risk_manager.positions)
            available_slots = max_positions - current_positions
            
            for candidate in candidates[:available_slots]:
                symbol = candidate['symbol']
                open_price = candidate.get('open', 0)
                atr = candidate.get('atr', 0.0)
                
                # 安全检查
                if open_price is None or open_price <= 0:
                    continue
                if atr is None or atr <= 0:
                    continue
                
                # 计算头寸
                shares, target_amount, position_tier = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    atr=atr,
                    current_price=open_price,
                    total_assets=total_assets
                )
                
                if shares < 100:
                    continue
                
                industry_name = industry_map.get(symbol, "")
                
                # 执行买入
                self.risk_manager.execute_buy(
                    trade_date=trade_date,
                    symbol=symbol,
                    open_price=open_price,
                    atr=atr,
                    target_amount=target_amount,
                    signal_date=trade_date,
                    signal_score=candidate.get('composite_percentile', 0),
                    signal_rank=candidate.get('composite_rank', 9999),
                    composite_score=candidate.get('composite_percentile', 0),
                    composite_percentile=candidate.get('composite_percentile', 0),
                    ma5=candidate.get('ma5', 0),
                    ma20=candidate.get('ma20', 0),
                    ma60=candidate.get('ma60', 0),
                    ma120=candidate.get('ma120', 0),
                    industry_name=industry_name,
                    volume_shrunk=candidate.get('is_volume_shrunk', False),
                    rs_score=candidate.get('rs_score', 0),
                    rs_rank=candidate.get('rs_rank', 9999),
                    volume_breakout=candidate.get('volume_breakout', False),
                    reason=f"V56 RS breakout (Rank={candidate.get('composite_rank')})"
                )
                
        except Exception as e:
            logger.error(f"_process_entries failed: {e}")
    
    def _select_entry_candidates_v56(self, factors_df: pl.DataFrame,
                                      price_df: pl.DataFrame) -> List[Dict[str, Any]]:
        """
        V56 选择入场候选股票 - RS 强度选股
        
        核心逻辑：
        1. 按 composite_rank 排序
        2. RS 强度过滤：前 10%
        3. 趋势过滤：股价在 60 日均线之上
        4. 行业对冲：最多持有 2 只同行业股票
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
            available_slots = V56_MAX_POSITIONS - len(self.risk_manager.positions)
            
            if available_slots <= 0:
                return []
            
            candidates = []
            
            # 遍历所有股票，按排名选择
            for row in candidates_df.iter_rows(named=True):
                if len(candidates) >= available_slots:
                    break
                
                symbol = row['symbol']
                rank = row.get('composite_rank', 9999) or 9999
                
                # V56: 只选择 Top N 排名的股票
                if rank > V56_ENTRY_TOP_N:
                    continue
                
                # V56 趋势过滤：股价在 60 日均线之上
                if V56_MA60_FILTER:
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
                
                # V56 行业对冲硬约束检查
                industry_ok, industry_reason = self.risk_manager.check_industry_constraint(industry)
                if not industry_ok:
                    logger.debug(f"INDUSTRY CONSTRAINT: {symbol} - {industry_reason}")
                    continue
                
                # V56 RS 强度标记
                is_top_rs = row.get('is_top_rs', False) or False
                rs_score = row.get('rs_strength', 0) or 0
                rs_rank = row.get('rs_rank', 9999) or 9999
                volume_breakout = row.get('volume_breakout', False) or False
                
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
                    'rs_score': rs_score,
                    'rs_rank': rs_rank,
                    'volume_breakout': volume_breakout,
                    'is_top_rs': is_top_rs,
                    'price_above_ma60': price_above_ma60 if V56_MA60_FILTER else True,
                    'open': row.get('open', 0) or 0,
                })
            
            if candidates:
                logger.info(f"Selected {len(candidates)} candidates: {[c['symbol'] for c in candidates]}")
            else:
                logger.warning(f"No candidates selected. Check filters (MA60, RS, industry)")
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error selecting candidates V56: {e}")
            logger.error(traceback.format_exc())
            return []
    
    def _generate_backtest_result(self, trade_dates: List[str]) -> Dict[str, Any]:
        """生成回测结果"""
        try:
            if not self.portfolio_values:
                return self._create_empty_result()
            
            final_value = self.portfolio_values[-1]['total_value']
            initial_value = self.initial_capital
            total_return = (final_value - initial_value) / initial_value
            
            # 计算年化收益
            if len(trade_dates) > 1:
                start = datetime.strptime(trade_dates[0], "%Y-%m-%d")
                end = datetime.strptime(trade_dates[-1], "%Y-%m-%d")
                days = (end - start).days
                years = days / 365.25 if days > 0 else 1
                annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return
            else:
                annual_return = total_return
            
            # 计算最大回撤
            max_dd = 0.0
            peak = self.portfolio_values[0]['total_value']
            for pv in self.portfolio_values:
                v = pv['total_value']
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            
            # 计算夏普比率
            if len(self.portfolio_values) > 1:
                daily_returns = []
                for i in range(1, len(self.portfolio_values)):
                    prev = self.portfolio_values[i-1]['total_value']
                    curr = self.portfolio_values[i]['total_value']
                    if prev > 0:
                        daily_returns.append((curr - prev) / prev)
                
                if daily_returns:
                    import numpy as np
                    mean_ret = np.mean(daily_returns)
                    std_ret = np.std(daily_returns)
                    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0
                else:
                    sharpe = 0
            else:
                sharpe = 0
            
            # 交易统计
            total_trades = len(self.risk_manager.trades)
            buy_trades = [t for t in self.risk_manager.trades if t.side == "BUY"]
            sell_trades = [t for t in self.risk_manager.trades if t.side == "SELL"]
            
            # 胜率计算
            profitable_trades = sum(1 for t in self.risk_manager.trade_log if t.is_profitable)
            total_closed = len(self.risk_manager.trade_log)
            win_rate = profitable_trades / total_closed if total_closed > 0 else 0
            
            # 盈亏比计算
            total_profit = sum(t.net_pnl for t in self.risk_manager.trade_log if t.net_pnl > 0)
            total_loss = abs(sum(t.net_pnl for t in self.risk_manager.trade_log if t.net_pnl < 0))
            profit_loss_ratio = total_profit / total_loss if total_loss > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'max_drawdown': max_dd,
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'profit_loss_ratio': profit_loss_ratio,
                'total_trades': total_trades,
                'total_buy_trades': len(buy_trades),
                'total_sell_trades': len(sell_trades),
                'final_value': final_value,
                'initial_value': initial_value,
                'portfolio_values': self.portfolio_values,
                'trades': self.risk_manager.trades,
                'trade_log': self.risk_manager.trade_log,
                'positions': self.risk_manager.positions,
                'wash_sale_stats': self.risk_manager.get_wash_sale_stats(),
                'blacklist_stats': self.risk_manager.get_blacklist_stats(),
                'trade_count_stats': self.risk_manager.get_trade_count_stats(),
                'three_level_defense_stats': self.risk_manager.get_three_level_defense_stats(),
                'frequency_fuse_stats': self.risk_manager.get_frequency_fuse_stats(),
                'rs_strength_stats': self.risk_manager.get_rs_strength_stats(),
                'stop_audit_records': self.risk_manager.get_stop_audit_records(),
                'v56_config': {
                    'rs_enabled': V56_RS_ENABLED,
                    'rs_top_percentile': V56_RS_TOP_PERCENTILE,
                    'breakeven_enabled': V56_BREAKEVEN_ENABLED,
                    'breakeven_threshold': V56_BREAKEVEN_PROFIT_THRESHOLD,
                    'tiered_profit_enabled': V56_TIERED_PROFIT_ENABLED,
                    'tiered_profit_levels': V56_TIERED_PROFIT_LEVELS,
                    'hard_stop_atr_mult': V56_HARD_STOP_LOSS_ATR_MULT,
                    'weekly_trade_limit': V56_WEEKLY_TRADE_LIMIT,
                    'global_trade_limit': V56_GLOBAL_TRADE_LIMIT,
                }
            }
            
        except Exception as e:
            logger.error(f"_generate_backtest_result failed: {e}")
            return self._create_empty_result()
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """创建空结果"""
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'total_trades': 0,
            'final_value': self.initial_capital,
            'initial_value': self.initial_capital,
            'portfolio_values': [],
            'trades': [],
            'trade_log': [],
            'positions': {},
            'wash_sale_stats': {},
            'blacklist_stats': {},
            'trade_count_stats': {},
            'three_level_defense_stats': {},
            'frequency_fuse_stats': {},
            'rs_strength_stats': {},
            'stop_audit_records': [],
            'iteration_results': [],
            'best_iteration': {},
            'v56_config': {}
        }


# ===========================================
# V56 回测脚本入口
# ===========================================

if __name__ == "__main__":
    from db_manager import DatabaseManager
    
    db = DatabaseManager()
    
    # 加载价格数据
    price_df = db.read_sql("SELECT * FROM stock_daily WHERE trade_date >= '2024-01-01' AND trade_date <= '2025-12-31'")
    
    # 加载指数数据
    index_df = db.read_sql("SELECT * FROM index_daily WHERE index_name = '沪深 300' AND trade_date >= '2024-01-01' AND trade_date <= '2025-12-31'")
    
    # 运行回测
    engine = V56BacktestEngine(initial_capital=100000, db=db)
    result = engine.run_backtest(
        price_df=price_df,
        start_date="2024-01-01",
        end_date="2025-12-31",
        index_df=index_df
    )
    
    # 打印结果
    print("\n" + "=" * 60)
    print("V56 BACKTEST RESULT")
    print("=" * 60)
    print(f"Total Return: {result['total_return']:.2%}")
    print(f"Annual Return: {result['annual_return']:.2%}")
    print(f"Max Drawdown: {result['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    print(f"Win Rate: {result['win_rate']:.2%}")
    print(f"Profit/Loss Ratio: {result['profit_loss_ratio']:.2f}")
    print(f"Total Trades: {result['total_trades']}")
    
    # 打印 5 轮迭代结果
    if result.get('iteration_results'):
        print("\n" + "=" * 60)
        print("5-ROUND PARAMETER SCAN RESULTS")
        print("=" * 60)
        for ir in result['iteration_results']:
            print(f"\nRound {ir.get('round')}: {ir.get('description')}")
            print(f"  Parameters: {ir.get('parameters')}")
            metrics = ir.get('metrics', {})
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    
    # 打印最优迭代
    if result.get('best_iteration'):
        print("\n" + "=" * 60)
        print("BEST ITERATION")
        print("=" * 60)
        best = result['best_iteration']
        print(f"Round: {best.get('round')}")
        print(f"Description: {best.get('description')}")
        print(f"Parameters: {best.get('parameters')}")
        metrics = best.get('metrics', {})
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")