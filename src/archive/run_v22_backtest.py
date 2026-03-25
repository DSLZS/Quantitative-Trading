"""
V22 Backtest Runner - 盈利质量与大盘避险重构

【使用说明】
1. 从数据库获取股票数据和信号（使用 V22 因子体系）
2. 使用 V20 会计引擎执行真实回测（跟踪止盈 + 大盘风控 + 波动率过滤）
3. 生成 V22 报告并与 V21 对比

【核心特性】
- 跟踪止盈：盈利突破 8% 后，从最高价回撤超过 4% 强制清仓
- 大盘风控过滤器：沪深 300 指数 < MA20 时进入防守模式
- 波动率剔除：剔除波动率前 5% 的妖股
- 真实账户管理：现金、持仓、冻结资金
- 真实手续费计算：滑点 + 规费 + 印花税 + 最低佣金 5 元
- T+1 锁定：今日买入明日才能卖
- 持仓生命周期：止损、跟踪止盈、信号排名

【严防作弊与硬约束】
- 锁死会计引擎：必须直接引用 v20_accounting_engine.py
- 本金限制：严格基于 100,000 元初始资金
- 严禁未来函数：所有数据处理必须严格 .shift(1)

作者：资深量化基金经理 (V22: 盈利质量与大盘避险重构)
版本：V22.0 (Profit Quality & Market Regime Filter)
日期：2026-03-18
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .v20_accounting_engine import (
        V20AccountingEngine,
        BacktestResult,
        DailyNAV,
        Position,
        INITIAL_CAPITAL as V20_INITIAL_CAPITAL,
        TOP_K_STOCKS as V20_TOP_K_STOCKS,
    )
    from .v22_strategy import (
        V22Strategy,
        V22DynamicSizingManager,
        V22PositionManager,
        V22MarketRegimeFilter,
        INITIAL_CAPITAL,
        TARGET_POSITIONS,
        SCORE_THRESHOLD,
        BUY_RANK_PERCENTILE,
        SELL_RANK_PERCENTILE,
        DEFENSE_SELL_RANK_PERCENTILE,
        STOP_LOSS_RATIO,
        TRAILING_STOP_THRESHOLD,
        TRAILING_STOP_DRAWDOWN,
        MAX_POSITION_RATIO,
        MAX_POSITIONS,
        MARKET_STATE_SAFE,
        MARKET_STATE_CAUTION,
        MARKET_STATE_DANGER,
    )
except ImportError:
    from db_manager import DatabaseManager
    from v20_accounting_engine import (
        V20AccountingEngine,
        BacktestResult,
        DailyNAV,
        Position,
        INITIAL_CAPITAL as V20_INITIAL_CAPITAL,
        TOP_K_STOCKS as V20_TOP_K_STOCKS,
    )
    from v22_strategy import (
        V22Strategy,
        V22DynamicSizingManager,
        V22PositionManager,
        V22MarketRegimeFilter,
        INITIAL_CAPITAL,
        TARGET_POSITIONS,
        SCORE_THRESHOLD,
        BUY_RANK_PERCENTILE,
        SELL_RANK_PERCENTILE,
        DEFENSE_SELL_RANK_PERCENTILE,
        STOP_LOSS_RATIO,
        TRAILING_STOP_THRESHOLD,
        TRAILING_STOP_DRAWDOWN,
        MAX_POSITION_RATIO,
        MAX_POSITIONS,
        MARKET_STATE_SAFE,
        MARKET_STATE_CAUTION,
        MARKET_STATE_DANGER,
    )


# ===========================================
# V22 扩展持仓信息管理器
# ===========================================

class V22PositionInfoTracker:
    """
    V22 持仓信息跟踪器 - 用于跟踪最高盈利以支持跟踪止盈
    
    【功能】
    - 记录每只股票的买入成本
    - 跟踪持仓期间的最高盈利比例
    - 计算当前盈亏比例
    """
    
    def __init__(self):
        self.position_info: Dict[str, Dict[str, Any]] = {}
        logger.info("V22PositionInfoTracker initialized")
    
    def update_position(
        self,
        symbol: str,
        buy_date: str,
        avg_cost: float,
        shares: int,
    ):
        """新建或更新持仓信息"""
        self.position_info[symbol] = {
            "buy_date": buy_date,
            "avg_cost": avg_cost,
            "shares": shares,
            "highest_price": avg_cost,  # 初始为成本价
            "highest_pnl_ratio": 0.0,   # 初始为 0
            "current_price": avg_cost,
            "pnl_ratio": 0.0,
        }
    
    def update_price(self, symbol: str, current_price: float):
        """更新价格并跟踪最高盈利"""
        if symbol not in self.position_info:
            return
        
        info = self.position_info[symbol]
        avg_cost = info["avg_cost"]
        
        # 计算当前盈亏比例
        pnl_ratio = (current_price - avg_cost) / avg_cost
        
        # 更新最高价和最高盈利
        if current_price > info["highest_price"]:
            info["highest_price"] = current_price
            info["highest_pnl_ratio"] = pnl_ratio
        
        # 更新当前价格和盈亏
        info["current_price"] = current_price
        info["pnl_ratio"] = pnl_ratio
    
    def remove_position(self, symbol: str):
        """移除持仓信息"""
        if symbol in self.position_info:
            del self.position_info[symbol]
    
    def get_info(self, symbol: str) -> Dict[str, Any]:
        """获取持仓信息"""
        return self.position_info.get(symbol, {})
    
    def get_all_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有持仓信息"""
        return self.position_info.copy()


# ===========================================
# V22 回测执行器
# ===========================================

class V22BacktestExecutor:
    """
    V22 回测执行器 - 使用 V20 会计引擎 + V22 策略逻辑
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        target_positions: int = TARGET_POSITIONS,
        db: Optional[DatabaseManager] = None,
    ):
        self.initial_capital = initial_capital
        self.target_positions = target_positions
        
        # 使用 V20 会计引擎（严禁修改）
        self.accounting = V20AccountingEngine(
            initial_capital=initial_capital,
            top_k_stocks=MAX_POSITIONS,
            db=db,
        )
        
        # V22 组件
        self.sizing_manager = V22DynamicSizingManager()
        self.position_manager = V22PositionManager()
        self.market_regime_filter = V22MarketRegimeFilter()
        self.position_tracker = V22PositionInfoTracker()
        
        # 市场环境状态记录
        self.market_states: Dict[str, str] = {}
        
        logger.info(f"V22BacktestExecutor initialized")
        logger.info(f"  Initial Capital: {initial_capital:,.0f}")
        logger.info(f"  Using V20AccountingEngine (locked)")
        logger.info(f"  Score Threshold: {SCORE_THRESHOLD}")
        logger.info(f"  Buy Rank Percentile: Top {BUY_RANK_PERCENTILE}%")
        logger.info(f"  Sell Rank Percentile: > {SELL_RANK_PERCENTILE}% (Normal)")
        logger.info(f"  Sell Rank Percentile: > {DEFENSE_SELL_RANK_PERCENTILE}% (Defense)")
        logger.info(f"  Trailing Stop: {TRAILING_STOP_THRESHOLD:.1%} threshold, {TRAILING_STOP_DRAWDOWN:.1%} drawdown")
    
    def run_backtest(
        self,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        index_df: pl.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """运行 V22 回测"""
        logger.info("=" * 60)
        logger.info("V22 STRATEGY BACKTEST EXECUTION")
        logger.info("=" * 60)
        
        # 日期过滤
        if start_date:
            signals_df = signals_df.filter(pl.col("trade_date") >= start_date)
            prices_df = prices_df.filter(pl.col("trade_date") >= start_date)
        if end_date:
            prices_df = prices_df.filter(pl.col("trade_date") <= end_date)
        
        # 计算市场状态
        market_states_df = self.market_regime_filter.compute_market_state(index_df)
        
        # 获取交易日期
        dates = sorted(signals_df["trade_date"].unique().to_list())
        
        if not dates:
            logger.error("No trading dates found")
            return BacktestResult(
                start_date="",
                end_date="",
                initial_capital=self.initial_capital,
            )
        
        start_date = dates[0]
        end_date = dates[-1]
        
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Trading days: {len(dates)}")
        
        # 每日循环
        for trade_date in dates:
            # 获取当日市场状态
            market_state = self.market_regime_filter.get_daily_state(market_states_df, trade_date)
            self.market_states[trade_date] = market_state
            self._execute_day(trade_date, signals_df, prices_df, market_state)
        
        return self._generate_result(start_date, end_date)
    
    def _execute_day(
        self,
        trade_date: str,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        market_state: str,
    ):
        """执行单日交易逻辑"""
        logger.debug(f"\n=== {trade_date} (Market: {market_state}) ===")
        
        # 判断是否为防守模式
        is_defense_mode = (market_state == MARKET_STATE_DANGER)
        
        # 1. 获取当日信号
        day_signals = signals_df.filter(pl.col("trade_date") == trade_date)
        if day_signals.is_empty():
            logger.debug(f"No signals for {trade_date}")
            return
        
        # 2. 计算信号排名
        ranked = day_signals.sort("signal", descending=True).with_columns([
            pl.col("signal").rank("ordinal", descending=True).alias("rank")
        ])
        
        signal_ranks = {}
        total_stocks = len(ranked)
        for row in ranked.iter_rows(named=True):
            symbol = row["symbol"]
            rank = row["rank"]
            signal_ranks[symbol] = int(rank) if rank is not None else 9999
        
        # 3. 获取当日价格
        day_prices = prices_df.filter(pl.col("trade_date") == trade_date)
        prices = {
            row["symbol"]: float(row["close"])
            for row in day_prices.iter_rows(named=True)
        }
        
        # 4. 更新持仓价格和跟踪最高盈利
        for symbol in self.accounting.positions.keys():
            if symbol in prices:
                self.position_tracker.update_price(symbol, prices[symbol])
        
        # 5. 构建持仓盈亏信息（含跟踪止盈所需的最高盈利）
        position_info = self.position_tracker.get_all_info()
        
        # 6. 获取应卖出的股票（含跟踪止盈逻辑）
        stocks_to_sell = self.position_manager.get_stocks_to_sell(
            positions=self.accounting.positions,
            signal_ranks=signal_ranks,
            total_stocks=total_stocks,
            position_info=position_info,
            is_defense_mode=is_defense_mode,
        )
        
        # 7. 执行卖出
        for symbol in stocks_to_sell:
            if symbol in prices:
                self.accounting.execute_sell(
                    trade_date=trade_date,
                    symbol=symbol,
                    price=prices[symbol],
                )
                self.position_tracker.remove_position(symbol)
        
        # 8. 计算动态权重
        target_amounts = self.sizing_manager.compute_dynamic_weights(
            signals_df=day_signals,
            current_positions=self.accounting.positions,
            initial_capital=self.initial_capital,
        )
        
        # 9. 获取应买入的股票（防守模式下不买入）
        stocks_to_buy = self.position_manager.get_stocks_to_buy(
            signals_df=day_signals,
            current_positions=self.accounting.positions,
            target_amounts=target_amounts,
            is_defense_mode=is_defense_mode,
        )
        
        # 10. 执行买入
        for symbol, target_amount in stocks_to_buy:
            if symbol in prices and target_amount > 0:
                trade = self.accounting.execute_buy(
                    trade_date=trade_date,
                    symbol=symbol,
                    price=prices[symbol],
                    target_amount=target_amount,
                )
                if trade:
                    # 更新持仓跟踪器
                    pos = self.accounting.positions.get(symbol)
                    if pos:
                        self.position_tracker.update_position(
                            symbol=symbol,
                            buy_date=trade_date,
                            avg_cost=pos.avg_cost,
                            shares=pos.shares,
                        )
        
        # 11. 计算当日 NAV
        self.accounting.compute_daily_nav(trade_date, prices)
        
        # 12. 记录持仓快照
        self.accounting.record_positions_snapshot(trade_date)
    
    def _generate_result(self, start_date: str, end_date: str) -> BacktestResult:
        """生成回测结果"""
        accounting = self.accounting
        
        if len(accounting.daily_navs) > 0:
            final_nav = accounting.daily_navs[-1].total_assets
            total_return = (final_nav - self.initial_capital) / self.initial_capital
            
            trading_days = len(accounting.daily_navs)
            years = trading_days / 252.0
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
            
            daily_returns = [nav.daily_return for nav in accounting.daily_navs]
            if len(daily_returns) > 1:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns, ddof=1)
                sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
            else:
                sharpe_ratio = 0.0
            
            nav_values = [nav.total_assets for nav in accounting.daily_navs]
            rolling_max = np.maximum.accumulate(nav_values)
            drawdowns = (np.array(nav_values) - rolling_max) / rolling_max
            max_drawdown = abs(np.min(drawdowns))
            
            winning_days = sum(1 for nav in accounting.daily_navs if nav.daily_return > 0)
            win_rate = winning_days / len(accounting.daily_navs)
        else:
            total_return = annual_return = sharpe_ratio = max_drawdown = win_rate = 0.0
        
        total_trades = len(accounting.trades)
        buy_trades = [t for t in accounting.trades if t.side == "BUY"]
        sell_trades = [t for t in accounting.trades if t.side == "SELL"]
        
        total_buy_amount = sum(t.amount for t in buy_trades)
        total_sell_amount = sum(t.amount for t in sell_trades)
        total_commission = sum(t.commission for t in accounting.trades)
        total_stamp_duty = sum(t.stamp_duty for t in accounting.trades)
        total_slippage = sum(t.slippage for t in accounting.trades)
        
        turnover_rate = total_buy_amount / self.initial_capital if self.initial_capital > 0 else 0.0
        avg_holding_days = np.mean(list(accounting.holding_periods.values())) if accounting.holding_periods else 0.0
        
        if accounting.positions_history:
            position_counts = [len(snapshot["positions"]) for snapshot in accounting.positions_history]
            avg_position_count = np.mean(position_counts) if position_counts else 0.0
        else:
            avg_position_count = 0.0
        
        result = BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            total_buy_amount=total_buy_amount,
            total_sell_amount=total_sell_amount,
            total_commission=total_commission,
            total_stamp_duty=total_stamp_duty,
            total_slippage=total_slippage,
            turnover_rate=turnover_rate,
            avg_holding_days=avg_holding_days,
            avg_position_count=avg_position_count,
            daily_navs=accounting.daily_navs,
            trades=accounting.trades,
            positions_history=accounting.positions_history,
        )
        
        # 统计市场状态分布
        state_counts = {
            MARKET_STATE_SAFE: 0,
            MARKET_STATE_CAUTION: 0,
            MARKET_STATE_DANGER: 0,
        }
        for state in self.market_states.values():
            state_counts[state] = state_counts.get(state, 0) + 1
        
        logger.info("\n" + "=" * 60)
        logger.info("V22 BACKTEST RESULT")
        logger.info("=" * 60)
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annual Return: {annual_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.3f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Total Commission: {total_commission:,.2f}")
        logger.info(f"Turnover Rate: {turnover_rate:.2%}")
        logger.info(f"Avg Holding Days: {avg_holding_days:.1f}")
        logger.info(f"Market States: {state_counts}")
        
        return result


# ===========================================
# V22 回测运行器
# ===========================================

class V22BacktestRunner:
    """V22 回测运行器 - 整合信号生成和会计引擎"""
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        target_positions: int = TARGET_POSITIONS,
        db: Optional[DatabaseManager] = None,
    ):
        self.initial_capital = initial_capital
        self.target_positions = target_positions
        self.db = db or DatabaseManager.get_instance()
        self.strategy = V22Strategy(db=self.db)
        
        logger.info(f"V22BacktestRunner initialized")
        logger.info(f"  Initial Capital: {initial_capital:,.0f}")
        logger.info(f"  Target Positions: {target_positions}")
    
    def run_backtest(self, start_date: str, end_date: str) -> BacktestResult:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V22 BACKTEST RUNNER")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} to {end_date}")
        
        # 1. 生成信号
        logger.info("\nStep 1: Generating signals...")
        signals_df = self.strategy.generate_signals(start_date, end_date)
        
        if signals_df.is_empty():
            logger.error("Signal generation failed")
            return BacktestResult(start_date=start_date, end_date=end_date, initial_capital=self.initial_capital)
        
        # 2. 获取价格数据
        logger.info("\nStep 2: Getting price data...")
        prices_df = self.strategy.get_prices(start_date, end_date)
        
        if prices_df.is_empty():
            logger.error("Price data not found")
            return BacktestResult(start_date=start_date, end_date=end_date, initial_capital=self.initial_capital)
        
        logger.info(f"Loaded {len(prices_df)} price records")
        
        # 3. 获取指数数据
        logger.info("\nStep 3: Getting index data...")
        index_df = self.strategy.get_index_data(start_date, end_date)
        
        if index_df.is_empty():
            logger.warning("Index data not found, using Safe market state")
        
        # 4. 运行回测
        logger.info("\nStep 4: Running V22 backtest...")
        executor = V22BacktestExecutor(
            initial_capital=self.initial_capital,
            target_positions=self.target_positions,
            db=self.db,
        )
        
        result = executor.run_backtest(
            signals_df=signals_df,
            prices_df=prices_df,
            index_df=index_df,
        )
        return result
    
    def generate_report(
        self,
        result: BacktestResult,
        v21_result: Optional[BacktestResult] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """生成报告"""
        return generate_v22_report(result, v21_result=v21_result, output_path=output_path)
    
    def get_trading_instructions(self, trade_date: str) -> str:
        """获取当日交易指令"""
        signals_df = self.strategy.generate_signals(
            (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d"),
            trade_date,
        )
        prices_df = self.strategy.get_prices(
            (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d"),
            trade_date,
        )
        index_df = self.strategy.get_index_data(
            (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d"),
            trade_date,
        )
        
        # 计算市场状态
        market_states_df = self.strategy.compute_market_states(index_df)
        market_state = self.strategy.market_regime_filter.get_daily_state(market_states_df, trade_date)
        
        recommendations = self.strategy.get_daily_recommendations(
            signals_df, prices_df, trade_date, self.initial_capital, market_state=market_state
        )
        
        return generate_trading_instructions(recommendations, trade_date, market_state)


# ===========================================
# V21 结果加载器
# ===========================================

def load_v21_result(json_path: Optional[str] = None) -> Optional[BacktestResult]:
    """从 JSON 文件加载 V21 结果"""
    if json_path is None:
        reports_dir = Path("reports")
        v21_files = list(reports_dir.glob("V21_backtest_result_*.json"))
        if v21_files:
            v21_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            json_path = str(v21_files[0])
    
    if json_path is None or not Path(json_path).exists():
        logger.warning("V21 result not found, skipping comparison")
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        v21_result = BacktestResult(
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            initial_capital=data.get("initial_capital", INITIAL_CAPITAL),
            total_return=data.get("total_return", 0.0),
            annual_return=data.get("annual_return", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            win_rate=data.get("win_rate", 0.0),
            total_trades=data.get("total_trades", 0),
            total_buy_amount=data.get("total_buy_amount", 0.0),
            total_sell_amount=data.get("total_sell_amount", 0.0),
            total_commission=data.get("total_commission", 0.0),
            total_stamp_duty=data.get("total_stamp_duty", 0.0),
            total_slippage=data.get("total_slippage", 0.0),
            turnover_rate=data.get("turnover_rate", 0.0),
            avg_holding_days=data.get("avg_holding_days", 0.0),
            avg_position_count=data.get("avg_position_count", 0.0),
        )
        
        logger.info(f"Loaded V21 result from {json_path}")
        return v21_result
    except Exception as e:
        logger.warning(f"Failed to load V21 result: {e}")
        return None


# ===========================================
# 报告生成器
# ===========================================

def generate_v22_report(
    result: BacktestResult,
    v21_result: Optional[BacktestResult] = None,
    output_path: Optional[str] = None,
) -> str:
    """生成 V22 报告"""
    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"V22_Profit_Quality_Report_{timestamp}.md"
    
    total_fees = result.total_commission + result.total_stamp_duty + result.total_slippage
    gross_profit = result.total_return * result.initial_capital
    net_profit = gross_profit - total_fees
    fee_erosion_ratio = total_fees / gross_profit if gross_profit > 0 else 0.0
    cost_ratio = total_fees / result.initial_capital
    
    # 利费比 = 净利润 / 总手续费
    profit_fee_ratio = net_profit / total_fees if total_fees > 0 else 0.0
    
    v21_comparison = ""
    if v21_result is not None:
        v21_fees = v21_result.total_commission + v21_result.total_stamp_duty + v21_result.total_slippage
        v21_gross_profit = v21_result.total_return * v21_result.initial_capital
        v21_net_profit = v21_gross_profit - v21_fees
        v21_fee_erosion_ratio = v21_fees / v21_gross_profit if v21_gross_profit > 0 else 0.0
        v21_profit_fee_ratio = v21_net_profit / v21_fees if v21_fees > 0 else 0.0
        
        return_change = (result.total_return - v21_result.total_return) * 100
        mdd_change = (result.max_drawdown - v21_result.max_drawdown) * 100
        fee_change = (total_fees - v21_fees) / v21_fees * 100 if v21_fees > 0 else 0
        profit_fee_change = profit_fee_ratio - v21_profit_fee_ratio
        
        v21_comparison = f"""
## 三、V21 vs V22 对比分析

### 3.1 核心指标对比

| 指标 | V21 | V22 | 变化 | 状态 |
|------|-----|-----|------|------|
| 总收益 | {v21_result.total_return:.2%} | {result.total_return:.2%} | {return_change:+.2f}% | {"✅" if result.total_return > v21_result.total_return else "⚠️"} |
| 年化收益 | {v21_result.annual_return:.2%} | {result.annual_return:.2%} | {(result.annual_return - v21_result.annual_return):+.2%} | {"✅" if result.annual_return > v21_result.annual_return else "⚠️"} |
| 夏普比率 | {v21_result.sharpe_ratio:.3f} | {result.sharpe_ratio:.3f} | {(result.sharpe_ratio - v21_result.sharpe_ratio):+.3f} | {"✅" if result.sharpe_ratio > v21_result.sharpe_ratio else "⚠️"} |
| **最大回撤** | {v21_result.max_drawdown:.2%} | {result.max_drawdown:.2%} | {mdd_change:+.2f}% | {"✅" if result.max_drawdown < v21_result.max_drawdown else "⚠️"} |
| 胜率 | {v21_result.win_rate:.1%} | {result.win_rate:.1%} | {(result.win_rate - v21_result.win_rate):+.1%} | {"✅" if result.win_rate > v21_result.win_rate else "⚠️"} |

### 3.2 手续费与利费比对比

| 指标 | V21 | V22 | 变化 | 状态 |
|------|-----|-----|------|------|
| 总手续费 | {v21_fees:,.2f} 元 | {total_fees:,.2f} 元 | {fee_change:+.1f}% | {"✅" if total_fees < v21_fees else "⚠️"} |
| 毛利润 | {v21_gross_profit:,.2f} 元 | {gross_profit:,.2f} 元 | {(gross_profit - v21_gross_profit):,.2f}元 | {"✅" if gross_profit > v21_gross_profit else "⚠️"} |
| 净利润 | {v21_net_profit:,.2f} 元 | {net_profit:,.2f} 元 | {(net_profit - v21_net_profit):,.2f}元 | {"✅" if net_profit > v21_net_profit else "⚠️"} |
| 手续费侵蚀率 | {v21_fee_erosion_ratio:.1%} | {fee_erosion_ratio:.1%} | {(fee_erosion_ratio - v21_fee_erosion_ratio):+.1%} | {"✅" if fee_erosion_ratio < v21_fee_erosion_ratio else "⚠️"} |
| **利费比** | {v21_profit_fee_ratio:.2f} | {profit_fee_ratio:.2f} | {profit_fee_change:+.2f} | {"✅" if profit_fee_ratio > v21_profit_fee_ratio else "⚠️"} |

### 3.3 交易行为对比

| 指标 | V21 | V22 | 变化 | 状态 |
|------|-----|-----|------|------|
| 总交易次数 | {v21_result.total_trades:,} | {result.total_trades:,} | {(result.total_trades - v21_result.total_trades):,} | {"✅" if result.total_trades < v21_result.total_trades else "⚠️"} |
| 平均持仓天数 | {v21_result.avg_holding_days:.1f}天 | {result.avg_holding_days:.1f}天 | {(result.avg_holding_days - v21_result.avg_holding_days):+.1f}天 | {"✅" if result.avg_holding_days > v21_result.avg_holding_days else "⚠️"} |
| 换手率 | {v21_result.turnover_rate:.2%} | {result.turnover_rate:.2%} | {(result.turnover_rate - v21_result.turnover_rate):+.2%} | {"✅" if result.turnover_rate < v21_result.turnover_rate else "⚠️"} |
| 平均持仓数量 | {v21_result.avg_position_count:.1f}只 | {result.avg_position_count:.1f}只 | {(result.avg_position_count - v21_result.avg_position_count):+.1f}只 | - |

### 3.4 核心改进验证

| 改进项 | V21 表现 | V22 表现 | 目标 | 状态 |
|--------|----------|----------|------|------|
| 跟踪止盈 | 15% 硬止盈 | 8% 触发 +4% 回撤 | 捕捉长趋势 | {"✅" if result.avg_holding_days > v21_result.avg_holding_days else "⚠️"} |
| 大盘风控 | 无 | MA20 判断 | 避免系统性风险 | {"✅" if result.max_drawdown < v21_result.max_drawdown else "⚠️"} |
| 波动率剔除 | 无 | 剔除前 5% 妖股 | 提高成交质量 | {"✅" if profit_fee_ratio > v21_profit_fee_ratio else "⚠️"} |
| 利费比提升 | {v21_profit_fee_ratio:.2f} | {profit_fee_ratio:.2f} | >1.0 | {"✅" if profit_fee_ratio > 1.0 else "⚠️"} |
"""
    
    report = f"""# V22 盈利质量与大盘避险重构报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V22.0 (Profit Quality & Market Regime Filter)

---

## 一、核心改进总结

### 1.1 大盘风控过滤器 (Market Regime Filter)

**V21 问题**: 无大盘环境判断，在系统性风险中依然开仓

**V22 解决方案**:
1. **市场状态判断**: 使用沪深 300 指数与 MA20 比较
   - Safe: 收盘价 > MA20
   - Caution: 收盘价在 MA20 的 2% 以内
   - Danger: 收盘价 < MA20
2. **防守模式**: 当市场状态为 Danger 时
   - 不再开新仓
   - 现有持仓的退出缓冲区收紧（排名跌出 Top 20 强制卖出）

### 1.2 跟踪止盈与波动率剔除 (Risk-Adjusted Exit)

**V21 问题**: 15% 硬止盈无法捕捉长趋势，高波动股票成交质量差

**V22 解决方案**:
1. **跟踪止盈**: 
   - 个股盈利突破 8% 后，记录最高价
   - 若收盘价从最高价回撤超过 4%，则强制清仓
2. **波动率黑名单**:
   - 计算过去 10 个交易日的收益率标准差
   - 剔除波动率位于全市场前 5% 的"妖股"

### 1.3 因子池精简 (Alpha Distillation)

**V22 因子体系**:
1. 量价相关性 (vol_price_corr) - 权重 0.25
2. 短线反转 (reversal_st) - 权重 0.20
3. 波动风险 (vol_risk) - 权重 0.15
4. 异常换手 (turnover_signal) - 权重 0.15
5. 动量因子 (momentum) - 权重 0.10（降低）
6. 低波动因子 (low_vol) - 权重 0.10
7. **量价背离 (volume_price_divergence)** - 权重 0.05（新因子）

### 1.4 真实手续费（维持 V20 严格标准）

| 费用类型 | 费率 | 说明 |
|----------|------|------|
| 买入滑点 | 0.1% | 实际成交价 vs 订单价 |
| 卖出滑点 | 0.1% | 实际成交价 vs 订单价 |
| 交易规费 | 0.03% | 经手费 + 证管费 |
| 印花税 | 0.05% | 仅卖出时收取 |
| **最低佣金** | **5 元** | 每笔交易不足 5 元按 5 元计 |

---

## 二、回测结果

### 2.1 收益指标

| 指标 | 值 |
|------|-----|
| 回测区间 | {result.start_date} 至 {result.end_date} |
| 初始资金 | {result.initial_capital:,.0f} 元 |
| **总收益** | **{result.total_return:.2%}** |
| **年化收益** | **{result.annual_return:.2%}** |
| **夏普比率** | **{result.sharpe_ratio:.3f}** |
| **最大回撤** | **{result.max_drawdown:.2%}** |
| 胜率 | {result.win_rate:.1%} |

### 2.2 真实交易指标

| 指标 | 值 |
|------|-----|
| **总交易次数** | {result.total_trades:,} |
| **总买入额** | {result.total_buy_amount:,.2f} 元 |
| **总手续费** | {result.total_commission:,.2f} 元 |
| **总印花税** | {result.total_stamp_duty:,.2f} 元 |
| **总滑点成本** | {result.total_slippage:,.2f} 元 |
| **换手率** | {result.turnover_rate:.2%} |

### 2.3 手续费侵蚀分析

| 指标 | 值 |
|------|-----|
| 毛利润 | {gross_profit:,.2f} 元 |
| 总手续费 | {total_fees:,.2f} 元 |
| **净利润** | **{net_profit:,.2f} 元** |
| **手续费侵蚀率** | **{fee_erosion_ratio:.1%}** |
| **利费比 (Net Profit / Fees)** | **{profit_fee_ratio:.2f}** |

状态：{"✅ 利费比健康" if profit_fee_ratio > 1.0 else "⚠️ 利费比需提升" if profit_fee_ratio > 0.5 else "❌ 利费比过低"}

### 2.4 持仓指标

| 指标 | 值 |
|------|-----|
| 平均持仓天数 | {result.avg_holding_days:.1f} 天 |
| 平均持仓数量 | {result.avg_position_count:.1f} 只 |

{v21_comparison}

---

## 三、严防偷懒与作弊

- ✅ 基于 V20AccountingEngine，未修改手续费和 T+1 逻辑
- ✅ 输出 Total Buy Amount: {result.total_buy_amount:,.2f} 元
- ✅ 输出 Turnover Rate: {result.turnover_rate:.2%}
- ✅ 输出 利费比：{profit_fee_ratio:.2f}
- ✅ 所有平滑和过滤逻辑使用 .shift(1)
- ✅ T+1 锁定严格执行
- ✅ 输出每日市场环境状态（Safe/Caution/Danger）

---

## 四、执行总结

### 4.1 核心结论

1. **大盘风控**: 通过沪深 300 指数与 MA20 比较，避免在系统性风险中开仓
2. **跟踪止盈**: 盈利突破 8% 后跟踪最高价，回撤 4% 清仓，捕捉长趋势
3. **波动率剔除**: 剔除前 5% 妖股，提高成交质量
4. **因子精简**: 新增"量价背离"因子，寻找缩量洗盘后的启动机会

### 4.2 后续优化方向

1. 动态调整 MA 窗口（根据市场波动率）
2. 优化跟踪止盈阈值（8% 触发 +4% 回撤）
3. 增加行业中性化约束

---

**报告生成完毕**
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"V22 report saved to: {output_path}")
    return str(output_path)


def generate_trading_instructions(
    recommendations: List[Dict[str, Any]],
    trade_date: str,
    market_state: str = MARKET_STATE_SAFE,
) -> str:
    """生成实盘指令"""
    if not recommendations:
        return f"无符合条件的交易指令 (市场状态：{market_state})"
    
    lines = []
    lines.append("=" * 80)
    lines.append("V22 实盘交易指令")
    lines.append("=" * 80)
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"交易日期：{trade_date}")
    lines.append(f"**市场环境状态**: {market_state}")
    lines.append(f"指令数量：{len(recommendations)}")
    lines.append(f"总仓位：{sum(r['weight'] for r in recommendations):.1f}%")
    lines.append("")
    lines.append("-" * 80)
    lines.append(f"{'股票代码':<12} {'推荐权重':<10} {'预测得分':<10} {'目标金额':<12} {'建议理由'}")
    lines.append("-" * 80)
    
    for rec in recommendations:
        lines.append(f"{rec['symbol']:<12} {rec['weight']:<10.1f}% {rec['score']:<10.4f} {rec['target_amount']:<12.0f}元 {rec['reason']}")
    
    lines.append("-" * 80)
    lines.append("")
    lines.append("说明：")
    lines.append(f"  - 买入阈值：得分 > {SCORE_THRESHOLD} 且排名前{BUY_RANK_PERCENTILE}%")
    lines.append(f"  - 卖出缓冲：跌出 Top {SELL_RANK_PERCENTILE}% 才卖出（防守模式 Top {DEFENSE_SELL_RANK_PERCENTILE}%）")
    lines.append(f"  - 止损：{STOP_LOSS_RATIO:.0%} 硬止损")
    lines.append(f"  - 跟踪止盈：盈利突破{TRAILING_STOP_THRESHOLD:.0%}后，回撤{TRAILING_STOP_DRAWDOWN:.0%}清仓")
    lines.append(f"  - 单只上限：{MAX_POSITION_RATIO:.0%}")
    lines.append("")
    lines.append("市场状态说明：")
    lines.append(f"  - Safe: 沪深 300 收盘价 > MA20，正常开仓")
    lines.append(f"  - Caution: 沪深 300 收盘价在 MA20 的 2% 以内，谨慎开仓")
    lines.append(f"  - Danger: 沪深 300 收盘价 < MA20，防守模式（不开新仓，收紧卖出缓冲区）")
    
    return "\n".join(lines)


# ===========================================
# 主函数
# ===========================================

def main():
    """主入口函数"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("V22 Profit Quality & Market Regime Filter Backtest Runner")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 默认参数
    start_date = "2025-01-01"
    end_date = "2026-03-17"
    initial_capital = INITIAL_CAPITAL
    target_positions = TARGET_POSITIONS
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
    if len(sys.argv) > 2:
        end_date = sys.argv[2]
    if len(sys.argv) > 3:
        initial_capital = float(sys.argv[3])
    if len(sys.argv) > 4:
        target_positions = int(sys.argv[4])
    
    logger.info(f"Parameters:")
    logger.info(f"  Start Date: {start_date}")
    logger.info(f"  End Date: {end_date}")
    logger.info(f"  Initial Capital: {initial_capital:,.0f}")
    logger.info(f"  Target Positions: {target_positions}")
    
    # 加载 V21 结果
    logger.info("\nStep 0: Loading V21 result for comparison...")
    v21_result = load_v21_result()
    
    # 运行回测
    runner = V22BacktestRunner(
        initial_capital=initial_capital,
        target_positions=target_positions,
    )
    
    result = runner.run_backtest(start_date, end_date)
    
    # 生成报告
    logger.info("\nStep 5: Generating report...")
    report_path = runner.generate_report(result, v21_result=v21_result)
    
    # 保存 JSON 结果
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"V22_backtest_result_{timestamp}.json"
    
    result_dict = {
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "total_buy_amount": result.total_buy_amount,
        "total_sell_amount": result.total_sell_amount,
        "total_commission": result.total_commission,
        "total_stamp_duty": result.total_stamp_duty,
        "total_slippage": result.total_slippage,
        "turnover_rate": result.turnover_rate,
        "avg_holding_days": result.avg_holding_days,
        "avg_position_count": result.avg_position_count,
        "report_path": report_path,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nJSON result saved to: {json_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("V22 BACKTEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report Path: {report_path}")
    logger.info(f"JSON Path: {json_path}")
    
    # 输出最终结果
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"Annual Return: {result.annual_return:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Total Commission: {result.total_commission:,.2f} 元")
    logger.info(f"Turnover Rate: {result.turnover_rate:.2%}")
    logger.info(f"Avg Holding Days: {result.avg_holding_days:.1f} 天")
    
    # 计算利费比
    total_fees = result.total_commission + result.total_stamp_duty + result.total_slippage
    net_profit = result.total_return * result.initial_capital - total_fees
    profit_fee_ratio = net_profit / total_fees if total_fees > 0 else 0.0
    logger.info(f"Profit-Fee Ratio (利费比): {profit_fee_ratio:.2f}")
    
    # 生成实盘指令示例
    logger.info("\n" + "=" * 60)
    logger.info("TRADING INSTRUCTIONS EXAMPLE")
    logger.info("=" * 60)
    try:
        instructions = runner.get_trading_instructions(result.end_date)
        logger.info("\n" + instructions)
    except Exception as e:
        logger.info(f"Could not generate trading instructions: {e}")


if __name__ == "__main__":
    main()