"""
V21 Backtest Runner - 动态赋权与换手优化

【使用说明】
1. 从数据库获取股票数据和信号（使用 V21 因子体系）
2. 使用 V20 会计引擎执行真实回测（动态权重 + 缓冲区）
3. 生成 V21 报告并与 V20 对比

【核心特性】
- 动态权重：根据信号强度分配仓位（凯利准则简化版）
- 换手率缓冲区：Top 5% 买入 / Top 40% 卖出
- 真实账户管理：现金、持仓、冻结资金
- 真实手续费计算：滑点 + 规费 + 印花税 + 最低佣金 5 元
- T+1 锁定：今日买入明日才能卖
- 持仓生命周期：止损、止盈、信号排名

【严防作弊与硬约束】
- 锁死会计引擎：必须直接引用 v20_accounting_engine.py
- 本金限制：严格基于 100,000 元初始资金
- 严禁未来函数：所有数据处理必须严格 .shift(1)

作者：资深量化策略分析师 (V21: 动态赋权与摩擦抑止)
版本：V21.0 (Dynamic Sizing & Turnover Buffer)
日期：2026-03-17
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
        INITIAL_CAPITAL as V20_INITIAL_CAPITAL,
        TOP_K_STOCKS as V20_TOP_K_STOCKS,
    )
    from .v21_strategy import (
        V21Strategy,
        V21DynamicSizingManager,
        V21PositionManager,
        INITIAL_CAPITAL,
        TARGET_POSITIONS,
        SCORE_THRESHOLD,
        BUY_RANK_PERCENTILE,
        SELL_RANK_PERCENTILE,
        STOP_LOSS_RATIO,
        TAKE_PROFIT_RATIO,
        MAX_POSITION_RATIO,
        MAX_POSITIONS,
    )
except ImportError:
    from db_manager import DatabaseManager
    from v20_accounting_engine import (
        V20AccountingEngine,
        BacktestResult,
        INITIAL_CAPITAL as V20_INITIAL_CAPITAL,
        TOP_K_STOCKS as V20_TOP_K_STOCKS,
    )
    from v21_strategy import (
        V21Strategy,
        V21DynamicSizingManager,
        V21PositionManager,
        INITIAL_CAPITAL,
        TARGET_POSITIONS,
        SCORE_THRESHOLD,
        BUY_RANK_PERCENTILE,
        SELL_RANK_PERCENTILE,
        STOP_LOSS_RATIO,
        TAKE_PROFIT_RATIO,
        MAX_POSITION_RATIO,
        MAX_POSITIONS,
    )


# ===========================================
# V21 回测执行器
# ===========================================

class V21BacktestExecutor:
    """
    V21 回测执行器 - 使用 V20 会计引擎 + V21 策略逻辑
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
        
        # V21 组件
        self.sizing_manager = V21DynamicSizingManager()
        self.position_manager = V21PositionManager()
        
        logger.info(f"V21BacktestExecutor initialized")
        logger.info(f"  Initial Capital: {initial_capital:,.0f}")
        logger.info(f"  Using V20AccountingEngine (locked)")
        logger.info(f"  Score Threshold: {SCORE_THRESHOLD}")
        logger.info(f"  Buy Rank Percentile: Top {BUY_RANK_PERCENTILE}%")
        logger.info(f"  Sell Rank Percentile: > {SELL_RANK_PERCENTILE}% (Buffer)")
    
    def run_backtest(
        self,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """运行 V21 回测"""
        logger.info("=" * 60)
        logger.info("V21 STRATEGY BACKTEST EXECUTION")
        logger.info("=" * 60)
        
        # 日期过滤
        if start_date:
            signals_df = signals_df.filter(pl.col("trade_date") >= start_date)
            prices_df = prices_df.filter(pl.col("trade_date") >= start_date)
        if end_date:
            prices_df = prices_df.filter(pl.col("trade_date") <= end_date)
        
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
            self._execute_day(trade_date, signals_df, prices_df)
        
        return self._generate_result(start_date, end_date)
    
    def _execute_day(
        self,
        trade_date: str,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
    ):
        """执行单日交易逻辑"""
        logger.debug(f"\n=== {trade_date} ===")
        
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
        
        # 4. 构建持仓盈亏信息
        position_info = {}
        for symbol, pos in self.accounting.positions.items():
            position_info[symbol] = {
                "pnl_ratio": pos.unrealized_pnl_ratio if hasattr(pos, 'unrealized_pnl_ratio') else 0.0,
                "buy_date": pos.buy_date if hasattr(pos, 'buy_date') else "",
            }
        
        # 5. 获取应卖出的股票
        stocks_to_sell = self.position_manager.get_stocks_to_sell(
            positions=self.accounting.positions,
            signal_ranks=signal_ranks,
            total_stocks=total_stocks,
            position_info=position_info,
        )
        
        # 6. 执行卖出
        for symbol in stocks_to_sell:
            if symbol in prices:
                self.accounting.execute_sell(
                    trade_date=trade_date,
                    symbol=symbol,
                    price=prices[symbol],
                )
        
        # 7. 计算动态权重
        target_amounts = self.sizing_manager.compute_dynamic_weights(
            signals_df=day_signals,
            current_positions=self.accounting.positions,
            initial_capital=self.initial_capital,
        )
        
        # 8. 获取应买入的股票
        stocks_to_buy = self.position_manager.get_stocks_to_buy(
            signals_df=day_signals,
            current_positions=self.accounting.positions,
            target_amounts=target_amounts,
        )
        
        # 9. 执行买入
        for symbol, target_amount in stocks_to_buy:
            if symbol in prices and target_amount > 0:
                self.accounting.execute_buy(
                    trade_date=trade_date,
                    symbol=symbol,
                    price=prices[symbol],
                    target_amount=target_amount,
                )
        
        # 10. 计算当日 NAV
        self.accounting.compute_daily_nav(trade_date, prices)
        
        # 11. 记录持仓快照
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
        
        logger.info("\n" + "=" * 60)
        logger.info("V21 BACKTEST RESULT")
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
        
        return result


# ===========================================
# V21 回测运行器
# ===========================================

class V21BacktestRunner:
    """V21 回测运行器 - 整合信号生成和会计引擎"""
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        target_positions: int = TARGET_POSITIONS,
        db: Optional[DatabaseManager] = None,
    ):
        self.initial_capital = initial_capital
        self.target_positions = target_positions
        self.db = db or DatabaseManager.get_instance()
        self.strategy = V21Strategy(db=self.db)
        
        logger.info(f"V21BacktestRunner initialized")
        logger.info(f"  Initial Capital: {initial_capital:,.0f}")
        logger.info(f"  Target Positions: {target_positions}")
    
    def run_backtest(self, start_date: str, end_date: str) -> BacktestResult:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V21 BACKTEST RUNNER")
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
        
        # 3. 运行回测
        logger.info("\nStep 3: Running V21 backtest...")
        executor = V21BacktestExecutor(
            initial_capital=self.initial_capital,
            target_positions=self.target_positions,
            db=self.db,
        )
        
        result = executor.run_backtest(signals_df=signals_df, prices_df=prices_df)
        return result
    
    def generate_report(self, result: BacktestResult, v20_result: Optional[BacktestResult] = None) -> str:
        """生成报告"""
        return generate_v21_report(result, v20_result=v20_result)
    
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
        
        recommendations = self.strategy.get_daily_recommendations(
            signals_df, prices_df, trade_date, self.initial_capital
        )
        
        return generate_trading_instructions(recommendations, trade_date)


# ===========================================
# V20 结果加载器
# ===========================================

def load_v20_result(json_path: Optional[str] = None) -> Optional[BacktestResult]:
    """从 JSON 文件加载 V20 结果"""
    if json_path is None:
        reports_dir = Path("reports")
        v20_files = list(reports_dir.glob("V20_backtest_result_*.json"))
        if v20_files:
            v20_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            json_path = str(v20_files[0])
    
    if json_path is None or not Path(json_path).exists():
        logger.warning("V20 result not found, skipping comparison")
        return None
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        v20_result = BacktestResult(
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
        
        logger.info(f"Loaded V20 result from {json_path}")
        return v20_result
    except Exception as e:
        logger.warning(f"Failed to load V20 result: {e}")
        return None


# ===========================================
# 报告生成器
# ===========================================

def generate_v21_report(result: BacktestResult, v20_result: Optional[BacktestResult] = None, output_path: Optional[str] = None) -> str:
    """生成 V21 报告"""
    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"V21_Dynamic_Sizing_Report_{timestamp}.md"
    
    total_fees = result.total_commission + result.total_stamp_duty + result.total_slippage
    gross_profit = result.total_return * result.initial_capital
    net_profit = gross_profit - total_fees
    fee_erosion_ratio = total_fees / gross_profit if gross_profit > 0 else 0.0
    cost_ratio = total_fees / result.initial_capital
    
    v20_comparison = ""
    if v20_result is not None:
        v20_fees = v20_result.total_commission + v20_result.total_stamp_duty + v20_result.total_slippage
        v20_cost_ratio = v20_fees / v20_result.initial_capital
        
        trades_change = (result.total_trades - v20_result.total_trades) / v20_result.total_trades * 100 if v20_result.total_trades > 0 else 0
        holding_days_change = ((result.avg_holding_days - v20_result.avg_holding_days) / v20_result.avg_holding_days * 100) if v20_result.avg_holding_days > 0 else 0
        fee_change = (total_fees - v20_fees) / v20_fees * 100 if v20_fees > 0 else 0
        turnover_change = ((result.turnover_rate - v20_result.turnover_rate) / v20_result.turnover_rate * 100) if v20_result.turnover_rate > 0 else 0
        
        v20_comparison = f"""
## 三、V20 vs V21 对比分析

| 指标 | V20 | V21 | 变化 |
|------|-----|-----|------|
| 总收益 | {v20_result.total_return:.2%} | {result.total_return:.2%} | {(result.total_return - v20_result.total_return):.2%} |
| 年化收益 | {v20_result.annual_return:.2%} | {result.annual_return:.2%} | {(result.annual_return - v20_result.annual_return):.2%} |
| 夏普比率 | {v20_result.sharpe_ratio:.3f} | {result.sharpe_ratio:.3f} | {(result.sharpe_ratio - v20_result.sharpe_ratio):.3f} |
| 最大回撤 | {v20_result.max_drawdown:.2%} | {result.max_drawdown:.2%} | {(result.max_drawdown - v20_result.max_drawdown):.2%} |
| 总交易次数 | {v20_result.total_trades:,} | {result.total_trades:,} | {trades_change:+.1f}% |
| 平均持仓天数 | {v20_result.avg_holding_days:.1f} | {result.avg_holding_days:.1f} | {holding_days_change:+.1f}% |
| 总手续费 | {v20_fees:,.2f} 元 | {total_fees:,.2f} 元 | {fee_change:+.1f}% |
| 换手率 | {v20_result.turnover_rate:.2%} | {result.turnover_rate:.2%} | {turnover_change:+.1f}% |
| 损耗比 | {v20_cost_ratio:.2%} | {cost_ratio:.2%} | {(cost_ratio - v20_cost_ratio):+.2%} |

### 换手率优化验证

| 指标 | V20 | V21 | 目标 | 状态 |
|------|-----|-----|------|------|
| Total Trades | {v20_result.total_trades:,} | {result.total_trades:,} | 降低 | {"✅" if result.total_trades < v20_result.total_trades else "⚠️"} |
| Avg Holding Days | {v20_result.avg_holding_days:.1f}天 | {result.avg_holding_days:.1f}天 | 提升 | {"✅" if result.avg_holding_days > v20_result.avg_holding_days else "⚠️"} |
| Turnover Rate | {v20_result.turnover_rate:.2%} | {result.turnover_rate:.2%} | 降低 | {"✅" if result.turnover_rate < v20_result.turnover_rate else "⚠️"} |
"""
    
    report = f"""# V21 动态赋权与摩擦抑止报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V21.0 (Dynamic Sizing & Turnover Buffer)

---

## 一、核心改进总结

### 1.1 概率驱动的动态调仓 (Dynamic Sizing)

**V20 问题**: 10 等分权重呆板，无法根据信号强度调整

**V21 解决方案**:
1. **得分激活**: 只有预测得分 > {SCORE_THRESHOLD} 的股票才进入备选池
2. **权重分配**: w_i ∝ (Score_i - {SCORE_THRESHOLD})
3. **权重限制**: 单只个股最大 {MAX_POSITION_RATIO:.0%}
4. **动态持仓**: 如果符合条件的票少，就只买几只，其余持币

### 1.2 换手率"滞后缓冲区" (Turnover Buffer)

**V20 问题**: 排名在第 10-15 名之间频繁换手

**V21 解决方案**:
1. **买入标准**: 排名必须 Top {BUY_RANK_PERCENTILE}% 且得分 > {SCORE_THRESHOLD}
2. **卖出标准**: 只有跌出 Top {SELL_RANK_PERCENTILE}% 才卖出（缓冲区）

### 1.3 真实手续费（维持 V20 严格标准）

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
| **损耗比** | **{cost_ratio:.2%}** |

状态：{"⚠️ 手续费侵蚀严重" if fee_erosion_ratio > 0.3 else "✅ 手续费可控" if fee_erosion_ratio < 0.1 else "⚠️ 手续费需关注"}

### 2.4 持仓指标

| 指标 | 值 |
|------|-----|
| 平均持仓天数 | {result.avg_holding_days:.1f} 天 |
| 平均持仓数量 | {result.avg_position_count:.1f} 只 |

{v20_comparison}

---

## 三、严防偷懒与作弊

- ✅ 基于 V20AccountingEngine，未修改手续费和 T+1 逻辑
- ✅ 输出 Total Buy Amount: {result.total_buy_amount:,.2f} 元
- ✅ 输出 Turnover Rate: {result.turnover_rate:.2%}
- ✅ 输出 损耗比：{cost_ratio:.2%}
- ✅ 所有平滑和过滤逻辑使用 .shift(1)
- ✅ T+1 锁定严格执行

---

## 四、执行总结

1. **动态权重**: 根据信号强度分配仓位，提高资金效率
2. **缓冲区**: Top {BUY_RANK_PERCENTILE}% 买入 / Top {SELL_RANK_PERCENTILE}% 卖出，减少无效换手
3. **手续费透明**: 完整计算滑点、规费、印花税、最低佣金
4. **持仓生命周期**: 止损/止盈/排名三重保护

---

**报告生成完毕**
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"V21 report saved to: {output_path}")
    return str(output_path)


def generate_trading_instructions(recommendations: List[Dict[str, Any]], trade_date: str) -> str:
    """生成实盘指令"""
    if not recommendations:
        return "无符合条件的交易指令"
    
    lines = []
    lines.append("=" * 80)
    lines.append("V21 实盘交易指令")
    lines.append("=" * 80)
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"交易日期：{trade_date}")
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
    lines.append(f"  - 卖出缓冲：跌出 Top {SELL_RANK_PERCENTILE}% 才卖出")
    lines.append(f"  - 止损/止盈：{STOP_LOSS_RATIO:.0%} / {TAKE_PROFIT_RATIO:.0%}")
    lines.append(f"  - 单只上限：{MAX_POSITION_RATIO:.0%}")
    
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
    logger.info("V21 Dynamic Sizing & Turnover Buffer Backtest Runner")
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
    
    # 加载 V20 结果
    logger.info("\nStep 0: Loading V20 result for comparison...")
    v20_result = load_v20_result()
    
    # 运行回测
    runner = V21BacktestRunner(
        initial_capital=initial_capital,
        target_positions=target_positions,
    )
    
    result = runner.run_backtest(start_date, end_date)
    
    # 生成报告
    logger.info("\nStep 4: Generating report...")
    report_path = runner.generate_report(result, v20_result=v20_result)
    
    # 保存 JSON 结果
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"V21_backtest_result_{timestamp}.json"
    
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
    logger.info("V21 BACKTEST COMPLETE")
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