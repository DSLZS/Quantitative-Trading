"""
Backtest Engine - 简易回测引擎，模拟 DailyTradeAdvisor 执行逻辑。

功能:
    - 模拟过去 N 个交易日按 DailyTradeAdvisor 逻辑执行
    - 考虑大盘择时和 5 万元本金约束
    - 计算策略收益率、最大回撤
    - 对比"策略收益" vs "沪深 300 收益"
    - 输出 Markdown 表格报告

使用示例:
    >>> python src/backtest_engine.py --days 30
    # 模拟过去 30 个交易日的回测
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field
from decimal import Decimal

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .daily_trade_advisor import DailyTradeAdvisor, RunMode
    from .factor_engine import FactorEngine
except ImportError:
    from db_manager import DatabaseManager
    from daily_trade_advisor import DailyTradeAdvisor, RunMode
    from factor_engine import FactorEngine


@dataclass
class BacktestConfig:
    """回测配置"""
    lookback_days: int = 30          # 回测天数
    initial_capital: float = 50000.0  # 初始资金 5 万元
    max_positions: int = 3           # 最大持仓数
    min_prob: float = 0.70           # 最小预测概率
    commission_rate: float = 0.0003  # 佣金率
    commission_min: float = 5.0      # 最低佣金
    stamp_duty_rate: float = 0.001   # 印花税率
    use_mock_ai: bool = True         # 使用 Mock AI（避免 API 调用）
    test_mode: bool = True           # 测试模式（固定 2 只股票）


@dataclass
class DailyRecord:
    """每日交易记录"""
    trade_date: str
    market_mode: str                 # NORMAL / DEFENSIVE
    index_close: float               # 中证 500 收盘价
    index_ma20: float                # 中证 500 20 日均线
    
    # 持仓信息
    holdings: list[dict] = field(default_factory=list)
    holding_symbols: list[str] = field(default_factory=list)
    
    # 交易信息
    bought_symbols: list[str] = field(default_factory=list)
    sold_symbols: list[str] = field(default_factory=list)
    
    # 资金信息
    cash: float = 0.0
    portfolio_value: float = 0.0
    daily_return: float = 0.0
    
    # 统计
    api_tokens: int = 0
    notes: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    # 配置信息
    config: BacktestConfig
    start_date: str
    end_date: str
    total_days: int
    
    # 绩效指标
    total_return: float              # 总收益率
    annualized_return: float         # 年化收益率
    max_drawdown: float              # 最大回撤
    sharpe_ratio: float              # 夏普比率
    win_rate: float                  # 胜率
    avg_daily_return: float          # 日均收益
    
    # 资金曲线
    daily_records: list[DailyRecord]
    
    # 对比基准
    benchmark_return: float          # 沪深 300 同期收益
    
    # 交易统计
    total_trades: int
    total_commission: float
    total_stamp_duty: float


class BacktestEngine:
    """
    回测引擎 - 模拟 DailyTradeAdvisor 执行逻辑
    
    核心流程:
        1. 获取历史交易日列表
        2. 逐日模拟 Advisor 决策
        3. 记录持仓和资金变化
        4. 计算绩效指标
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        初始化回测引擎
        
        Args:
            config: 回测配置
        """
        self.config = config or BacktestConfig()
        self.db = DatabaseManager.get_instance()
        
        # 资金状态
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        
        # 持仓状态
        self.holdings: dict[str, dict] = {}  # symbol -> {shares, buy_price, buy_date}
        
        # 交易记录
        self.daily_records: list[DailyRecord] = []
        
        # 统计
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_trades = 0
        
        logger.info(f"BacktestEngine initialized: days={self.config.lookback_days}, "
                   f"capital={self.config.initial_capital}")
    
    def get_trade_dates(self, end_date: str = None, days: int = 60) -> list[str]:
        """
        获取历史交易日列表
        
        Args:
            end_date: 结束日期
            days: 获取天数（多获取一些用于计算 MA20）
            
        Returns:
            交易日列表（按时间正序）
        """
        if end_date is None:
            # 获取数据库中最新日期
            query = "SELECT MAX(trade_date) as max_date FROM stock_daily"
            result = self.db.read_sql(query)
            if result.is_empty():
                logger.error("Database is empty")
                return []
            end_date = str(result["max_date"][0])
        
        # 获取交易日列表（多获取用于计算因子）
        query = f"""
            SELECT DISTINCT trade_date 
            FROM stock_daily 
            WHERE trade_date <= '{end_date}'
            ORDER BY trade_date DESC
            LIMIT {days + 30}
        """
        
        result = self.db.read_sql(query)
        if result.is_empty():
            logger.warning("No trade dates found")
            return []
        
        dates = [str(d) for d in result["trade_date"].to_list()]
        dates.sort()  # 正序排列
        
        logger.info(f"Got {len(dates)} trade dates from {dates[0]} to {dates[-1]}")
        return dates
    
    def simulate_daily_decision(self, trade_date: str) -> DailyRecord:
        """
        模拟单日的 DailyTradeAdvisor 决策
        
        注意：这里简化了 Advisor 逻辑，直接基于数据库数据模拟
        而不是完整运行 Advisor（避免 AI API 调用和模型推理）
        
        Args:
            trade_date: 交易日
            
        Returns:
            DailyRecord: 当日交易记录
        """
        record = DailyRecord(
            trade_date=trade_date,
            market_mode="NORMAL",
            index_close=0.0,
            index_ma20=0.0,
            cash=self.cash,
            portfolio_value=self.portfolio_value
        )
        
        try:
            # ========== Step 1: 获取大盘状态（中证 500） ==========
            index_data = self._get_index_data(trade_date)
            if index_data:
                record.index_close = float(index_data["close"])
                record.index_ma20 = float(index_data["ma20"]) if index_data.get("ma20") else 0.0
                
                # 大盘择时判断
                if record.index_close < record.index_ma20:
                    record.market_mode = "DEFENSIVE"
                    record.notes = "大盘低于均线，防守模式"
                else:
                    record.market_mode = "NORMAL"
            
            # ========== Step 2: 获取持仓股票当日价格 ==========
            holding_values = {}
            for symbol, pos in list(self.holdings.items()):
                price_data = self._get_stock_price(symbol, trade_date)
                if price_data:
                    current_price = float(price_data["close"])
                    holding_values[symbol] = current_price * pos["shares"]
                    
                    # 检查是否应该卖出（预测概率低于阈值或防守模式）
                    should_sell = False
                    
                    # 防守模式：清空所有持仓
                    if record.market_mode == "DEFENSIVE":
                        should_sell = True
                        record.notes += " 防守模式清仓;"
                    
                    # 执行卖出
                    if should_sell and symbol in self.holdings:
                        sell_value = current_price * pos["shares"]
                        
                        # 计算交易成本
                        commission = max(sell_value * self.config.commission_rate, self.config.commission_min)
                        stamp_duty = sell_value * self.config.stamp_duty_rate
                        
                        self.cash += sell_value - commission - stamp_duty
                        self.total_commission += commission
                        self.total_stamp_duty += stamp_duty
                        self.total_trades += 1
                        
                        record.sold_symbols.append(symbol)
                        del self.holdings[symbol]
                        logger.debug(f"  Sold {symbol} @ {current_price:.2f}")
            
            # ========== Step 3: 选股逻辑（简化版） ==========
            # 在真实 Advisor 中，这里会运行模型预测和 AI 审计
            # 回测中我们简化为：选取当日涨幅前 N 的股票（后视偏差，仅用于演示）
            
            if record.market_mode == "NORMAL" and len(self.holdings) < self.config.max_positions:
                # 获取当日所有股票数据
                query = f"""
                    SELECT symbol, close, pct_chg, volume
                    FROM stock_daily
                    WHERE trade_date = '{trade_date}'
                    AND pct_chg IS NOT NULL
                    AND volume > 0
                    ORDER BY pct_chg DESC
                    LIMIT 10
                """
                candidates = self.db.read_sql(query)
                
                if not candidates.is_empty():
                    # 计算可分配资金
                    positions_left = self.config.max_positions - len(self.holdings)
                    budget_per_stock = self.cash / positions_left * 0.95  # 留 5% 现金
                    
                    for row in candidates.iter_rows():
                        if len(self.holdings) >= self.config.max_positions:
                            break
                        
                        symbol = row[0]
                        close = float(row[1])
                        
                        # 跳过已持仓股票
                        if symbol in self.holdings:
                            continue
                        
                        # 计算买入股数（100 股整数倍）
                        raw_shares = int(budget_per_stock / close)
                        shares = (raw_shares // 100) * 100
                        
                        if shares >= 100:
                            # 计算买入成本
                            buy_value = shares * close
                            commission = max(buy_value * self.config.commission_rate, self.config.commission_min)
                            
                            # 检查现金是否足够
                            if buy_value + commission > self.cash:
                                continue
                            
                            # 执行买入
                            self.cash -= (buy_value + commission)
                            self.total_commission += commission
                            self.total_trades += 1
                            
                            self.holdings[symbol] = {
                                "shares": shares,
                                "buy_price": close,
                                "buy_date": trade_date
                            }
                            
                            record.bought_symbols.append(symbol)
                            logger.debug(f"  Bought {symbol} @ {close:.2f} x {shares}")
            
            # ========== Step 4: 计算组合价值 ==========
            self.portfolio_value = self.cash
            
            for symbol, pos in self.holdings.items():
                price_data = self._get_stock_price(symbol, trade_date)
                if price_data:
                    self.portfolio_value += float(price_data["close"]) * pos["shares"]
            
            record.cash = self.cash
            record.portfolio_value = self.portfolio_value
            record.holding_symbols = list(self.holdings.keys())
            
            # 计算日收益
            if self.daily_records:
                prev_value = self.daily_records[-1].portfolio_value
                if prev_value > 0:
                    record.daily_return = (self.portfolio_value - prev_value) / prev_value
            
            record.holdings = [
                {"symbol": s, "shares": p["shares"], "buy_price": p["buy_price"]}
                for s, p in self.holdings.items()
            ]
            
        except Exception as e:
            logger.error(f"Error simulating {trade_date}: {e}")
            record.notes += f" 错误：{e}"
        
        return record
    
    def _get_index_data(self, trade_date: str) -> Optional[dict]:
        """获取指数数据（中证 500）"""
        try:
            from datetime import date
            
            # 将字符串日期转换为 date 对象（数据库返回的是 date 类型）
            target_date = datetime.strptime(trade_date, '%Y-%m-%d').date()
            
            # 获取中证 500 数据（包含 MA20）
            query = f"""
                SELECT trade_date, close, ma20
                FROM index_daily
                WHERE symbol = '000905.SH'
                ORDER BY trade_date DESC
                LIMIT 30
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                # 尝试上证指数作为备选
                query = f"""
                    SELECT trade_date, close
                    FROM index_daily
                    WHERE symbol = '000001.SH'
                    ORDER BY trade_date DESC
                    LIMIT 30
                """
                result = self.db.read_sql(query)
            
            if result.is_empty():
                return None
            
            # 计算 MA20（如果不存在）
            df = result.sort("trade_date")
            if "ma20" not in df.columns:
                df = df.with_columns(
                    pl.col("close").rolling_mean(window_size=20).alias("ma20")
                )
            
            # 过滤出目标日期或之前的数据
            # 数据库返回的 trade_date 是 date 类型，需要转换为字符串比较
            latest = df.filter(
                pl.col("trade_date").cast(pl.Utf8) == trade_date
            )
            
            if latest.is_empty():
                # 如果没有当天的，取最近一天
                latest = df.tail(1)
            
            if latest.is_empty():
                return None
            
            ma20_value = None
            if "ma20" in latest.columns:
                ma20_raw = latest["ma20"][0]
                if ma20_raw is not None:
                    if isinstance(ma20_raw, Decimal):
                        ma20_value = float(ma20_raw)
                    else:
                        ma20_value = float(ma20_raw)
            
            return {
                "close": float(latest["close"][0]),
                "ma20": ma20_value
            }
            
        except Exception as e:
            logger.error(f"Failed to get index data: {e}")
            return None
    
    def _get_stock_price(self, symbol: str, trade_date: str) -> Optional[dict]:
        """获取股票价格"""
        try:
            query = f"""
                SELECT symbol, trade_date, close, open, high, low
                FROM stock_daily
                WHERE symbol = '{symbol}'
                AND trade_date = '{trade_date}'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return None
            
            return {
                "close": float(result["close"][0]),
                "open": float(result["open"][0]) if result["open"][0] else 0.0,
                "high": float(result["high"][0]) if result["high"][0] else 0.0,
                "low": float(result["low"][0]) if result["low"][0] else 0.0,
            }
            
        except Exception as e:
            logger.error(f"Failed to get stock price for {symbol}: {e}")
            return None
    
    def run(self) -> BacktestResult:
        """
        运行回测
        
        Returns:
            BacktestResult: 回测结果
        """
        logger.info("=" * 60)
        logger.info("BACKTEST ENGINE - Starting")
        logger.info(f"Lookback days: {self.config.lookback_days}")
        logger.info(f"Initial capital: {self.config.initial_capital:,.2f}")
        logger.info("=" * 60)
        
        # 获取交易日列表
        trade_dates = self.get_trade_dates(days=self.config.lookback_days)
        
        if len(trade_dates) < self.config.lookback_days:
            logger.warning(f"Only {len(trade_dates)} days available, adjusting...")
        
        # 只取需要的天数
        trade_dates = trade_dates[-self.config.lookback_days:]
        
        if not trade_dates:
            logger.error("No trade dates available")
            return self._empty_result()
        
        start_date = trade_dates[0]
        end_date = trade_dates[-1]
        
        logger.info(f"Backtest period: {start_date} to {end_date}")
        logger.info(f"Total trading days: {len(trade_dates)}")
        
        # 逐日模拟
        for i, date in enumerate(trade_dates, 1):
            logger.info(f"[{i}/{len(trade_dates)}] Simulating {date}...")
            record = self.simulate_daily_decision(date)
            self.daily_records.append(record)
            
            if record.bought_symbols or record.sold_symbols:
                logger.info(f"  Buys: {record.bought_symbols}, Sells: {record.sold_symbols}")
        
        # 计算绩效指标
        metrics = self._calculate_metrics()
        
        # 计算基准收益（沪深 300）
        benchmark_return = self._calculate_benchmark_return(start_date, end_date)
        
        logger.info("=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Benchmark Return: {benchmark_return:.2%}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        return BacktestResult(
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            total_days=len(trade_dates),
            total_return=metrics["total_return"],
            annualized_return=metrics["annualized_return"],
            max_drawdown=metrics["max_drawdown"],
            sharpe_ratio=metrics["sharpe_ratio"],
            win_rate=metrics["win_rate"],
            avg_daily_return=metrics["avg_daily_return"],
            daily_records=self.daily_records,
            benchmark_return=benchmark_return,
            total_trades=self.total_trades,
            total_commission=self.total_commission,
            total_stamp_duty=self.total_stamp_duty
        )
    
    def _calculate_metrics(self) -> dict:
        """计算绩效指标"""
        if not self.daily_records:
            return self._empty_metrics()
        
        # 提取资金曲线
        values = [r.portfolio_value for r in self.daily_records]
        returns = []
        
        for i in range(1, len(values)):
            if values[i-1] > 0:
                daily_ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(daily_ret)
        
        if not returns:
            return self._empty_metrics()
        
        returns_arr = np.array(returns)
        
        # 总收益率
        total_return = (values[-1] - self.config.initial_capital) / self.config.initial_capital
        
        # 年化收益率
        days = len(self.daily_records)
        annualized_return = (1 + total_return) ** (252 / max(days, 1)) - 1
        
        # 最大回撤
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        # 夏普比率
        mean_ret = float(np.mean(returns_arr))
        std_ret = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0
        
        if std_ret > 1e-10:
            sharpe = (mean_ret * 252 - 0.03) / std_ret  # 3% 无风险利率
        else:
            sharpe = 0.0
        
        # 胜率
        positive_days = sum(1 for r in returns if r > 0)
        win_rate = positive_days / len(returns) if returns else 0.0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "avg_daily_return": mean_ret
        }
    
    def _empty_metrics(self) -> dict:
        """返回空指标"""
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "avg_daily_return": 0.0
        }
    
    def _calculate_benchmark_return(self, start_date: str, end_date: str) -> float:
        """计算基准收益（沪深 300）"""
        try:
            query = f"""
                SELECT trade_date, close
                FROM index_daily
                WHERE symbol = '000300.SH'
                AND trade_date >= '{start_date}'
                AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                logger.warning("No benchmark data available")
                return 0.0
            
            closes = result["close"].to_list()
            if len(closes) < 2:
                return 0.0
            
            # 处理 Decimal 类型
            start_close = float(closes[0]) if closes[0] else 0.0
            end_close = float(closes[-1]) if closes[-1] else 0.0
            
            if start_close > 0:
                return (end_close - start_close) / start_close
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmark return: {e}")
            return 0.0
    
    def _empty_result(self) -> BacktestResult:
        """返回空结果"""
        return BacktestResult(
            config=self.config,
            start_date="",
            end_date="",
            total_days=0,
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            win_rate=0.0,
            avg_daily_return=0.0,
            daily_records=[],
            benchmark_return=0.0,
            total_trades=0,
            total_commission=0.0,
            total_stamp_duty=0.0
        )
    
    def generate_report(self, result: BacktestResult) -> str:
        """生成 Markdown 报告"""
        lines = []
        
        lines.append("# 📊 回测报告 - Backtest Report")
        lines.append("")
        lines.append(f"**回测期间**: {result.start_date} 至 {result.end_date}")
        lines.append(f"**交易天数**: {result.total_days} 天")
        lines.append(f"**初始资金**: ¥{result.config.initial_capital:,.2f}")
        lines.append("")
        
        # 绩效对比表
        lines.append("## 📈 绩效对比")
        lines.append("")
        lines.append("| 指标 | 策略值 | 基准值 |")
        lines.append("|------|--------|--------|")
        lines.append(f"| **总收益率** | {result.total_return:.2%} | {result.benchmark_return:.2%} |")
        lines.append(f"| **年化收益率** | {result.annualized_return:.2%} | - |")
        lines.append(f"| **最大回撤** | {result.max_drawdown:.2%} | - |")
        lines.append(f"| **夏普比率** | {result.sharpe_ratio:.2f} | - |")
        lines.append(f"| **胜率** | {result.win_rate:.2%} | - |")
        lines.append("")
        
        # 超额收益
        excess_return = result.total_return - result.benchmark_return
        lines.append(f"**超额收益**: {excess_return:.2%} (策略 - 基准)")
        lines.append("")
        
        # 交易统计
        lines.append("## 📝 交易统计")
        lines.append("")
        lines.append(f"- **总交易次数**: {result.total_trades}")
        lines.append(f"- **总佣金**: ¥{result.total_commission:.2f}")
        lines.append(f"- **总印花税**: ¥{result.total_stamp_duty:.2f}")
        lines.append(f"- **总交易成本**: ¥{result.total_commission + result.total_stamp_duty:.2f}")
        lines.append(f"- **成本率**: {(result.total_commission + result.total_stamp_duty) / result.config.initial_capital:.2%}")
        lines.append("")
        
        # 每日记录摘要
        lines.append("## 📅 每日记录摘要")
        lines.append("")
        lines.append("| 日期 | 市场模式 | 持仓数 | 买入 | 卖出 | 组合价值 | 日收益 |")
        lines.append("|------|----------|--------|------|------|----------|--------|")
        
        for record in result.daily_records[-15:]:  # 只显示最后 15 天
            buys = len(record.bought_symbols)
            sells = len(record.sold_symbols)
            lines.append(
                f"| {record.trade_date[5:]} | {record.market_mode[:4]} | {len(record.holding_symbols)} | "
                f"{buys} | {sells} | ¥{record.portfolio_value:,.0f} | {record.daily_return:+.2%} |"
            )
        
        lines.append("")
        
        # 资金曲线数据（可用于绘图）
        lines.append("## 📉 资金曲线数据")
        lines.append("")
        lines.append("```")
        lines.append("Date,Portfolio Value,Cash,Num Holdings")
        for record in result.daily_records:
            lines.append(
                f"{record.trade_date},{record.portfolio_value:.2f},{record.cash:.2f},{len(record.holding_symbols)}"
            )
        lines.append("```")
        lines.append("")
        
        # 风险提示
        lines.append("---")
        lines.append("")
        lines.append("> ⚠️ **风险提示**: 本回测使用简化逻辑，未考虑:")
        lines.append("> - 真实的模型预测和 AI 审计流程")
        lines.append("> - 股票停牌、涨跌停限制")
        lines.append("> - 流动性约束和冲击成本")
        lines.append("> - 实际交易中的滑点")
        lines.append("")
        lines.append("> 回测结果仅供参考，不代表实际收益。")
        
        return "\n".join(lines)


def run_backtest(
    days: int = 30,
    capital: float = 50000.0,
    output_dir: str = "reports"
) -> BacktestResult:
    """
    便捷函数：运行回测并生成报告
    
    Args:
        days: 回测天数
        capital: 初始资金
        output_dir: 报告输出目录
        
    Returns:
        BacktestResult: 回测结果
    """
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 创建配置
    config = BacktestConfig(
        lookback_days=days,
        initial_capital=capital,
        use_mock_ai=True,
        test_mode=True
    )
    
    # 运行回测
    engine = BacktestEngine(config)
    result = engine.run()
    
    # 生成报告
    report = engine.generate_report(result)
    
    # 保存报告
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(output_dir) / f"backtest_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    
    # 输出摘要
    print("\n" + "=" * 60)
    print("📊 回测结果摘要")
    print("=" * 60)
    print(f"回测期间：{result.start_date} 至 {result.end_date}")
    print(f"总收益率：{result.total_return:.2%}")
    print(f"沪深 300 收益：{result.benchmark_return:.2%}")
    print(f"超额收益：{result.total_return - result.benchmark_return:.2%}")
    print(f"最大回撤：{result.max_drawdown:.2%}")
    print(f"夏普比率：{result.sharpe_ratio:.2f}")
    print(f"总交易次数：{result.total_trades}")
    print("=" * 60)
    
    return result


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Engine - 简易回测引擎')
    parser.add_argument('--days', type=int, default=30, help='回测天数 (default: 30)')
    parser.add_argument('--capital', type=float, default=50000.0, help='初始资金 (default: 50000)')
    parser.add_argument('--output', type=str, default='reports', help='报告输出目录')
    
    args = parser.parse_args()
    
    run_backtest(days=args.days, capital=args.capital, output_dir=args.output)


if __name__ == "__main__":
    main()