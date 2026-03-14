"""
Backtest Engine - 多因子模型回测引擎，适配 FactorEngine 新架构。

功能:
    - 模拟过去 N 个交易日按 DailyTradeAdvisor 逻辑执行
    - 考虑大盘择时和 5 万元本金约束
    - 计算策略收益率、最大回撤
    - 对比"策略收益" vs "沪深 300 收益"
    - 输出 Markdown 表格报告
    - 【新增】直接使用 FactorEngine 的 predict_score（Z-Score 标准化值）
    - 【新增】多级排序规则：predict_score + sharpe_label
    - 【新增】防御模式动态阈值调整
    - 【新增】完善卖出风控（负分触发卖出）

使用示例:
    >>> python src/backtest_engine.py --days 30 --threshold 0.0
"""

import sys
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


# ===========================================
# 可配置参数 - 多因子模型阈值
# ===========================================
# Z-Score 阈值：0.0 表示强于平均水平
# predict_score 是 Z-Score 标准化后的值，均值为 0，标准差为 1
SCORE_THRESHOLD = 0.0  # 默认准入阈值（强于平均）
DEFENSIVE_THRESHOLD_ADDON = 0.3  # 防御模式额外阈值（+0.3 个标准差，更平滑过渡）

MIN_HOLD_DAYS = 2  # 最小持有天数（防止频繁换手）
DEFENSIVE_STOP_LOSS = -0.05  # 防御模式下止损线（-5%）

# 【新增 - 2026-03-14】评分融合权重
SCORE_WEIGHT = 0.7  # predict_score 权重
SHARPE_WEIGHT = 0.3  # hist_sharpe_20d 权重

# 【新增 - 2026-03-14】强制止损
HARD_STOP_LOSS = -0.05  # 持仓 2 天后收益<-5% 强制清仓


@dataclass
class BacktestConfig:
    """回测配置"""
    lookback_days: int = 30
    initial_capital: float = 50000.0
    max_positions: int = 3
    min_score: float = SCORE_THRESHOLD  # 最小预测分值阈值（Z-Score）
    commission_rate: float = 0.0003
    commission_min: float = 5.0
    stamp_duty_rate: float = 0.001
    use_mock_ai: bool = True
    test_mode: bool = True
    verbose: bool = True


@dataclass
class DailyRecord:
    """每日交易记录"""
    trade_date: str
    market_mode: str
    index_close: float
    index_ma20: float
    holdings: list[dict] = field(default_factory=list)
    holding_symbols: list[str] = field(default_factory=list)
    bought_symbols: list[str] = field(default_factory=list)
    sold_symbols: list[str] = field(default_factory=list)
    cash: float = 0.0
    portfolio_value: float = 0.0
    daily_return: float = 0.0
    daily_pnl: float = 0.0
    top_10_scores: list[dict] = field(default_factory=list)
    api_tokens: int = 0
    notes: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    start_date: str
    end_date: str
    total_days: int
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_daily_return: float
    daily_records: list[DailyRecord]
    benchmark_return: float
    total_trades: int
    total_commission: float
    total_stamp_duty: float


def format_pct_change(pct: float) -> str:
    """
    格式化百分比变化，确保显示在合理范围。
    
    Args:
        pct: 原始百分比（如 0.05 表示 5%）
        
    Returns:
        格式化后的字符串（如 "5.00%"）
    """
    # 截断到合理范围 [-50%, +50%]
    clipped = max(-0.50, min(0.50, pct))
    
    # 对于异常值，添加标记
    if pct > 0.50:
        return f">{50:.2f}%"
    elif pct < -0.50:
        return f"<{-50:.2f}%"
    else:
        return f"{clipped * 100:.2f}%"


class BacktestEngine:
    """
    回测引擎 - 适配多因子模型架构。
    
    核心变更:
        1. 删除概率映射函数：直接使用 predict_score（Z-Score 标准化值）
        2. 多级排序规则：predict_score（主）+ hist_sharpe_20d（次）
        3. 动态阈值调整：防御模式自动提高阈值 + hist_sharpe_20d > 0 约束
        4. 完善卖出风控：predict_score 降至负数触发卖出
        5. 拒绝未来函数：使用 hist_sharpe_20d（历史夏普）替代 sharpe_label（未来属性）
    """
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.db = DatabaseManager.get_instance()
        self.factor_engine: Optional[FactorEngine] = None
        self.cash = self.config.initial_capital
        self.portfolio_value = self.config.initial_capital
        self.holdings: dict[str, dict] = {}
        self.daily_records: list[DailyRecord] = []
        self.total_commission = 0.0
        self.total_stamp_duty = 0.0
        self.total_trades = 0
        
        # 初始化 FactorEngine
        try:
            self.factor_engine = FactorEngine("config/factors.yaml", validate=False)
            logger.info("FactorEngine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize FactorEngine: {e}")
        
        logger.info(f"BacktestEngine initialized: days={self.config.lookback_days}, "
                   f"capital={self.config.initial_capital}, "
                   f"score_threshold={self.config.min_score:.2f}")
    
    def get_trade_dates(self, end_date: str = None, days: int = 60) -> list[str]:
        """获取历史交易日列表"""
        if end_date is None:
            query = "SELECT MAX(trade_date) as max_date FROM stock_daily"
            result = self.db.read_sql(query)
            if result.is_empty():
                logger.error("Database is empty")
                return []
            end_date = str(result["max_date"][0])
        
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
        dates.sort()
        
        logger.info(f"Got {len(dates)} trade dates from {dates[0]} to {dates[-1]}")
        return dates
    
    def get_top_10_predictions(self, trade_date: str, lookback_days: int = 20) -> list[dict]:
        """
        获取当日所有股票的预测排名（前 10 名）用于诊断输出。
        
        【核心变更 - 2026-03-14 动态 Top-N 逻辑】
        1. 使用 FactorEngine 计算 predict_score 和 hist_sharpe_20d（历史夏普，无未来函数）
        2. 多级排序：predict_score（主序降序）+ hist_sharpe_20d（次序降序）
        3. 不再使用概率映射，直接使用原始 Z-Score 分值
        4. 【动态 Top-N】在满足阈值的股票中，仅取 Z-Score 排名最高且 hist_sharpe > 1.0 的前 2 名
        
        动态 Top-N 逻辑说明:
            - 不再使用固定的 > 0.1 门槛（过低导致过度交易）
            - 改为：在满足基础阈值的股票中，按 Z-Score 排序
            - 仅选取 hist_sharpe_20d > 1.0 的前 2 名（宁缺毋滥）
        """
        try:
            start_date = (datetime.strptime(trade_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            # 获取股票数据
            query = f"""
                SELECT 
                    s1.symbol,
                    s1.trade_date,
                    s1.open,
                    s1.high,
                    s1.low,
                    s1.close,
                    s1.volume,
                    s1.pct_chg,
                    s1.pre_close
                FROM stock_daily s1
                WHERE s1.trade_date = '{trade_date}'
                AND s1.volume > 0
                AND s1.close > 0
            """
            
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning(f"No stock data found for {trade_date}")
                return []
            
            logger.debug(f"Got {len(df)} stocks for prediction scoring")
            
            # 使用 FactorEngine 计算因子和评分
            if self.factor_engine:
                try:
                    # 按 symbol 分组，为每只股票计算因子
                    # 需要获取每只股票的历史数据来计算因子
                    results = []
                    
                    for symbol in df["symbol"].unique().limit(100).to_list():  # 限制处理数量
                        symbol_df = df.filter(pl.col("symbol") == symbol)
                        
                        # 获取该股票的历史数据用于因子计算
                        history_query = f"""
                            SELECT symbol, trade_date, open, high, low, close, volume, pct_chg, pre_close
                            FROM stock_daily
                            WHERE symbol = '{symbol}'
                            AND trade_date <= '{trade_date}'
                            ORDER BY trade_date DESC
                            LIMIT 60
                        """
                        history_df = self.db.read_sql(history_query)
                        
                        if len(history_df) < 30:
                            continue
                        
                        # 反转数据为时间正序
                        history_df = history_df.sort("trade_date")
                        
                        # 计算因子和评分
                        result_df = self.factor_engine.compute_factors(history_df)
                        
                        # 获取最新一行的评分
                        latest_row = result_df.tail(1)
                        
                        if latest_row.is_empty():
                            continue
                        
                        # 安全获取字段值，处理可能的缺失
                        # 【重要】使用 hist_sharpe_20d 替代 sharpe_label（无未来函数）
                        predict_score = self._safe_get_value(latest_row, "predict_score", 0.0)
                        hist_sharpe = self._safe_get_value(latest_row, "hist_sharpe_20d", 0.0)
                        rsi_14 = self._safe_get_value(latest_row, "rsi_14", 50.0)
                        volume_price_health = self._safe_get_value(latest_row, "volume_price_health", 0.0)
                        close = self._safe_get_value(latest_row, "close", 0.0)
                        pct_chg = self._safe_get_value(latest_row, "pct_chg", 0.0)
                        
                        results.append({
                            "symbol": symbol,
                            "close": close,
                            "pct_chg": pct_chg,
                            "predict_score": predict_score,
                            "hist_sharpe_20d": hist_sharpe,
                            "sharpe_label": hist_sharpe,  # 兼容旧代码，实际使用的是 hist_sharpe_20d
                            "rsi_14": rsi_14,
                            "volume_price_health": volume_price_health,
                        })
                    
                    # ========== 【动态 Top-N 逻辑 - 2026-03-14 评分融合】 ==========
                    # 【修复目标】移除 hist_sharpe_20d > 0 的硬过滤，改用评分融合
                    # 1. 计算融合分数：combined_score = 0.7*predict_score + 0.3*hist_sharpe
                    # 2. 按融合分数排序选取前 2 名
                    
                    # 第一步：获取当前阈值（考虑防御模式调整）
                    current_threshold = self._get_current_threshold(trade_date)
                    
                    # 第二步：【正向选股逻辑】按 predict_score 降序排序（选取最高分）
                    results.sort(key=lambda x: x["predict_score"], reverse=True)
                    logger.info(f"[FORWARD SIGNAL] Sorting by predict_score DESC (selecting highest scores)")
                    
                    # 第三步：取前 10 名
                    top_10_by_score = results[:10]
                    
                    logger.info(f"[DYNAMIC TOP-N] Top 10 by predict_score (threshold={current_threshold:.2f}):")
                    for i, stock in enumerate(top_10_by_score, 1):
                        logger.info(f"  #{i} {stock['symbol']}: Score={stock['predict_score']:+.2f}, Sharpe={stock['hist_sharpe_20d']:.2f}")
                    
                    # 第四步：【评分融合】combined_score = 0.7*predict_score + 0.3*hist_sharpe
                    for stock in top_10_by_score:
                        combined_score = SCORE_WEIGHT * stock["predict_score"] + SHARPE_WEIGHT * stock["hist_sharpe_20d"]
                        stock["combined_score"] = combined_score
                        logger.debug(f"  {stock['symbol']}: combined={combined_score:+.2f} (0.7*{stock['predict_score']:+.2f} + 0.3*{stock['hist_sharpe_20d']:+.2f})")
                    
                    # 第五步：【正向选股】按融合分数降序排序选取
                    top_10_by_score.sort(key=lambda x: x["combined_score"], reverse=True)
                    logger.info(f"[FORWARD SIGNAL] Selecting {len(top_10_by_score[:2])} stocks with HIGHEST combined scores")
                    
                    top_2 = top_10_by_score[:2]
                    top_10 = top_2
                    
                    logger.info(f"[DYNAMIC TOP-N] Selected {len(top_10)} stocks (by combined score):")
                    for i, stock in enumerate(top_10, 1):
                        logger.info(f"  #{i} {stock['symbol']}: Combined={stock['combined_score']:+.2f}, Score={stock['predict_score']:+.2f}, Sharpe={stock['hist_sharpe_20d']:.2f}")
                    
                    # 添加是否达到阈值的标记（基于融合分数）
                    for item in top_10:
                        item["meets_threshold"] = item["combined_score"] >= current_threshold
                    
                    return top_10
                    
                except Exception as e:
                    logger.error(f"Failed to compute factors: {e}")
                    return self._fallback_scoring(df, lookback_days)
            else:
                return self._fallback_scoring(df, lookback_days)
            
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            return []
    
    def _safe_get_value(self, df: pl.DataFrame, column: str, default: float = 0.0) -> float:
        """安全获取 DataFrame 列值，处理缺失列和 null 值"""
        try:
            if column in df.columns:
                val = df[column][0]
                if val is None:
                    return default
                return float(val)
            return default
        except (IndexError, TypeError, ValueError):
            return default
    
    def _fallback_scoring(self, df: pl.DataFrame, lookback_days: int) -> list[dict]:
        """
        降级评分逻辑（当 FactorEngine 不可用时使用）。
        
        使用简单的动量 + 成交量评分。
        """
        df = df.with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("pct_chg").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        ])
        
        # 简单评分：当日涨跌幅 + 成交量因子
        df = df.with_columns(
            (pl.col("pct_chg").fill_null(0.0) * 0.7 + 
             pl.col("volume").fill_null(0.0).rank() / len(df) * 0.3).alias("predict_score")
        )
        
        df = df.sort("predict_score", descending=True).head(10)
        
        results = []
        current_threshold = self._get_current_threshold("")
        
        for row in df.iter_rows(named=True):
            results.append({
                "symbol": row["symbol"],
                "close": float(row["close"]) if row["close"] else 0.0,
                "pct_chg": float(row["pct_chg"]) if row["pct_chg"] else 0.0,
                "predict_score": float(row["predict_score"]) if row["predict_score"] else 0.0,
                "sharpe_label": 0.0,
                "rsi_14": 50.0,
                "volume_price_health": 0.0,
                "meets_threshold": float(row["predict_score"]) >= current_threshold
            })
        
        return results
    
    def _get_current_threshold(self, trade_date: str) -> float:
        """
        获取当前准入阈值。
        
        逻辑:
        - 正常模式：使用基础阈值（默认 0.0）
        - 防御模式：基础阈值 + 0.5（更严格筛选）
        
        Args:
            trade_date: 交易日期（用于获取大盘状态）
            
        Returns:
            当前适用的阈值
        """
        base_threshold = self.config.min_score
        
        # 检查市场状态
        market_mode = self._get_market_mode(trade_date)
        
        if market_mode == "DEFENSIVE":
            # 防御模式：提高阈值
            current_threshold = base_threshold + DEFENSIVE_THRESHOLD_ADDON
            logger.info(f"[THRESHOLD] Defensive mode: threshold raised from {base_threshold:.2f} to {current_threshold:.2f}")
        else:
            current_threshold = base_threshold
        
        return current_threshold
    
    def _get_market_mode(self, trade_date: str) -> str:
        """获取市场状态（NORMAL / DEFENSIVE）"""
        index_data = self._get_index_data(trade_date)
        if index_data:
            close = index_data.get("close", 0)
            ma20 = index_data.get("ma20", 0)
            if close < ma20:
                return "DEFENSIVE"
        return "NORMAL"
    
    def simulate_daily_decision(self, trade_date: str, prev_date: str = None) -> DailyRecord:
        """
        模拟单日的 DailyTradeAdvisor 决策。
        
        【核心变更】
        1. 防御模式动态阈值：自动提高 +0.5
        2. 卖出风控增强：predict_score 降至负数也触发卖出
        3. 日志格式更新：显示 Score/Sharpe/RSI
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
            # Step 1: 获取大盘状态
            index_data = self._get_index_data(trade_date)
            if index_data:
                record.index_close = float(index_data["close"])
                record.index_ma20 = float(index_data["ma20"]) if index_data.get("ma20") else 0.0
                
                if record.index_close < record.index_ma20:
                    record.market_mode = "DEFENSIVE"
                    record.notes = "大盘低于均线，防守模式"
                else:
                    record.market_mode = "NORMAL"
            
            # 获取当前阈值（考虑防御模式调整）
            current_threshold = self._get_current_threshold(trade_date)
            
            # Step 2: 获取预测分值
            if self.config.verbose:
                record.top_10_scores = self.get_top_10_predictions(trade_date)
                
                logger.info(f"[PREDICT] Top 10 scores (threshold={current_threshold:.2f}):")
                for i, score in enumerate(record.top_10_scores, 1):
                    status = "OK" if score["meets_threshold"] else "NO"
                    # 新日志格式：Score | Sharpe | RSI
                    rsi_14 = score.get("rsi_14", 50)
                    if rsi_14 > 75:
                        rsi_status = "OVERBOUGHT"
                    elif rsi_14 > 70:
                        rsi_status = "WARM"
                    else:
                        rsi_status = "OK"
                    logger.info(
                        f"  #{i} {score['symbol']}: Score: {score['predict_score']:+.2f} | "
                        f"Sharpe: {score.get('sharpe_label', 0):.2f} | "
                        f"RSI: {rsi_14:.0f} [{rsi_status}] [{status}]"
                    )
                
                meets_count = sum(1 for s in record.top_10_scores if s["meets_threshold"])
                logger.info(f"  Meets threshold: {meets_count}/10 stocks")
            
            # Step 3: 处理持仓卖出
            to_sell = []
            
            for symbol, pos in list(self.holdings.items()):
                price_data = self._get_stock_price(symbol, trade_date)
                if price_data:
                    current_price = float(price_data["close"])
                    buy_price = pos.get("buy_price", current_price)
                    buy_date = pos.get("buy_date", "")
                    
                    # 计算持有天数
                    hold_days = 1
                    if buy_date and prev_date:
                        try:
                            buy_dt = datetime.strptime(buy_date, '%Y-%m-%d')
                            curr_dt = datetime.strptime(trade_date, '%Y-%m-%d')
                            hold_days = (curr_dt - buy_dt).days
                        except ValueError:
                            hold_days = 1
                    
                    # 计算当前盈亏比例
                    pnl_pct = (current_price - buy_price) / buy_price if buy_price > 0 else 0
                    
                    should_sell = False
                    sell_reason = ""
                    
                    # 【新增 - 2026-03-14】强制止损逻辑：持仓 2 天后收益<-5% 强制清仓
                    if hold_days >= 2 and pnl_pct < HARD_STOP_LOSS:
                        should_sell = True
                        sell_reason = f"hard stop loss ({pnl_pct:.1%} < {HARD_STOP_LOSS:.1%} after {hold_days} days)"
                        logger.info(f"  [HARD STOP] {symbol}: {sell_reason}")
                        to_sell.append((symbol, pos, current_price))
                        continue
                    
                    # 检查是否达到最小持有天数
                    if hold_days < MIN_HOLD_DAYS:
                        logger.debug(f"  [HOLD] {symbol}: only held {hold_days} days (min={MIN_HOLD_DAYS})")
                        continue
                    
                    # 获取该股票的当前评分
                    stock_score = 0.0
                    for s in record.top_10_scores:
                        if s["symbol"] == symbol:
                            stock_score = s["predict_score"]
                            break
                    
                    if record.market_mode == "DEFENSIVE":
                        # 【增强】防御模式下的卖出逻辑
                        # 1. 预测概率低于阈值
                        # 2. 达到止损线
                        # 3. 【新增】predict_score 降至负数（弱于平均）
                        
                        # 条件 1: 预测分值低于阈值
                        if stock_score < current_threshold:
                            should_sell = True
                            sell_reason = f"low score ({stock_score:+.2f} < {current_threshold:.2f})"
                        
                        # 条件 2: 达到止损线
                        if pnl_pct <= DEFENSIVE_STOP_LOSS:
                            should_sell = True
                            sell_reason = f"stop loss ({pnl_pct:.1%} <= {DEFENSIVE_STOP_LOSS:.1%})"
                        
                        # 条件 3: 【新增】predict_score 降至负数
                        if stock_score < 0 and not should_sell:
                            should_sell = True
                            sell_reason = f"negative score ({stock_score:+.2f})"
                        
                        if should_sell:
                            logger.info(f"  [DEFENSIVE SELL] {symbol}: {sell_reason}")
                    else:
                        # 【增强】正常模式下的卖出逻辑
                        # 如果 predict_score 降至负数，也触发卖出
                        if stock_score < 0:
                            should_sell = True
                            sell_reason = f"negative score ({stock_score:+.2f})"
                        
                        if should_sell:
                            logger.info(f"  [SELL] {symbol}: {sell_reason}")
                    
                    if should_sell:
                        to_sell.append((symbol, pos, current_price))
            
            # 执行卖出
            for symbol, pos, current_price in to_sell:
                sell_value = current_price * pos["shares"]
                commission = max(sell_value * self.config.commission_rate, self.config.commission_min)
                stamp_duty = sell_value * self.config.stamp_duty_rate
                
                self.cash += sell_value - commission - stamp_duty
                self.total_commission += commission
                self.total_stamp_duty += stamp_duty
                self.total_trades += 1
                
                record.sold_symbols.append(symbol)
                del self.holdings[symbol]
                logger.info(f"  [SOLD] {symbol} @ {current_price:.2f} x {pos['shares']} = ¥{sell_value:.2f}")
            
            # Step 4: 选股买入逻辑
            # 【重构 - 2026-03-14】降低选股门槛，增加交易频率
            # 移除 hist_sharpe_20d > 0 的硬过滤，仅使用评分融合排序
            if record.market_mode == "DEFENSIVE":
                # 防御模式：仅要求 predict_score > threshold（不再强制 hist_sharpe > 0）
                logger.info("  [DEFENSIVE] Using score-based selection (no hist_sharpe filter)")
                if record.top_10_scores:
                    # 筛选：仅要求 predict_score > threshold
                    qualified_stocks = [
                        s for s in record.top_10_scores 
                        if s["meets_threshold"]
                    ]
                    logger.info(f"  [DEFENSIVE SCAN] Qualified stocks (score>={current_threshold:.2f}): {len(qualified_stocks)}")
                    
                    if not qualified_stocks:
                        # 放宽到前 2 名
                        qualified_stocks = record.top_10_scores[:2]
                        logger.info(f"  [DEFENSIVE] No stocks meet threshold, using top 2 by combined score")
                else:
                    qualified_stocks = []
            elif len(self.holdings) < self.config.max_positions:
                if record.top_10_scores:
                    # 使用当前阈值筛选
                    qualified_stocks = [s for s in record.top_10_scores if s["meets_threshold"]]
                    
                    # 如果没有满足阈值的股票，取前 2 名
                    if not qualified_stocks:
                        qualified_stocks = record.top_10_scores[:2]
                        logger.info(f"  [SCAN] No stocks meet threshold, using top 2 by combined score")
                    
                    logger.info(f"  [SCAN] Qualified stocks: {len(qualified_stocks)} (threshold={current_threshold:.2f})")
                    
                    if qualified_stocks:
                        positions_left = self.config.max_positions - len(self.holdings)
                        budget_per_stock = self.cash / positions_left * 0.95
                        
                        logger.info(f"  [BUDGET] ¥{budget_per_stock:.2f} per stock (cash: ¥{self.cash:.2f}, positions left: {positions_left})")
                        
                        for stock in qualified_stocks:
                            if len(self.holdings) >= self.config.max_positions:
                                logger.info(f"  [LIMIT] Max positions ({self.config.max_positions}) reached")
                                break
                            
                            symbol = stock["symbol"]
                            close = stock["close"]
                            
                            # 【新增】RSI 超买过滤：如果 rsi_14 > 75，直接跳过
                            rsi_14 = stock.get("rsi_14", 50)
                            if rsi_14 > 75:
                                logger.info(f"  [SKIP] {symbol}: RSI={rsi_14:.0f} > 75 [OVERBOUGHT]")
                                continue
                            elif rsi_14 > 70:
                                logger.debug(f"  [WARN] {symbol}: RSI={rsi_14:.0f} [WARM] - Proceeding with caution")
                            
                            if symbol in self.holdings:
                                logger.debug(f"  [SKIP] {symbol}: already holding")
                                continue
                            
                            # 检查价格有效性
                            if close is None or close <= 0:
                                logger.warning(f"  [SKIP] {symbol}: invalid close price ({close})")
                                continue
                            
                            # 计算买入股数
                            raw_shares = int(budget_per_stock / close)
                            shares = (raw_shares // 100) * 100
                            
                            logger.debug(f"  [CALC] {symbol}: close=¥{close:.2f}, raw_shares={raw_shares}, rounded={shares}")
                            
                            if shares < 100:
                                logger.info(f"  [SKIP] {symbol}: not enough budget for 100 shares (need ¥{close*100:.2f})")
                                continue
                            
                            # 计算买入成本
                            buy_value = shares * close
                            commission = max(buy_value * self.config.commission_rate, self.config.commission_min)
                            total_cost = buy_value + commission
                            
                            if total_cost > self.cash:
                                logger.warning(f"  [SKIP] {symbol}: insufficient cash (need ¥{total_cost:.2f}, have ¥{self.cash:.2f})")
                                continue
                            
                            # 执行买入
                            self.cash -= total_cost
                            self.total_commission += commission
                            self.total_trades += 1
                            
                            self.holdings[symbol] = {
                                "shares": shares,
                                "buy_price": close,
                                "buy_date": trade_date
                            }
                            
                            record.bought_symbols.append(symbol)
                            logger.info(f"  [BUY] {symbol} @ ¥{close:.2f} x {shares} = ¥{buy_value:.2f} (fee: ¥{commission:.2f})")
            
            # Step 5: 计算组合价值
            prev_portfolio_value = self.portfolio_value
            self.portfolio_value = self.cash
            
            for symbol, pos in self.holdings.items():
                price_data = self._get_stock_price(symbol, trade_date)
                if price_data:
                    self.portfolio_value += float(price_data["close"]) * pos["shares"]
            
            record.cash = self.cash
            record.portfolio_value = self.portfolio_value
            record.holding_symbols = list(self.holdings.keys())
            record.daily_pnl = self.portfolio_value - prev_portfolio_value
            
            if prev_portfolio_value > 0:
                record.daily_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
            
            record.holdings = [
                {"symbol": s, "shares": p["shares"], "buy_price": p["buy_price"], "buy_date": p.get("buy_date", "")}
                for s, p in self.holdings.items()
            ]
            
            # 日志输出：每日总结
            logger.info(f"  [DAILY SUMMARY] Date={trade_date}, Mode={record.market_mode}")
            logger.info(f"  [DAILY SUMMARY] Portfolio Value: ¥{self.portfolio_value:,.2f}, Cash: ¥{self.cash:,.2f}")
            logger.info(f"  [DAILY SUMMARY] Holdings: {record.holding_symbols}")
            logger.info(f"  [DAILY SUMMARY] Daily P&L: ¥{record.daily_pnl:+,.2f} ({record.daily_return:+.2%})")
            
        except Exception as e:
            logger.error(f"Error simulating {trade_date}: {e}")
            record.notes += f" Error: {e}"
        
        return record
    
    def _get_index_data(self, trade_date: str) -> Optional[dict]:
        """获取指数数据（中证 500）"""
        try:
            query = f"""
                SELECT trade_date, close, ma20
                FROM index_daily
                WHERE symbol = '000905.SH'
                ORDER BY trade_date DESC
                LIMIT 30
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
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
            
            df = result.sort("trade_date")
            if "ma20" not in df.columns:
                df = df.with_columns(
                    pl.col("close").rolling_mean(window_size=20).alias("ma20")
                )
            
            latest = df.filter(pl.col("trade_date").cast(pl.Utf8) == trade_date)
            
            if latest.is_empty():
                latest = df.tail(1)
            
            if latest.is_empty():
                return None
            
            ma20_value = None
            if "ma20" in latest.columns:
                ma20_raw = latest["ma20"][0]
                if ma20_raw is not None:
                    ma20_value = float(ma20_raw) if not isinstance(ma20_raw, Decimal) else float(ma20_raw)
            
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
        """运行回测"""
        logger.info("=" * 60)
        logger.info("BACKTEST ENGINE (Multi-Factor Model) - Starting")
        logger.info(f"Lookback days: {self.config.lookback_days}")
        logger.info(f"Initial capital: {self.config.initial_capital:,.2f}")
        logger.info(f"Score Threshold: {self.config.min_score:.2f} (Z-Score)")
        logger.info(f"Defensive Threshold Addon: +{DEFENSIVE_THRESHOLD_ADDON:.2f}")
        logger.info(f"Min Hold Days: {MIN_HOLD_DAYS}")
        logger.info(f"Defensive Stop Loss: {DEFENSIVE_STOP_LOSS:.1%}")
        logger.info("=" * 60)
        
        trade_dates = self.get_trade_dates(days=self.config.lookback_days)
        
        if len(trade_dates) < self.config.lookback_days:
            logger.warning(f"Only {len(trade_dates)} days available, adjusting...")
        
        trade_dates = trade_dates[-self.config.lookback_days:]
        
        if not trade_dates:
            logger.error("No trade dates available")
            return self._empty_result()
        
        start_date = trade_dates[0]
        end_date = trade_dates[-1]
        
        logger.info(f"Backtest period: {start_date} to {end_date}")
        logger.info(f"Total trading days: {len(trade_dates)}")
        
        prev_date = None
        for i, date in enumerate(trade_dates, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"[{i}/{len(trade_dates)}] Simulating {date}...")
            record = self.simulate_daily_decision(date, prev_date)
            self.daily_records.append(record)
            prev_date = date
            
            if record.bought_symbols or record.sold_symbols:
                logger.info(f"  [TRADES] Buys: {record.bought_symbols}, Sells: {record.sold_symbols}")
                logger.info(f"  [PORTFOLIO] Value: ¥{record.portfolio_value:,.0f} (Cash: ¥{record.cash:,.0f})")
        
        metrics = self._calculate_metrics()
        benchmark_return = self._calculate_benchmark_return(start_date, end_date)
        
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Return: {metrics['total_return']:.2%}")
        logger.info(f"Benchmark Return: {benchmark_return:.2%}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Total Trades: {self.total_trades}")
        
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
        
        values = [r.portfolio_value for r in self.daily_records]
        returns = []
        
        for i in range(1, len(values)):
            if values[i-1] > 0:
                daily_ret = (values[i] - values[i-1]) / values[i-1]
                returns.append(daily_ret)
        
        if not returns:
            return self._empty_metrics()
        
        returns_arr = np.array(returns)
        total_return = (values[-1] - self.config.initial_capital) / self.config.initial_capital
        days = len(self.daily_records)
        annualized_return = (1 + total_return) ** (252 / max(days, 1)) - 1
        
        peak = values[0]
        max_dd = 0.0
        for v in values:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        
        mean_ret = float(np.mean(returns_arr))
        std_ret = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0
        
        if std_ret > 1e-10:
            sharpe = (mean_ret * 252 - 0.03) / std_ret
        else:
            sharpe = 0.0
        
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
                return 0.0
            
            closes = result["close"].to_list()
            if len(closes) < 2:
                return 0.0
            
            start_close = float(closes[0]) if closes[0] else 0.0
            end_close = float(closes[-1]) if closes[-1] else 0.0
            
            if start_close > 0:
                return (end_close - start_close) / start_close
            return 0.0
            
        except Exception as e:
            logger.error(f"Failed to calculate benchmark return: {e}")
            return 0.0
    
    def _empty_result(self) -> BacktestResult:
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
        
        lines.append("# Backtest Report (Multi-Factor Model)")
        lines.append("")
        lines.append(f"**Period**: {result.start_date} to {result.end_date}")
        lines.append(f"**Trading Days**: {result.total_days}")
        lines.append(f"**Initial Capital**: ¥{result.config.initial_capital:,.2f}")
        lines.append(f"**Score Threshold**: {result.config.min_score:.2f} (Z-Score)")
        lines.append(f"**Defensive Threshold Addon**: +{DEFENSIVE_THRESHOLD_ADDON:.2f}")
        lines.append(f"**Min Hold Days**: {MIN_HOLD_DAYS}")
        lines.append("")
        
        lines.append("## Performance Summary")
        lines.append("")
        lines.append("| Metric | Strategy | Benchmark |")
        lines.append("|--------|----------|-----------|")
        lines.append(f"| **Total Return** | {result.total_return:.2%} | {result.benchmark_return:.2%} |")
        lines.append(f"| **Annualized Return** | {result.annualized_return:.2%} | - |")
        lines.append(f"| **Max Drawdown** | {result.max_drawdown:.2%} | - |")
        lines.append(f"| **Sharpe Ratio** | {result.sharpe_ratio:.2f} | - |")
        lines.append(f"| **Win Rate** | {result.win_rate:.2%} | - |")
        lines.append("")
        
        excess_return = result.total_return - result.benchmark_return
        lines.append(f"**Excess Return**: {excess_return:.2%} (Strategy - Benchmark)")
        lines.append("")
        
        lines.append("## Trade Statistics")
        lines.append("")
        lines.append(f"- **Total Trades**: {result.total_trades}")
        lines.append(f"- **Total Commission**: ¥{result.total_commission:.2f}")
        lines.append(f"- **Total Stamp Duty**: ¥{result.total_stamp_duty:.2f}")
        lines.append(f"- **Total Cost**: ¥{result.total_commission + result.total_stamp_duty:.2f}")
        lines.append(f"- **Cost Ratio**: {(result.total_commission + result.total_stamp_duty) / result.config.initial_capital:.2%}")
        lines.append("")
        
        lines.append("## Daily Records (Last 15 Days)")
        lines.append("")
        lines.append("| Date | Mode | Holdings | Buys | Sells | Portfolio | Daily Return |")
        lines.append("|------|------|----------|------|-------|-----------|--------------|")
        
        for record in result.daily_records[-15:]:
            buys = len(record.bought_symbols)
            sells = len(record.sold_symbols)
            lines.append(
                f"| {record.trade_date[5:]} | {record.market_mode[:4]} | {len(record.holding_symbols)} | "
                f"{buys} | {sells} | ¥{record.portfolio_value:,.0f} | {record.daily_return:+.2%} |"
            )
        
        lines.append("")
        lines.append("## Equity Curve Data")
        lines.append("")
        lines.append("```")
        lines.append("Date,Portfolio Value,Cash,Num Holdings,Daily P&L")
        for record in result.daily_records:
            lines.append(
                f"{record.trade_date},{record.portfolio_value:.2f},{record.cash:.2f},{len(record.holding_symbols)},{record.daily_pnl:.2f}"
            )
        lines.append("```")
        lines.append("")
        
        lines.append("---")
        lines.append("")
        lines.append("> Note: This backtest uses multi-factor model with Z-Score normalization.")
        lines.append("> - predict_score: Z-Score standardized value (mean=0, std=1)")
        lines.append("> - Threshold: 0.0 means above average")
        lines.append("> - Defensive mode adds +0.5 threshold for stricter filtering")
        
        return "\n".join(lines)


def run_backtest(
    days: int = 30,
    capital: float = 50000.0,
    threshold: float = None,
    output_dir: str = "reports",
) -> BacktestResult:
    """便捷函数：运行回测并生成报告"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    score_threshold = threshold if threshold is not None else SCORE_THRESHOLD
    
    config = BacktestConfig(
        lookback_days=days,
        initial_capital=capital,
        min_score=score_threshold,
        use_mock_ai=True,
        test_mode=True,
        verbose=True,
    )
    
    
    engine = BacktestEngine(config)
    result = engine.run()
    
    report = engine.generate_report(result)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = Path(output_dir) / f"backtest_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY (Multi-Factor Model)")
    print("=" * 60)
    print(f"Period: {result.start_date} to {result.end_date}")
    print(f"Score Threshold: {score_threshold:.2f} (Z-Score)")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Benchmark Return: {result.benchmark_return:.2%}")
    print(f"Excess Return: {result.total_return - result.benchmark_return:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Total Trades: {result.total_trades}")
    print("=" * 60)
    
    return result


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Engine (Multi-Factor Model)')
    parser.add_argument('--days', type=int, default=30, help='Backtest days')
    parser.add_argument('--capital', type=float, default=50000.0, help='Initial capital')
    parser.add_argument('--threshold', type=float, default=SCORE_THRESHOLD, help='Score threshold (Z-Score)')
    parser.add_argument('--output', type=str, default='reports', help='Output directory')
    
    args = parser.parse_args()
    
    # 【修复 - 2026-03-14】彻底删除反向选股逻辑，仅使用正向选股
    run_backtest(
        days=args.days, 
        capital=args.capital, 
        threshold=args.threshold,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()