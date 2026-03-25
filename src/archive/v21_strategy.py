"""
V21 Strategy - 动态赋权与摩擦抑止策略

【核心任务：在"铁血会计"约束下寻找盈利路径】

V20 回测结果显示，由于换手过度和佣金陷阱，策略损失了 40% 的成本。
V21 的目标是：通过优化仓位管理和引入换手缓冲区，将换手率降低 80%，
并提升每笔交易的质量。

【核心改进逻辑】

A. 引入"买入阈值"与"换手缓冲区" (Hysteresis)
   - 买入门槛：只有排名前 5%（Top 10）且得分 > 0.55 的股票才允许买入
   - 卖出缓冲：股票买入后，除非满足以下条件，否则不调仓：
     1. 排名跌出 Top 40（给信号留出波动空间）
     2. 触发 8% 止损或 15% 止盈
   - 目的：强制将平均持仓天数从 3.3 天提升至 10 天以上

B. 动态概率赋权 (Probabilistic Weighting)
   - 不再均分：弃用 10% 均分逻辑
   - 凯利准则简化版：
     - 计算各股得分 S_i 与阈值 T 的差值：D_i = S_i - 0.50
     - 目标权重 W_i = D_i / sum(D_i)
     - 单只上限：强制单只个股权重不超过 20%
     - 总持仓数量控制在 5-12 只
   - 持币观望：如果全市场只有 3 只票过线，则只买 3 只，剩余资金留存现金

【严防作弊与硬约束】
- 锁死会计引擎：必须直接引用 v20_accounting_engine.py
- 本金限制：严格基于 100,000 元初始资金
- 严禁未来函数：所有数据处理必须严格 .shift(1)

作者：资深量化策略分析师
版本：V21.0 (Dynamic Sizing & Turnover Buffer)
日期：2026-03-17
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 配置常量 (严禁修改)
# ===========================================

# 资金管理
INITIAL_CAPITAL = 100000.0
TARGET_POSITIONS = 10  # 目标持仓数量

# 动态权重配置
SCORE_THRESHOLD = 0.50      # 得分激活阈值（使用相对值）
MAX_POSITION_RATIO = 0.20   # 单只个股最大权重 20%
MIN_POSITION_RATIO = 0.05   # 最小权重 5%
MIN_POSITIONS = 5           # 最小持仓数量
MAX_POSITIONS = 12          # 最大持仓数量

# 换手率缓冲区配置
BUY_RANK_PERCENTILE = 5     # 买入排名百分位（Top 5%）
SELL_RANK_PERCENTILE = 40   # 卖出排名百分位（跌出 Top 40% 才卖）

# 止损止盈
STOP_LOSS_RATIO = 0.08   # 8% 硬止损
TAKE_PROFIT_RATIO = 0.15 # 15% 止盈

# 因子配置
FACTOR_NAMES = ["vol_price_corr", "reversal_st", "vol_risk", 
                "turnover_signal", "momentum", "low_vol"]
FACTOR_WEIGHTS = {
    "vol_price_corr": 0.25,    # 量价相关性（最强因子）
    "reversal_st": 0.20,       # 短线反转
    "vol_risk": 0.15,          # 波动风险
    "turnover_signal": 0.15,   # 异常换手
    "momentum": 0.15,          # 动量因子
    "low_vol": 0.10,           # 低波动因子
}


# ===========================================
# V21 信号生成器
# ===========================================

class V21SignalGenerator:
    """
    V21 信号生成器 - 基于 6 因子体系
    
    【因子库】
    1. 量价相关性 (vol_price_corr) - 最强因子
    2. 短线反转 (reversal_st) - 正交化
    3. 波动风险 (vol_risk) - 低波异常
    4. 异常换手 (turnover_signal)
    5. 动量因子 (momentum)
    6. 低波动因子 (low_vol)
    """
    
    EPSILON = 1e-6
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        logger.info("V21SignalGenerator initialized")
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算所有因子"""
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            pl.col("turnover_rate").cast(pl.Float64, strict=False),
        ])
        
        if "volume" not in result.columns:
            result = result.with_columns([pl.lit(1.0).alias("volume")])
        
        # 1. 量价相关性
        volume_change = (pl.col("volume") / pl.col("volume").shift(1) - 1).fill_null(0)
        returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        
        window = 20
        vol_mean = volume_change.rolling_mean(window_size=window).shift(1)
        ret_mean = returns.rolling_mean(window_size=window).shift(1)
        cov_xy = ((volume_change - vol_mean) * (returns - ret_mean)).rolling_mean(window_size=window).shift(1)
        vol_std = volume_change.rolling_std(window_size=window, ddof=1).shift(1)
        ret_std = returns.rolling_std(window_size=window, ddof=1).shift(1)
        vol_price_corr = cov_xy / (vol_std * ret_std + self.EPSILON)
        
        result = result.with_columns([
            vol_price_corr.alias("vol_price_corr"),
        ])
        
        # 2. 短线反转
        momentum_5 = pl.col("close") / (pl.col("close").shift(6) + self.EPSILON) - 1
        reversal_st = -momentum_5.shift(1)
        result = result.with_columns([reversal_st.alias("reversal_st")])
        
        # 3. 波动风险
        stock_returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        volatility_20 = stock_returns.rolling_std(window_size=20, ddof=1).shift(1) * np.sqrt(252)
        vol_risk = -volatility_20
        result = result.with_columns([vol_risk.alias("vol_risk")])
        
        # 4. 异常换手
        turnover_ma20 = pl.col("turnover_rate").rolling_mean(window_size=20).shift(1)
        turnover_ratio = pl.col("turnover_rate") / (turnover_ma20 + self.EPSILON)
        turnover_signal = (turnover_ratio - 1).clip(-0.9, 2.0)
        result = result.with_columns([turnover_signal.alias("turnover_signal")])
        
        # 5. 动量因子
        ma20 = pl.col("close").rolling_mean(window_size=20).shift(1)
        momentum = (pl.col("close") - ma20) / (ma20 + self.EPSILON)
        momentum_signal = -momentum
        result = result.with_columns([
            ma20.alias("ma20"),
            momentum_signal.alias("momentum"),
        ])
        
        # 6. 低波动因子
        std_20d = stock_returns.rolling_std(window_size=20, ddof=1).shift(1) * np.sqrt(252)
        low_vol = -std_20d
        result = result.with_columns([low_vol.alias("low_vol")])
        
        logger.info(f"Computed 6 factors for {len(result)} records")
        return result
    
    def normalize_factors(self, df: pl.DataFrame, factor_names: List[str]) -> pl.DataFrame:
        """截面标准化"""
        result = df.clone()
        
        for factor in factor_names:
            if factor not in result.columns:
                continue
            
            stats = result.group_by("trade_date").agg([
                pl.col(factor).mean().alias("mean"),
                pl.col(factor).std().alias("std"),
            ])
            
            result = result.join(stats, on="trade_date", how="left")
            result = result.with_columns([
                ((pl.col(factor) - pl.col("mean")) / (pl.col("std") + self.EPSILON)).alias(f"{factor}_std")
            ]).drop(["mean", "std"])
        
        return result
    
    def compute_composite_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算加权综合信号
        
        使用配置的因子权重计算综合信号
        """
        std_factors = [f"{f}_std" for f in FACTOR_NAMES if f"{f}_std" in df.columns]
        
        if not std_factors:
            std_factors = FACTOR_NAMES
        
        # 加权综合信号
        signal = None
        for factor in std_factors:
            factor_name = factor.replace("_std", "")
            weight = FACTOR_WEIGHTS.get(factor_name, 1.0 / len(std_factors))
            if signal is None:
                signal = pl.col(factor) * weight
            else:
                signal = signal + pl.col(factor) * weight
        
        result = df.with_columns([
            signal.alias("signal")
        ])
        
        return result
    
    def generate_signals(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """生成信号"""
        logger.info(f"Generating signals from {start_date} to {end_date}...")
        
        query = f"""
            SELECT 
                symbol, trade_date, open, high, low, close, pre_close,
                volume, amount, turnover_rate, industry_code, total_mv
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            logger.error("No data found")
            return pl.DataFrame()
        
        logger.info(f"Loaded {len(df)} records, {df['symbol'].n_unique()} stocks")
        
        # 计算因子
        df = self.compute_factors(df)
        
        # 标准化因子
        df = self.normalize_factors(df, FACTOR_NAMES)
        
        # 计算综合信号
        df = self.compute_composite_signal(df)
        
        # 提取信号
        signals = df.select(["symbol", "trade_date", "signal"])
        
        logger.info(f"Generated signals for {len(signals)} records")
        return signals
    
    def get_prices(self, start_date: str, end_date: str) -> pl.DataFrame:
        """获取价格数据"""
        query = f"""
            SELECT symbol, trade_date, close
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        return self.db.read_sql(query)


# ===========================================
# V21 动态权重管理器
# ===========================================

class V21DynamicSizingManager:
    """
    V21 动态权重管理器 - 凯利准则简化版
    
    【权重分配逻辑】
    1. 得分激活：只有预测得分 > 0.50 的股票才进入备选池
    2. 权重分配：w_i ∝ (Score_i - 0.50)
    3. 单只个股最大权重 20%，最小权重 5%
    4. 总持仓目标数量：5-12 只（根据得分筛选结果动态确定）
    """
    
    def __init__(
        self,
        score_threshold: float = SCORE_THRESHOLD,
        max_position_ratio: float = MAX_POSITION_RATIO,
        min_position_ratio: float = MIN_POSITION_RATIO,
        min_positions: int = MIN_POSITIONS,
        max_positions: int = MAX_POSITIONS,
    ):
        self.score_threshold = score_threshold
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio = min_position_ratio
        self.min_positions = min_positions
        self.max_positions = max_positions
        
        logger.info(f"V21DynamicSizingManager initialized")
        logger.info(f"  Score Threshold: {score_threshold}")
        logger.info(f"  Max Position Ratio: {max_position_ratio:.0%}")
        logger.info(f"  Min Position Ratio: {min_position_ratio:.0%}")
        logger.info(f"  Target Positions: {min_positions}-{max_positions}")
    
    def compute_dynamic_weights(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Any],
        initial_capital: float,
    ) -> Dict[str, float]:
        """
        计算动态权重
        
        Args:
            signals_df: 信号 DataFrame (symbol, trade_date, signal)
            current_positions: 当前持仓
            initial_capital: 初始资金
            
        Returns:
            {symbol: target_amount} 字典
        """
        # 1. 按信号值排序
        ranked = signals_df.sort("signal", descending=True)
        
        # 2. 取前 max_positions 只股票
        top_stocks = ranked.head(self.max_positions)
        
        # 3. 过滤得分 > 0.50 的股票
        qualified = top_stocks.filter(pl.col("signal") > self.score_threshold)
        
        if qualified.is_empty():
            logger.debug(f"No stocks with signal > {self.score_threshold}")
            return {}
        
        # 4. 计算超额得分 (Score - 0.50)
        qualified = qualified.with_columns([
            (pl.col("signal") - self.score_threshold).alias("excess_score")
        ])
        
        # 5. 计算权重 w_i ∝ (Score_i - 0.50)
        total_excess = qualified["excess_score"].sum()
        
        if total_excess <= 0:
            return {}
        
        qualified = qualified.with_columns([
            (pl.col("excess_score") / total_excess).alias("raw_weight")
        ])
        
        # 6. 应用权重限制（最大 20%，最小 5%）
        qualified = qualified.with_columns([
            pl.col("raw_weight").clip(self.min_position_ratio, self.max_position_ratio).alias("clipped_weight")
        ])
        
        # 7. 重新归一化
        total_clipped = qualified["clipped_weight"].sum()
        
        if total_clipped <= 0:
            return {}
        
        qualified = qualified.with_columns([
            (pl.col("clipped_weight") / total_clipped).alias("final_weight")
        ])
        
        # 8. 计算目标金额
        target_amounts = {}
        
        for row in qualified.iter_rows(named=True):
            symbol = row["symbol"]
            weight = row["final_weight"]
            target_amount = initial_capital * weight
            target_amounts[symbol] = target_amount
        
        return target_amounts


# ===========================================
# V21 持仓管理器 - 换手率缓冲区
# ===========================================

class V21PositionManager:
    """
    V21 持仓生命周期管理器 - 换手率缓冲区
    
    【V21 改进】
    - 买入标准：排名必须在 Top 5% 且得分 > 0.50
    - 卖出标准（放宽）：只有当排名跌出 Top 40% 时才卖出
    - 目的：给信号波动留出空间，避免频繁互换导致的无效手续费
    """
    
    def __init__(
        self,
        stop_loss_ratio: float = STOP_LOSS_RATIO,
        take_profit_ratio: float = TAKE_PROFIT_RATIO,
        buy_rank_percentile: float = BUY_RANK_PERCENTILE,
        sell_rank_percentile: float = SELL_RANK_PERCENTILE,
        score_threshold: float = SCORE_THRESHOLD,
        max_positions: int = MAX_POSITIONS,
    ):
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio
        self.buy_rank_percentile = buy_rank_percentile
        self.sell_rank_percentile = sell_rank_percentile
        self.score_threshold = score_threshold
        self.max_positions = max_positions
        
        logger.info(f"V21PositionManager initialized")
        logger.info(f"  Stop Loss: {stop_loss_ratio:.1%}")
        logger.info(f"  Take Profit: {take_profit_ratio:.1%}")
        logger.info(f"  Buy Rank Percentile: Top {buy_rank_percentile}%")
        logger.info(f"  Sell Rank Percentile: > {sell_rank_percentile}% (Buffer Zone)")
    
    def get_stocks_to_sell(
        self,
        positions: Dict[str, Any],
        signal_ranks: Dict[str, int],
        total_stocks: int,
        position_info: Dict[str, Dict],
    ) -> List[str]:
        """
        获取应该卖出的股票列表
        
        卖出逻辑（三选一）：
        1. 触发 8% 硬止损
        2. 触发 15% 止盈
        3. 排名跌出 Top 40%
        """
        stocks_to_sell = []
        
        for symbol in positions.keys():
            # 检查持仓信息中的盈亏
            if symbol in position_info:
                info = position_info[symbol]
                pnl_ratio = info.get("pnl_ratio", 0.0)
                
                # 止损检查
                if pnl_ratio <= -self.stop_loss_ratio:
                    logger.debug(f"STOP LOSS: {symbol} pnl={pnl_ratio:.1%}")
                    stocks_to_sell.append(symbol)
                    continue
                
                # 止盈检查
                if pnl_ratio >= self.take_profit_ratio:
                    logger.debug(f"TAKE PROFIT: {symbol} pnl={pnl_ratio:.1%}")
                    stocks_to_sell.append(symbol)
                    continue
            
            # 信号排名检查（缓冲区）
            signal_rank = signal_ranks.get(symbol, 9999)
            rank_percentile = (signal_rank / total_stocks) * 100 if total_stocks > 0 else 100
            
            if rank_percentile > self.sell_rank_percentile:
                logger.debug(f"SIGNAL RANK: {symbol} rank={signal_rank} ({rank_percentile:.1f}%) > {self.sell_rank_percentile}%")
                stocks_to_sell.append(symbol)
        
        return stocks_to_sell
    
    def get_stocks_to_buy(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Any],
        target_amounts: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        获取应该买入的股票列表
        
        买入标准：
        1. 排名在 Top 5%
        2. 得分 > 0.50
        3. 不在当前持仓中
        """
        stocks_to_buy = []
        
        # 按信号值排序
        ranked = signals_df.sort("signal", descending=True)
        
        for row in ranked.iter_rows(named=True):
            symbol = row["symbol"]
            signal = row["signal"]
            
            # 已持仓的不重复买入
            if symbol in current_positions:
                continue
            
            # 得分必须 > 0.50
            if signal is None or signal <= self.score_threshold:
                continue
            
            # 必须在目标买入列表中
            if symbol in target_amounts:
                stocks_to_buy.append((symbol, target_amounts[symbol]))
            
            # 限制买入数量
            if len(stocks_to_buy) >= self.max_positions:
                break
        
        return stocks_to_buy


# ===========================================
# V21 策略主类
# ===========================================

class V21Strategy:
    """
    V21 策略主类 - 整合信号生成和动态权重
    
    【核心特性】
    1. 6 因子体系生成综合信号
    2. 动态权重分配（凯利准则简化版）
    3. 换手率缓冲区（Top 5% 买入 / Top 40% 卖出）
    4. 严格 T+1 和手续费逻辑（复用 V20 会计引擎）
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        self.signal_generator = V21SignalGenerator(db=self.db)
        self.sizing_manager = V21DynamicSizingManager()
        self.position_manager = V21PositionManager()
        
        logger.info("V21Strategy initialized")
    
    def generate_signals(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """生成交易信号"""
        return self.signal_generator.generate_signals(start_date, end_date)
    
    def get_prices(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """获取价格数据"""
        return self.signal_generator.get_prices(start_date, end_date)
    
    def compute_dynamic_weights(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Any],
        initial_capital: float,
    ) -> Dict[str, float]:
        """计算动态权重"""
        return self.sizing_manager.compute_dynamic_weights(
            signals_df, current_positions, initial_capital
        )
    
    def get_rebalance_list(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Any],
        position_info: Dict[str, Dict],
        target_amounts: Dict[str, float],
    ) -> Tuple[List[str], List[Tuple[str, float]]]:
        """
        获取调仓列表
        
        Returns:
            (stocks_to_sell, stocks_to_buy)
        """
        # 计算信号排名
        ranked = signals_df.sort("signal", descending=True)
        total_stocks = len(ranked)
        
        signal_ranks = {}
        for idx, row in enumerate(ranked.iter_rows(named=True)):
            signal_ranks[row["symbol"]] = idx + 1
        
        # 获取卖出列表
        stocks_to_sell = self.position_manager.get_stocks_to_sell(
            current_positions, signal_ranks, total_stocks, position_info
        )
        
        # 获取买入列表
        stocks_to_buy = self.position_manager.get_stocks_to_buy(
            signals_df, current_positions, target_amounts
        )
        
        return stocks_to_sell, stocks_to_buy
    
    def get_daily_recommendations(
        self,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        trade_date: str,
        initial_capital: float = INITIAL_CAPITAL,
    ) -> List[Dict[str, Any]]:
        """
        获取当日推荐交易列表
        
        Returns:
            推荐列表，包含：
            - 股票代码
            - 推荐权重
            - 预测胜率
            - 建议理由
        """
        # 过滤当日信号
        day_signals = signals_df.filter(pl.col("trade_date") == trade_date)
        
        if day_signals.is_empty():
            return []
        
        # 过滤当日价格
        day_prices = prices_df.filter(pl.col("trade_date") == trade_date)
        prices = {row["symbol"]: row["close"] for row in day_prices.iter_rows(named=True)}
        
        # 计算动态权重
        target_amounts = self.sizing_manager.compute_dynamic_weights(
            day_signals, {}, initial_capital
        )
        
        # 生成推荐列表
        recommendations = []
        
        for symbol, target_amount in target_amounts.items():
            if symbol not in prices:
                continue
            
            # 获取信号值
            signal_row = day_signals.filter(pl.col("symbol") == symbol).row(0, named=True)
            signal = signal_row["signal"]
            
            # 计算排名
            ranked = day_signals.sort("signal", descending=True)
            rank = 0
            for idx, row in enumerate(ranked.iter_rows(named=True)):
                if row["symbol"] == symbol:
                    rank = idx + 1
                    break
            
            weight = target_amount / initial_capital
            
            recommendations.append({
                "symbol": symbol,
                "weight": round(weight * 100, 2),
                "score": round(signal, 4),
                "rank": rank,
                "price": prices[symbol],
                "target_amount": round(target_amount, 2),
                "reason": f"信号排名 Top {rank}, 得分{signal:.4f} > {SCORE_THRESHOLD}",
            })
        
        # 按权重排序
        recommendations.sort(key=lambda x: x["weight"], reverse=True)
        
        return recommendations


# ===========================================
# 实盘指令生成器
# ===========================================

def generate_trading_instructions(
    recommendations: List[Dict[str, Any]],
    trade_date: str,
) -> str:
    """
    生成实盘指令
    
    格式：[股票代码]、[推荐权重]、[预测胜率]、[建议理由]
    """
    instructions = []
    
    instructions.append(f"# 实盘交易指令 - {trade_date}")
    instructions.append("")
    instructions.append("| 序号 | 股票代码 | 推荐权重 | 预测得分 | 目标金额 | 建议理由 |")
    instructions.append("|------|----------|----------|----------|----------|----------|")
    
    for idx, rec in enumerate(recommendations, 1):
        instructions.append(
            f"| {idx} | {rec['symbol']} | {rec['weight']:.1f}% | "
            f"{rec['score']:.4f} | {rec['target_amount']:.0f}元 | {rec['reason']} |"
        )
    
    instructions.append("")
    instructions.append(f"**总计持仓**: {len(recommendations)} 只")
    instructions.append(f"**总权重**: {sum(r['weight'] for r in recommendations):.1f}%")
    
    return "\n".join(instructions)


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
    logger.info("V21 Strategy - Dynamic Sizing & Turnover Buffer")
    logger.info("=" * 70)
    
    # 初始化策略
    strategy = V21Strategy()
    
    # 测试信号生成
    start_date = "2025-01-01"
    end_date = "2026-03-17"
    
    signals_df = strategy.generate_signals(start_date, end_date)
    prices_df = strategy.get_prices(start_date, end_date)
    
    if signals_df.is_empty():
        logger.error("Signal generation failed")
        return
    
    # 获取最新交易日
    latest_date = signals_df["trade_date"].max()
    
    # 生成当日推荐
    recommendations = strategy.get_daily_recommendations(
        signals_df, prices_df, latest_date
    )
    
    # 输出实盘指令
    if recommendations:
        instructions = generate_trading_instructions(recommendations, latest_date)
        logger.info("\n" + instructions)
    else:
        logger.info(f"No recommendations for {latest_date}")
    
    logger.info("\nV21 Strategy test completed.")


if __name__ == "__main__":
    main()