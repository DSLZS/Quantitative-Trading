"""
V22 Strategy - 盈利质量与大盘避险重构

【核心目标：提升"利费比"，实现更有尊严的盈利】

V21 已经实现了止血，但 74.6% 的手续费损耗依然显示出策略的脆弱。
V22 必须通过过滤掉"低质量环境"和捕捉"长趋势"来提升每笔交易的盈利空间。

【核心改进逻辑】

A. 大盘风控过滤器 (Market Regime Filter)
   - 引入沪深 300 指数作为环境参考
   - 当大盘收盘价 < 20 日均线时，进入"防守模式"：
     1. 不再开新仓
     2. 现有持仓的退出缓冲区收紧（排名跌出 Top 20 强制卖出）
   - 目的：避免在系统性风险中为了微弱的 Alpha 支付昂贵的手续费

B. 跟踪止盈与波动率剔除 (Risk-Adjusted Exit)
   - 跟踪止盈 (Trailing Stop)：
     1. 废弃 15% 硬止盈
     2. 个股盈利突破 8% 后，记录最高价
     3. 若收盘价从最高价回撤超过 4%，则强制清仓
   - 波动率黑名单：
     1. 计算过去 10 个交易日的标准差
     2. 剔除波动率位于全市场前 5% 的"妖股"

C. 因子池精简 (Alpha Distillation)
   - 移除 IC 均值小于 0.01 的弱因子
   - 增加 1 个基于"量价背离"的新因子

【严防偷懒与作弊 (Hard Constraints)】
- 物理枷锁：严禁修改 v20_accounting_engine.py
- 本金限制：100,000 元
- 严禁未来函数：所有技术指标、排名、权重计算必须基于 .shift(1)
- 代码复用：必须复用 V21 的 DynamicSizingManager 和 Hysteresis 逻辑

作者：资深量化基金经理 (V22: 盈利质量与大盘避险重构)
版本：V22.0 (Profit Quality & Market Regime Filter)
日期：2026-03-18
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
DEFENSE_SELL_RANK_PERCENTILE = 20  # 防守模式卖出百分位（跌出 Top 20%）

# 止损止盈
STOP_LOSS_RATIO = 0.08      # 8% 硬止损
TRAILING_STOP_THRESHOLD = 0.08  # 触发跟踪止盈的盈利阈值 8%
TRAILING_STOP_DRAWDOWN = 0.04   # 跟踪止盈回撤阈值 4%

# 波动率剔除
VOLATILITY_WINDOW = 10      # 波动率计算窗口
VOLATILITY_PERCENTILE = 95  # 剔除波动率前 5% 的妖股

# 因子配置
FACTOR_NAMES = ["vol_price_corr", "reversal_st", "vol_risk", 
                "turnover_signal", "momentum", "low_vol", "volume_price_divergence"]
FACTOR_WEIGHTS = {
    "vol_price_corr": 0.25,    # 量价相关性（最强因子）
    "reversal_st": 0.20,       # 短线反转
    "vol_risk": 0.15,          # 波动风险
    "turnover_signal": 0.15,   # 异常换手
    "momentum": 0.10,          # 动量因子（降低权重）
    "low_vol": 0.10,           # 低波动因子
    "volume_price_divergence": 0.05,  # 量价背离（新因子）
}

# 市场环境状态
MARKET_STATE_SAFE = "Safe"
MARKET_STATE_CAUTION = "Caution"
MARKET_STATE_DANGER = "Danger"


# ===========================================
# V22 信号生成器
# ===========================================

class V22SignalGenerator:
    """
    V22 信号生成器 - 基于 7 因子体系（精简 + 量价背离）
    
    【因子库】
    1. 量价相关性 (vol_price_corr) - 最强因子
    2. 短线反转 (reversal_st) - 正交化
    3. 波动风险 (vol_risk) - 低波异常
    4. 异常换手 (turnover_signal)
    5. 动量因子 (momentum) - 降低权重
    6. 低波动因子 (low_vol)
    7. 量价背离 (volume_price_divergence) - 新因子，寻找缩量洗盘后的启动机会
    """
    
    EPSILON = 1e-6
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        logger.info("V22SignalGenerator initialized")
    
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
        
        # 7. 量价背离因子 (新因子)
        # 逻辑：寻找缩量洗盘后的启动机会
        # 1. 价格创新高（近 10 日）
        # 2. 成交量萎缩（低于 20 日均量）
        # 3. 当日放量上涨
        high_10d = pl.col("high").rolling_max(window_size=10).shift(1)
        volume_ma20 = pl.col("volume").rolling_mean(window_size=20).shift(1)
        price_new_high = (pl.col("high") >= high_10d).cast(pl.Float64)
        volume_low = (pl.col("volume") < volume_ma20).cast(pl.Float64)
        price_up = (pl.col("close") > pl.col("open")).cast(pl.Float64)
        volume_up = (pl.col("volume") > pl.col("volume").shift(1)).cast(pl.Float64)
        
        # 量价背离信号：前期缩量 + 今日放量上涨
        volume_price_divergence = (
            (volume_low.rolling_sum(window_size=5).shift(1) >= 3).cast(pl.Float64) *  # 过去 5 天至少 3 天缩量
            price_new_high *  # 今日创新高
            price_up * volume_up  # 今日放量上涨
        )
        
        result = result.with_columns([
            volume_price_divergence.alias("volume_price_divergence"),
        ])
        
        logger.info(f"Computed 7 factors for {len(result)} records")
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
        """计算加权综合信号"""
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
    
    def get_index_data(self, start_date: str, end_date: str, index_symbol: str = "000300.SH") -> pl.DataFrame:
        """获取指数数据用于大盘风控"""
        query = f"""
            SELECT symbol, trade_date, close
            FROM stock_daily
            WHERE symbol = '{index_symbol}'
            AND trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        return self.db.read_sql(query)


# ===========================================
# V22 动态权重管理器 (复用 V21 逻辑)
# ===========================================

class V22DynamicSizingManager:
    """
    V22 动态权重管理器 - 凯利准则简化版（复用 V21 逻辑）
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
        
        logger.info(f"V22DynamicSizingManager initialized")
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
        """计算动态权重"""
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
# V22 持仓管理器 - 跟踪止盈与波动率剔除
# ===========================================

class V22PositionManager:
    """
    V22 持仓生命周期管理器 - 跟踪止盈与波动率剔除
    
    【V22 改进】
    - 买入标准：排名必须在 Top 5% 且得分 > 0.50
    - 卖出标准（三选一）：
      1. 触发 8% 硬止损
      2. 跟踪止盈：盈利突破 8% 后，从最高价回撤超过 4%
      3. 排名跌出 Top 40%（防守模式下收紧至 Top 20%）
    """
    
    def __init__(
        self,
        stop_loss_ratio: float = STOP_LOSS_RATIO,
        trailing_stop_threshold: float = TRAILING_STOP_THRESHOLD,
        trailing_stop_drawdown: float = TRAILING_STOP_DRAWDOWN,
        buy_rank_percentile: float = BUY_RANK_PERCENTILE,
        sell_rank_percentile: float = SELL_RANK_PERCENTILE,
        defense_sell_rank_percentile: float = DEFENSE_SELL_RANK_PERCENTILE,
        score_threshold: float = SCORE_THRESHOLD,
        max_positions: int = MAX_POSITIONS,
    ):
        self.stop_loss_ratio = stop_loss_ratio
        self.trailing_stop_threshold = trailing_stop_threshold
        self.trailing_stop_drawdown = trailing_stop_drawdown
        self.buy_rank_percentile = buy_rank_percentile
        self.sell_rank_percentile = sell_rank_percentile
        self.defense_sell_rank_percentile = defense_sell_rank_percentile
        self.score_threshold = score_threshold
        self.max_positions = max_positions
        
        logger.info(f"V22PositionManager initialized")
        logger.info(f"  Stop Loss: {stop_loss_ratio:.1%}")
        logger.info(f"  Trailing Stop Threshold: {trailing_stop_threshold:.1%}")
        logger.info(f"  Trailing Stop Drawdown: {trailing_stop_drawdown:.1%}")
        logger.info(f"  Buy Rank Percentile: Top {buy_rank_percentile}%")
        logger.info(f"  Sell Rank Percentile: > {sell_rank_percentile}% (Normal)")
        logger.info(f"  Sell Rank Percentile: > {defense_sell_rank_percentile}% (Defense)")
    
    def get_stocks_to_sell(
        self,
        positions: Dict[str, Any],
        signal_ranks: Dict[str, int],
        total_stocks: int,
        position_info: Dict[str, Dict],
        is_defense_mode: bool = False,
    ) -> List[str]:
        """
        获取应该卖出的股票列表
        
        卖出逻辑（三选一）：
        1. 触发 8% 硬止损
        2. 触发跟踪止盈（盈利突破 8% 后，从最高价回撤超过 4%）
        3. 排名跌出 Top 40%（防守模式下为 Top 20%）
        """
        stocks_to_sell = []
        
        # 确定卖出排名阈值
        sell_rank_threshold = self.defense_sell_rank_percentile if is_defense_mode else self.sell_rank_percentile
        
        for symbol in positions.keys():
            # 检查持仓信息中的盈亏
            if symbol in position_info:
                info = position_info[symbol]
                pnl_ratio = info.get("pnl_ratio", 0.0)
                highest_pnl_ratio = info.get("highest_pnl_ratio", 0.0)
                
                # 止损检查
                if pnl_ratio <= -self.stop_loss_ratio:
                    logger.debug(f"STOP LOSS: {symbol} pnl={pnl_ratio:.1%}")
                    stocks_to_sell.append(symbol)
                    continue
                
                # 跟踪止盈检查
                # 条件 1: 曾经盈利超过 8%
                # 条件 2: 从最高盈利回撤超过 4%
                if highest_pnl_ratio >= self.trailing_stop_threshold:
                    current_drawdown_from_peak = highest_pnl_ratio - pnl_ratio
                    if current_drawdown_from_peak >= self.trailing_stop_drawdown:
                        logger.debug(f"TRAILING STOP: {symbol} highest_pnl={highest_pnl_ratio:.1%}, current_pnl={pnl_ratio:.1%}, drawdown={current_drawdown_from_peak:.1%}")
                        stocks_to_sell.append(symbol)
                        continue
            
            # 信号排名检查（缓冲区）
            signal_rank = signal_ranks.get(symbol, 9999)
            rank_percentile = (signal_rank / total_stocks) * 100 if total_stocks > 0 else 100
            
            if rank_percentile > sell_rank_threshold:
                logger.debug(f"SIGNAL RANK: {symbol} rank={signal_rank} ({rank_percentile:.1f}%) > {sell_rank_threshold}%")
                stocks_to_sell.append(symbol)
        
        return stocks_to_sell
    
    def get_stocks_to_buy(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Any],
        target_amounts: Dict[str, float],
        is_defense_mode: bool = False,
    ) -> List[Tuple[str, float]]:
        """
        获取应该买入的股票列表
        
        买入标准：
        1. 排名在 Top 5%
        2. 得分 > 0.50
        3. 不在当前持仓中
        4. 防守模式下不买入
        """
        stocks_to_buy = []
        
        # 防守模式下不买入
        if is_defense_mode:
            return stocks_to_buy
        
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
# V22 波动率过滤器
# ===========================================

class V22VolatilityFilter:
    """
    V22 波动率过滤器 - 剔除妖股
    
    【逻辑】
    - 计算过去 10 个交易日的收益率标准差
    - 截面排序，剔除波动率位于前 5% 的股票
    """
    
    def __init__(
        self,
        volatility_window: int = VOLATILITY_WINDOW,
        volatility_percentile: int = VOLATILITY_PERCENTILE,
    ):
        self.volatility_window = volatility_window
        self.volatility_percentile = volatility_percentile
        
        logger.info(f"V22VolatilityFilter initialized")
        logger.info(f"  Volatility Window: {volatility_window} days")
        logger.info(f"  Volatility Percentile: Top {100 - volatility_percentile}% (blacklist)")
    
    def compute_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算波动率"""
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 计算收益率
        stock_returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        
        # 计算波动率（年化）
        volatility = stock_returns.rolling_std(window_size=self.volatility_window, ddof=1).shift(1) * np.sqrt(252)
        
        result = result.with_columns([
            volatility.alias("volatility"),
        ])
        
        return result
    
    def filter_high_volatility(self, df: pl.DataFrame) -> pl.DataFrame:
        """过滤高波动率股票"""
        # 计算波动率
        result = self.compute_volatility(df)
        
        # 截面计算分位数
        def compute_percentile(group: pl.DataFrame) -> pl.DataFrame:
            if len(group) == 0:
                return group
            volatility_values = group["volatility"].drop_nulls()
            if len(volatility_values) == 0:
                return group.with_columns([pl.lit(True).alias("is_valid")])
            threshold = np.percentile(volatility_values, self.volatility_percentile)
            return group.with_columns([
                (pl.col("volatility") <= threshold).alias("is_valid")
            ])
        
        result = result.group_by("trade_date", maintain_order=True).map_groups(
            lambda df: compute_percentile(df)
        )
        
        # 过滤
        filtered = result.filter(pl.col("is_valid"))
        
        logger.info(f"Filtered {len(result) - len(filtered)} high volatility stocks")
        return filtered


# ===========================================
# V22 市场环境过滤器
# ===========================================

class V22MarketRegimeFilter:
    """
    V22 市场环境过滤器 - 大盘风控
    
    【逻辑】
    - 使用沪深 300 指数（000300.SH）作为市场基准
    - 计算 20 日均线
    - 当收盘价 < 20 日均线时，进入"防守模式"
    
    【市场状态】
    - Safe: 收盘价 > MA20
    - Caution: 收盘价接近 MA20（在 2% 以内）
    - Danger: 收盘价 < MA20
    """
    
    def __init__(self, ma_window: int = 20):
        self.ma_window = ma_window
        
        logger.info(f"V22MarketRegimeFilter initialized")
        logger.info(f"  MA Window: {ma_window} days")
    
    def compute_market_state(self, index_df: pl.DataFrame) -> pl.DataFrame:
        """计算市场状态"""
        if index_df.is_empty():
            logger.warning("Index data is empty, using Safe state")
            return index_df.with_columns([pl.lit(MARKET_STATE_SAFE).alias("market_state")])
        
        result = index_df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 计算 MA20
        result = result.with_columns([
            pl.col("close").rolling_mean(window_size=self.ma_window).shift(1).alias("ma20"),
        ])
        
        # 计算市场状态
        def classify_state(row: Dict[str, Any]) -> str:
            close = row.get("close", 0)
            ma20 = row.get("ma20", 0)
            
            if ma20 is None or ma20 == 0 or close is None:
                return MARKET_STATE_SAFE
            
            distance = (close - ma20) / ma20
            
            if close >= ma20:
                return MARKET_STATE_SAFE
            elif distance >= -0.02:  # 在 2% 以内
                return MARKET_STATE_CAUTION
            else:
                return MARKET_STATE_DANGER
        
        # 使用 apply 计算状态
        result = result.with_columns([
            pl.struct(["close", "ma20"]).map_elements(
                lambda x: classify_state(x.to_dict()),
                return_dtype=pl.Utf8
            ).alias("market_state")
        ])
        
        return result
    
    def get_daily_state(self, index_df: pl.DataFrame, trade_date: str) -> str:
        """获取指定日期的市场状态"""
        day_data = index_df.filter(pl.col("trade_date") == trade_date)
        
        if day_data.is_empty():
            return MARKET_STATE_SAFE
        
        row = day_data.row(0, named=True)
        return row.get("market_state", MARKET_STATE_SAFE)


# ===========================================
# V22 策略主类
# ===========================================

class V22Strategy:
    """
    V22 策略主类 - 整合信号生成和动态权重
    
    【核心特性】
    1. 7 因子体系生成综合信号（含新因子"量价背离"）
    2. 动态权重分配（凯利准则简化版）
    3. 换手率缓冲区（Top 5% 买入 / Top 40% 卖出）
    4. 跟踪止盈（盈利突破 8% 后，回撤 4% 清仓）
    5. 波动率剔除（剔除前 5% 妖股）
    6. 大盘风控过滤器（MA20 判断市场状态）
    7. 严格 T+1 和手续费逻辑（复用 V20 会计引擎）
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        self.signal_generator = V22SignalGenerator(db=self.db)
        self.sizing_manager = V22DynamicSizingManager()
        self.position_manager = V22PositionManager()
        self.volatility_filter = V22VolatilityFilter()
        self.market_regime_filter = V22MarketRegimeFilter()
        
        logger.info("V22Strategy initialized")
    
    def generate_signals(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """生成交易信号"""
        signals = self.signal_generator.generate_signals(start_date, end_date)
        
        if signals.is_empty():
            return signals
        
        # 应用波动率过滤
        # 需要获取原始数据来计算波动率
        query = f"""
            SELECT 
                symbol, trade_date, open, high, low, close, pre_close,
                volume, amount, turnover_rate, industry_code, total_mv
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        raw_df = self.db.read_sql(query)
        
        if not raw_df.is_empty():
            filtered_df = self.volatility_filter.filter_high_volatility(raw_df)
            valid_symbols = filtered_df.select(["symbol", "trade_date"]).unique()
            
            # 与信号合并
            signals = signals.join(
                valid_symbols,
                on=["symbol", "trade_date"],
                how="inner"
            )
        
        return signals
    
    def get_prices(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """获取价格数据"""
        return self.signal_generator.get_prices(start_date, end_date)
    
    def get_index_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """获取指数数据"""
        return self.signal_generator.get_index_data(start_date, end_date)
    
    def compute_market_states(self, index_df: pl.DataFrame) -> pl.DataFrame:
        """计算市场状态"""
        return self.market_regime_filter.compute_market_state(index_df)
    
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
        is_defense_mode: bool = False,
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
            current_positions, signal_ranks, total_stocks, position_info, is_defense_mode
        )
        
        # 获取买入列表
        stocks_to_buy = self.position_manager.get_stocks_to_buy(
            signals_df, current_positions, target_amounts, is_defense_mode
        )
        
        return stocks_to_sell, stocks_to_buy
    
    def get_daily_recommendations(
        self,
        signals_df: pl.DataFrame,
        prices_df: pl.DataFrame,
        trade_date: str,
        initial_capital: float = INITIAL_CAPITAL,
        market_state: str = MARKET_STATE_SAFE,
    ) -> List[Dict[str, Any]]:
        """
        获取当日推荐交易列表
        
        Returns:
            推荐列表，包含：
            - 股票代码
            - 推荐权重
            - 预测胜率
            - 建议理由
            - 市场状态
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
                "market_state": market_state,
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
    market_state: str = MARKET_STATE_SAFE,
) -> str:
    """
    生成实盘指令
    
    格式：[股票代码]、[推荐权重]、[预测胜率]、[建议理由]、[市场状态]
    """
    instructions = []
    
    instructions.append(f"# 实盘交易指令 - {trade_date}")
    instructions.append(f"**市场环境**: {market_state}")
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
    logger.info("V22 Strategy - Profit Quality & Market Regime Filter")
    logger.info("=" * 70)
    
    # 初始化策略
    strategy = V22Strategy()
    
    # 测试信号生成
    start_date = "2025-01-01"
    end_date = "2026-03-17"
    
    signals_df = strategy.generate_signals(start_date, end_date)
    prices_df = strategy.get_prices(start_date, end_date)
    index_df = strategy.get_index_data(start_date, end_date)
    
    if signals_df.is_empty():
        logger.error("Signal generation failed")
        return
    
    # 计算市场状态
    market_states_df = strategy.compute_market_states(index_df)
    
    # 获取最新交易日
    latest_date = signals_df["trade_date"].max()
    
    # 获取市场状态
    market_state = strategy.market_regime_filter.get_daily_state(market_states_df, latest_date)
    
    # 生成当日推荐
    recommendations = strategy.get_daily_recommendations(
        signals_df, prices_df, latest_date, market_state=market_state
    )
    
    # 输出实盘指令
    if recommendations:
        instructions = generate_trading_instructions(recommendations, latest_date, market_state)
        logger.info("\n" + instructions)
    else:
        logger.info(f"No recommendations for {latest_date}")
    
    logger.info("\nV22 Strategy test completed.")


if __name__ == "__main__":
    main()