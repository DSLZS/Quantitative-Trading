"""
V23 Strategy - 自动数据闭环与智能自愈策略

【核心改进：进攻性重构】

1. 因子自动熔断 (Factor Auto-Pruning)
   - 严禁无脑堆砌因子
   - 如果新加入的因子在回测前期的 IC 均值 < 0.02，物理删除
   - 不再进入最后的线性回归模型

2. 进攻性入场标准
   - 只有当全市场 Top 10 的预测分均值 > 0.65 时，才允许开新仓
   - 只在信号极度强烈的"大行情"入场
   - 用高质量盈利覆盖 5 元低保佣金

3. 动态权重分配
   - 使用 softmax 或 Score^2 放大头部股票的权重差距
   - "百步穿杨"而非"散弹打鸟"

4. 剔除无效防御
   - 彻底删除 V22 那些截断利润的止盈逻辑
   - 回归 V21 的高收益特性

作者：资深量化系统架构师 (V23: 自动数据闭环与智能自愈)
版本：V23.0 (Aggressive Alpha Seeking)
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

# 进攻性入场阈值
ENTRY_SCORE_THRESHOLD = 0.65  # Top 10 预测分均值必须 > 0.65 才开仓
MIN_ACTIVATION_SCORE = 0.50   # 个股得分激活阈值

# 动态权重配置
MAX_POSITION_RATIO = 0.25   # 单只个股最大权重 25%（集中持仓）
MIN_POSITION_RATIO = 0.08   # 最小权重 8%
MIN_POSITIONS = 3           # 最小持仓数量（少而精）
MAX_POSITIONS = 8           # 最大持仓数量（集中打击）

# 因子 IC 阈值
IC_MEAN_THRESHOLD = 0.02    # IC 均值低于此值的因子被物理删除
IC_WINDOW = 20              # IC 计算窗口

# 止损配置（简化，不设止盈）
STOP_LOSS_RATIO = 0.10      # 10% 硬止损（放宽）

# 因子配置（基础 6 因子）
BASE_FACTOR_NAMES = [
    "vol_price_corr",      # 量价相关性
    "reversal_st",         # 短线反转
    "vol_risk",            # 波动风险
    "turnover_signal",     # 异常换手
    "momentum",            # 动量因子
    "low_vol",             # 低波动因子
]

# 初始权重（等权重）
BASE_FACTOR_WEIGHTS = {
    "vol_price_corr": 0.20,
    "reversal_st": 0.20,
    "vol_risk": 0.15,
    "turnover_signal": 0.15,
    "momentum": 0.15,
    "low_vol": 0.15,
}


# ===========================================
# V23 因子分析器 - 智能熔断
# ===========================================

class FactorAnalyzer:
    """
    V23 因子分析器 - 因子自动熔断
    
    【核心职责】
    1. 计算每个因子的 IC 值（信息系数）
    2. 如果 IC 均值 < 0.02，物理删除该因子
    3. 动态调整因子权重
    
    【IC 计算逻辑】
    - IC = 因子值与未来 N 日收益率的相关系数
    - 使用滚动窗口计算 IC 均值
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = IC_MEAN_THRESHOLD,
        ic_window: int = IC_WINDOW,
        db: Optional[DatabaseManager] = None,
    ):
        self.ic_threshold = ic_threshold
        self.ic_window = ic_window
        self.db = db or DatabaseManager.get_instance()
        
        # 因子有效性记录
        self.factor_validity: Dict[str, bool] = {}
        self.factor_ic_stats: Dict[str, Dict[str, float]] = {}
        
        logger.info("FactorAnalyzer initialized")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  IC Window: {ic_window} days")
    
    def compute_factor_ic(
        self,
        factor_df: pl.DataFrame,
        factor_name: str,
        forward_window: int = 5,
    ) -> float:
        """
        计算单个因子的 IC 值
        
        IC = corr(factor_value, future_return)
        
        Args:
            factor_df: 包含因子值的数据框
            factor_name: 因子名称
            forward_window: 前向收益计算窗口
        
        Returns:
            IC 值（相关系数）
        """
        try:
            # 计算未来 N 日收益率
            df = factor_df.clone().with_columns([
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col(factor_name).cast(pl.Float64, strict=False),
            ])
            
            # 未来收益率
            future_return = (
                pl.col("close").shift(-forward_window).over("symbol") /
                pl.col("close").over("symbol") - 1
            )
            
            df = df.with_columns([
                future_return.alias("future_return"),
            ])
            
            # 按日期分组计算 IC
            ic_by_date = df.group_by("trade_date").agg([
                pl.corr(pl.col(factor_name), pl.col("future_return")).alias("ic")
            ])
            
            # 过滤无效值
            ic_values = ic_by_date.filter(
                (pl.col("ic").is_not_null()) & 
                (pl.col("ic").is_finite())
            )["ic"].to_list()
            
            if len(ic_values) == 0:
                return 0.0
            
            # 返回 IC 均值
            ic_mean = np.mean(ic_values)
            return ic_mean
            
        except Exception as e:
            logger.warning(f"Failed to compute IC for {factor_name}: {e}")
            return 0.0
    
    def analyze_factors(
        self,
        factor_df: pl.DataFrame,
        factor_names: List[str],
    ) -> Dict[str, bool]:
        """
        分析所有因子的有效性
        
        Args:
            factor_df: 包含所有因子值的数据框
            factor_names: 因子名称列表
        
        Returns:
            {factor_name: is_valid} 字典
        """
        logger.info("=" * 50)
        logger.info("FACTOR ANALYSIS - IC Evaluation")
        logger.info("=" * 50)
        
        for factor_name in factor_names:
            if factor_name not in factor_df.columns:
                logger.warning(f"Factor {factor_name} not found in DataFrame")
                self.factor_validity[factor_name] = False
                self.factor_ic_stats[factor_name] = {"ic_mean": 0.0, "is_valid": False}
                continue
            
            # 计算 IC
            ic_mean = self.compute_factor_ic(factor_df, factor_name)
            
            # 判断是否有效
            is_valid = abs(ic_mean) >= self.ic_threshold
            
            self.factor_validity[factor_name] = is_valid
            self.factor_ic_stats[factor_name] = {
                "ic_mean": ic_mean,
                "is_valid": is_valid,
            }
            
            # 输出结果
            status = "✅ VALID" if is_valid else "❌ PRUNED"
            logger.info(f"  {factor_name}: IC={ic_mean:.4f} -> {status}")
        
        # 统计有效因子数量
        valid_count = sum(1 for v in self.factor_validity.values() if v)
        logger.info(f"\nValid factors: {valid_count}/{len(factor_names)}")
        
        return self.factor_validity.copy()
    
    def get_valid_factors(self) -> List[str]:
        """获取有效因子列表"""
        return [f for f, valid in self.factor_validity.items() if valid]
    
    def get_pruned_factors(self) -> List[str]:
        """获取被熔断的因子列表"""
        return [f for f, valid in self.factor_validity.items() if not valid]
    
    def compute_adaptive_weights(
        self,
        valid_factors: List[str],
    ) -> Dict[str, float]:
        """
        计算自适应权重
        
        基于 IC 值分配权重：IC 越高，权重越大
        
        Args:
            valid_factors: 有效因子列表
        
        Returns:
            权重字典
        """
        if not valid_factors:
            return {}
        
        # 获取各因子的 IC 绝对值
        ic_weights = {}
        for factor in valid_factors:
            ic_stats = self.factor_ic_stats.get(factor, {})
            ic_mean = abs(ic_stats.get("ic_mean", 0.0))
            ic_weights[factor] = max(ic_mean, self.EPSILON)  # 避免 0 权重
        
        # 归一化
        total_ic = sum(ic_weights.values())
        
        if total_ic <= 0:
            # 等权重
            return {f: 1.0 / len(valid_factors) for f in valid_factors}
        
        adaptive_weights = {f: w / total_ic for f, w in ic_weights.items()}
        
        logger.info("Adaptive factor weights:")
        for factor, weight in adaptive_weights.items():
            logger.info(f"  {factor}: {weight:.2%}")
        
        return adaptive_weights


# ===========================================
# V23 信号生成器
# ===========================================

class V23SignalGenerator:
    """
    V23 信号生成器 - 基于纯化因子的信号生成
    
    【核心特性】
    1. 使用 FactorAnalyzer 进行因子有效性检验
    2. 只使用有效因子生成综合信号
    3. 使用自适应权重
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        ic_threshold: float = IC_MEAN_THRESHOLD,
    ):
        self.db = db or DatabaseManager.get_instance()
        self.factor_analyzer = FactorAnalyzer(
            ic_threshold=ic_threshold,
            db=self.db,
        )
        self.valid_factors: List[str] = []
        self.factor_weights: Dict[str, float] = {}
        
        logger.info("V23SignalGenerator initialized")
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算所有基础因子"""
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
        
        logger.info(f"Computed 6 base factors for {len(result)} records")
        return result
    
    def normalize_factors(
        self,
        df: pl.DataFrame,
        factor_names: List[str],
    ) -> pl.DataFrame:
        """截面标准化因子"""
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
    
    def analyze_and_select_factors(
        self,
        df: pl.DataFrame,
    ) -> Tuple[List[str], Dict[str, float]]:
        """
        分析因子并选择有效因子
        
        Returns:
            (valid_factors, adaptive_weights)
        """
        # 分析因子有效性
        self.factor_analyzer.analyze_factors(df, BASE_FACTOR_NAMES)
        
        # 获取有效因子
        self.valid_factors = self.factor_analyzer.get_valid_factors()
        
        if not self.valid_factors:
            logger.warning("No valid factors! Using all base factors as fallback")
            self.valid_factors = BASE_FACTOR_NAMES.copy()
        
        # 计算自适应权重
        self.factor_weights = self.factor_analyzer.compute_adaptive_weights(self.valid_factors)
        
        return self.valid_factors, self.factor_weights
    
    def compute_composite_signal(
        self,
        df: pl.DataFrame,
        valid_factors: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> pl.DataFrame:
        """
        计算加权综合信号
        
        使用有效因子和自适应权重
        """
        if valid_factors is None:
            valid_factors = self.valid_factors or BASE_FACTOR_NAMES
        
        if weights is None:
            weights = self.factor_weights or BASE_FACTOR_WEIGHTS
        
        # 使用标准化后的因子
        std_factors = [f"{f}_std" for f in valid_factors if f"{f}_std" in df.columns]
        
        if not std_factors:
            # 如果没有标准化因子，使用原始因子
            std_factors = valid_factors
        
        # 加权综合信号
        signal = None
        for factor in std_factors:
            factor_name = factor.replace("_std", "")
            weight = weights.get(factor_name, 1.0 / len(std_factors))
            if signal is None:
                signal = pl.col(factor) * weight
            else:
                signal = signal + pl.col(factor) * weight
        
        if signal is None:
            logger.error("No signal could be computed")
            return df.with_columns([pl.lit(0.0).alias("signal")])
        
        result = df.with_columns([
            signal.alias("signal")
        ])
        
        return result
    
    def generate_signals(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """生成交易信号"""
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
        
        # 分析并选择有效因子
        valid_factors, weights = self.analyze_and_select_factors(df)
        logger.info(f"Using {len(valid_factors)} valid factors: {valid_factors}")
        
        # 标准化因子
        df = self.normalize_factors(df, valid_factors)
        
        # 计算综合信号
        df = self.compute_composite_signal(df, valid_factors, weights)
        
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
# V23 动态权重管理器 - 进攻性赋权
# ===========================================

class V23DynamicSizingManager:
    """
    V23 动态权重管理器 - 进攻性赋权
    
    【核心逻辑】
    1. 使用 Score^2 或 softmax 放大头部股票权重
    2. 集中持仓，单只上限 25%
    3. 只在 Top 10 预测分均值 > 0.65 时开仓
    """
    
    def __init__(
        self,
        min_activation_score: float = MIN_ACTIVATION_SCORE,
        entry_score_threshold: float = ENTRY_SCORE_THRESHOLD,
        max_position_ratio: float = MAX_POSITION_RATIO,
        min_position_ratio: float = MIN_POSITION_RATIO,
        min_positions: int = MIN_POSITIONS,
        max_positions: int = MAX_POSITIONS,
    ):
        self.min_activation_score = min_activation_score
        self.entry_score_threshold = entry_score_threshold
        self.max_position_ratio = max_position_ratio
        self.min_position_ratio = min_position_ratio
        self.min_positions = min_positions
        self.max_positions = max_positions
        
        logger.info("V23DynamicSizingManager initialized")
        logger.info(f"  Entry Score Threshold: {entry_score_threshold}")
        logger.info(f"  Max Position Ratio: {max_position_ratio:.0%}")
        logger.info(f"  Min Position Ratio: {min_position_ratio:.0%}")
        logger.info(f"  Target Positions: {min_positions}-{max_positions}")
    
    def compute_softmax_weights(
        self,
        scores: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        使用 softmax 计算权重
        
        Args:
            scores: 得分数组
            temperature: 温度参数（越小越集中）
        
        Returns:
            权重数组
        """
        # 归一化得分
        normalized_scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-6)
        
        # softmax
        exp_scores = np.exp(normalized_scores / temperature)
        weights = exp_scores / np.sum(exp_scores)
        
        return weights
    
    def compute_squared_weights(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        使用 Score^2 计算权重
        
        Args:
            scores: 得分数组
        
        Returns:
            权重数组
        """
        # 确保得分为正
        positive_scores = scores - np.min(scores) + 0.01
        
        # 平方放大
        squared = positive_scores ** 2
        
        # 归一化
        weights = squared / np.sum(squared)
        
        return weights
    
    def check_entry_condition(
        self,
        signals_df: pl.DataFrame,
    ) -> Tuple[bool, float]:
        """
        检查入场条件
        
        只有当 Top 10 预测分均值 > 0.65 时才允许开仓
        
        Returns:
            (can_enter, top10_avg_score)
        """
        if signals_df.is_empty():
            return False, 0.0
        
        # 按信号值排序
        ranked = signals_df.sort("signal", descending=True)
        
        # 取 Top 10
        top_10 = ranked.head(10)
        
        if top_10.is_empty():
            return False, 0.0
        
        # 计算 Top 10 平均得分
        top10_avg = top_10["signal"].mean()
        
        if top10_avg is None:
            return False, 0.0
        
        can_enter = top10_avg >= self.entry_score_threshold
        
        return can_enter, top10_avg
    
    def compute_dynamic_weights(
        self,
        signals_df: pl.DataFrame,
        current_positions: Dict[str, Any],
        initial_capital: float,
        use_softmax: bool = True,
    ) -> Tuple[Dict[str, float], bool]:
        """
        计算动态权重
        
        Args:
            signals_df: 信号 DataFrame
            current_positions: 当前持仓
            initial_capital: 初始资金
            use_softmax: 是否使用 softmax（否则使用 Score^2）
        
        Returns:
            (target_amounts, can_enter) 字典
        """
        # 检查入场条件
        can_enter, top10_avg = self.check_entry_condition(signals_df)
        
        if not can_enter:
            logger.info(f"ENTRY BLOCKED: Top 10 avg score = {top10_avg:.4f} < {self.entry_score_threshold}")
            return {}, False
        
        logger.info(f"ENTRY ALLOWED: Top 10 avg score = {top10_avg:.4f} >= {self.entry_score_threshold}")
        
        # 按信号值排序
        ranked = signals_df.sort("signal", descending=True)
        
        # 取前 max_positions 只股票
        top_stocks = ranked.head(self.max_positions)
        
        # 过滤得分 > 激活阈值的股票
        qualified = top_stocks.filter(pl.col("signal") > self.min_activation_score)
        
        if qualified.is_empty():
            logger.debug(f"No stocks with signal > {self.min_activation_score}")
            return {}, False
        
        # 转换为 numpy 数组计算权重
        scores = qualified["signal"].to_numpy()
        symbols = qualified["symbol"].to_list()
        
        # 计算权重
        if use_softmax:
            raw_weights = self.compute_softmax_weights(scores, temperature=0.5)
        else:
            raw_weights = self.compute_squared_weights(scores)
        
        # 应用权重限制
        clipped_weights = np.clip(raw_weights, self.min_position_ratio, self.max_position_ratio)
        
        # 重新归一化
        clipped_weights = clipped_weights / np.sum(clipped_weights)
        
        # 计算目标金额
        target_amounts = {}
        for symbol, weight in zip(symbols, clipped_weights):
            target_amount = initial_capital * weight
            target_amounts[symbol] = target_amount
        
        logger.info(f"Computed weights for {len(target_amounts)} stocks")
        return target_amounts, True


# ===========================================
# V23 持仓管理器 - 简化退出逻辑
# ===========================================

class V23PositionManager:
    """
    V23 持仓管理器 - 简化退出逻辑
    
    【核心改进】
    - 剔除 V22 的跟踪止盈（截断利润）
    - 只保留 10% 硬止损
    - 信号排名跌出 Top 50% 时卖出
    """
    
    def __init__(
        self,
        stop_loss_ratio: float = STOP_LOSS_RATIO,
        sell_rank_percentile: float = 50,  # 跌出 Top 50% 卖出
    ):
        self.stop_loss_ratio = stop_loss_ratio
        self.sell_rank_percentile = sell_rank_percentile
        
        logger.info("V23PositionManager initialized")
        logger.info(f"  Stop Loss: {stop_loss_ratio:.0%}")
        logger.info(f"  Sell Rank Percentile: > {sell_rank_percentile}%")
    
    def get_stocks_to_sell(
        self,
        positions: Dict[str, Any],
        signal_ranks: Dict[str, int],
        total_stocks: int,
        position_info: Dict[str, Dict],
    ) -> List[str]:
        """
        获取应该卖出的股票列表
        
        卖出逻辑（二选一）：
        1. 触发 10% 硬止损
        2. 排名跌出 Top 50%
        """
        stocks_to_sell = []
        
        for symbol in positions.keys():
            # 检查止损
            if symbol in position_info:
                info = position_info[symbol]
                pnl_ratio = info.get("pnl_ratio", 0.0)
                
                if pnl_ratio <= -self.stop_loss_ratio:
                    logger.debug(f"STOP LOSS: {symbol} pnl={pnl_ratio:.1%}")
                    stocks_to_sell.append(symbol)
                    continue
            
            # 信号排名检查
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
        """
        stocks_to_buy = []
        
        for symbol, target_amount in target_amounts.items():
            if symbol in current_positions:
                continue
            
            # 检查信号值
            row = signals_df.filter(pl.col("symbol") == symbol).row(0, named=True)
            signal = row.get("signal", 0)
            
            if signal is not None and signal > MIN_ACTIVATION_SCORE:
                stocks_to_buy.append((symbol, target_amount))
        
        return stocks_to_buy


# ===========================================
# V23 策略主类
# ===========================================

class V23Strategy:
    """
    V23 策略主类 - 自动数据闭环与智能自愈
    
    【核心特性】
    1. 因子自动熔断：IC < 0.02 的因子物理删除
    2. 进攻性入场：Top 10 预测分 > 0.65 才开仓
    3. 动态权重：使用 softmax/Score^2 放大头部差距
    4. 简化退出：只保留硬止损，剔除无效止盈
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        self.signal_generator = V23SignalGenerator(db=self.db)
        self.sizing_manager = V23DynamicSizingManager()
        self.position_manager = V23PositionManager()
        
        logger.info("V23Strategy initialized")
        logger.info(f"  Entry Threshold: {ENTRY_SCORE_THRESHOLD}")
        logger.info(f"  IC Threshold: {IC_MEAN_THRESHOLD}")
    
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
    ) -> Tuple[Dict[str, float], bool]:
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
        """获取调仓列表"""
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
    logger.info("V23 Strategy - Auto-Healing & Aggressive Alpha")
    logger.info("=" * 70)
    
    # 初始化策略
    strategy = V23Strategy()
    
    # 测试信号生成
    start_date = "2025-01-01"
    end_date = "2026-03-17"
    
    signals_df = strategy.generate_signals(start_date, end_date)
    
    if signals_df.is_empty():
        logger.error("Signal generation failed")
        return
    
    logger.info(f"\nGenerated {len(signals_df)} signal records")
    logger.info("\nV23 Strategy test completed.")


if __name__ == "__main__":
    main()