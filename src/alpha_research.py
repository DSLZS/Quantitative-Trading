"""
Alpha Research Module - V101 Prediction Core.

唯一存放预测算法的地方。
核心功能:
    - 因子计算引擎
    - 量价非线性交互 (Non-linear Vol-Price)
    - T+1 收益截面排名预测
    - Spearman Rank IC 损失函数
    - 因子贡献度分析
    - 自检机制 (IC < 0.03 触发 AlphaWeakWarning)
"""

import math
from typing import Any, Optional
from pathlib import Path

import polars as pl
import numpy as np
from loguru import logger
import yaml

# 内存优化
pl.Config.set_streaming_chunk_size(10000)


class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告 - 当 T+1 IC 低于 0.03 时触发"""
    pass


class AlphaResearch:
    """
    V101 Alpha 预测核心引擎。
    
    【核心逻辑】
    - 预测目标：T+1 收益的截面排名
    - 损失函数：Spearman Rank IC
    - 优化重点：量价非线性交互
    
    【因子体系】
    1. 基础动量因子 (momentum_5, momentum_10, momentum_20)
    2. 波动率因子 (volatility_5, volatility_20)
    3. 量价交互因子 (volume_price_divergence, volume_entropy)
    4. 非线性因子 (volume_price_health, vcp_score)
    
    【自检机制】
    - T+1 IC < 0.03: 触发 AlphaWeakWarning
    - 输出各因子贡献度分析
    """
    
    EPSILON = 1e-6
    
    # 因子权重配置
    FACTOR_WEIGHTS = {
        # 基础动量
        "momentum_5": 0.15,
        "momentum_10": 0.10,
        "momentum_20": 0.05,
        
        # 波动率
        "volatility_5": -0.05,
        "volatility_20": -0.05,
        
        # 量价交互 (核心)
        "volume_ma_ratio_5": 0.10,
        "volume_ma_ratio_20": 0.05,
        "volume_price_divergence_5": 0.12,
        "volume_price_health": 0.10,
        
        # 非线性因子
        "vcp_score": 0.12,
        "volume_entropy_20": 0.08,
        "turnover_stable": 0.08,
        
        # 技术形态
        "rsi_14": 0.05,
        "macd": 0.08,
        "macd_signal": 0.06,
    }
    
    def __init__(self, config_path: str = "config/factors.yaml") -> None:
        """
        初始化 Alpha 研究引擎。
        
        Args:
            config_path: 因子配置文件路径
        """
        self.config_path = Path(config_path)
        self.factors: list[dict[str, Any]] = []
        self.label_config: dict[str, Any] | None = None
        self._load_config()
        
        logger.info("AlphaResearch initialized")
        logger.info(f"  Config path: {self.config_path}")
        logger.info(f"  Factors loaded: {len(self.factors)}")
    
    def _load_config(self) -> None:
        """加载因子配置文件。"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.factors = config.get("factors", [])
            self.label_config = config.get("label", None)
            logger.info(f"Loaded {len(self.factors)} factor configurations")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self.factors = []
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            self.factors = []
    
    # ==================== 基础因子计算 ====================
    
    def compute_momentum(self, df: pl.DataFrame, periods: list[int] = [5, 10, 20]) -> pl.DataFrame:
        """
        计算动量因子。
        
        Momentum = Close_t / Close_{t-n} - 1
        
        【金融逻辑】
        - 捕捉价格趋势延续性
        - 短期动量 (5 日) 反应更快
        - 长期动量 (20 日) 更稳定
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        for period in periods:
            result = result.with_columns([
                (pl.col("close") / (pl.col("close").shift(period) + self.EPSILON) - 1.0)
                .alias(f"momentum_{period}")
            ])
        
        logger.debug(f"[Momentum] Computed for periods {periods}")
        return result
    
    def compute_volatility(self, df: pl.DataFrame, periods: list[int] = [5, 20]) -> pl.DataFrame:
        """
        计算波动率因子。
        
        Volatility = Std(Returns, window=n)
        
        【金融逻辑】
        - 高波动率通常伴随低未来收益
        - 作为风险调整因子
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        # 计算收益率
        returns = pl.col("close").pct_change().fill_null(0)
        
        for period in periods:
            result = result.with_columns([
                returns.rolling_std(window_size=period, ddof=1).alias(f"volatility_{period}")
            ])
        
        logger.debug(f"[Volatility] Computed for periods {periods}")
        return result
    
    def compute_volume_ma_ratio(self, df: pl.DataFrame, periods: list[int] = [5, 20]) -> pl.DataFrame:
        """
        计算成交量均线比率。
        
        Volume_MA_Ratio = Volume / MA(Volume, n)
        
        【金融逻辑】
        - 成交量放大通常预示价格突破
        - 量价配合是趋势延续的关键
        """
        result = df.clone().with_columns([
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0)
        ])
        
        for period in periods:
            volume_ma = pl.col("volume").rolling_mean(window_size=period)
            result = result.with_columns([
                (pl.col("volume") / (volume_ma + self.EPSILON)).alias(f"volume_ma_ratio_{period}")
            ])
        
        logger.debug(f"[Volume MA Ratio] Computed for periods {periods}")
        return result
    
    # ==================== 量价非线性交互 (核心) ====================
    
    def compute_volume_price_divergence(self, df: pl.DataFrame, period: int = 5) -> pl.DataFrame:
        """
        计算量价背离因子。
        
        【核心逻辑 - 量价非线性交互】
        Volume_Price_Divergence = Price_Change - Volume_Change
        
        【金融逻辑】
        - 价升量缩：背离信号，可能反转
        - 价跌量增：背离信号，抛压加重
        - 价量齐升：健康上涨
        - 价量齐跌：弱势下跌
        
        【非线性特性】
        - 当价格和成交量同向变化时，信号增强
        - 当价格和成交量反向变化时，信号减弱
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0),
        ])
        
        # 计算 N 日价格变化
        price_change = pl.col("close") / (pl.col("close").shift(period) + self.EPSILON) - 1.0
        
        # 计算 N 日成交量变化
        volume_change = pl.col("volume") / (pl.col("volume").shift(period) + self.EPSILON) - 1.0
        
        # 量价背离 = 价格变化 - 成交量变化
        # 正值表示价强量弱（可能背离）
        # 负值表示量强价弱（可能背离）
        divergence = price_change - volume_change
        
        result = result.with_columns([
            divergence.alias(f"volume_price_divergence_{period}"),
            price_change.alias(f"price_change_{period}"),
            volume_change.alias(f"volume_change_{period}"),
        ])
        
        logger.debug(f"[Volume Price Divergence] Computed with period={period}")
        return result
    
    def compute_volume_price_health(self, df: pl.DataFrame, volume_window: int = 5, price_window: int = 5) -> pl.DataFrame:
        """
        计算量价健康度因子。
        
        【核心逻辑 - 量价非线性交互】
        根据价格和成交量的相对变化，判断走势健康程度：
        
        | 价格变化 | 成交量变化 | 健康度 | 说明           |
        |---------|-----------|-------|----------------|
        | 上涨    | 放大      | +1.0  | 健康上涨       |
        | 上涨    | 萎缩      | -0.5  | 背离，可能反转 |
        | 下跌    | 萎缩      | -0.2  | 正常调整       |
        | 下跌    | 放大      | -1.0  | 危险信号       |
        
        【非线性特性】
        - 价涨量增：正向增强
        - 价跌量增：负向增强（恐慌性抛售）
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0),
        ])
        
        # 计算成交量相对水平
        volume_ma = pl.col("volume").rolling_mean(window_size=volume_window)
        volume_ratio = pl.col("volume") / (volume_ma + self.EPSILON)
        
        # 计算价格变化
        price_change = pl.col("close") / (pl.col("close").shift(price_window) + self.EPSILON) - 1.0
        
        # 量价健康度评分（非线性映射）
        volume_price_health = pl.when(
            (price_change > 0) & (volume_ratio > 1.0)
        ).then(1.0).when(  # 价涨量增：健康
            (price_change > 0) & (volume_ratio <= 1.0)
        ).then(-0.5).when(  # 价涨量缩：背离
            (price_change <= 0) & (volume_ratio <= 1.0)
        ).then(-0.2).otherwise(-1.0)  # 价跌量缩：正常调整 / 价跌量增：危险
        
        result = result.with_columns([
            volume_price_health.alias("volume_price_health"),
            volume_ratio.alias("volume_ratio"),
            price_change.alias(f"price_change_{price_window}d"),
        ])
        
        logger.debug(f"[Volume Price Health] Computed")
        return result
    
    def compute_vcp_score(self, df: pl.DataFrame, lookback: int = 10) -> pl.DataFrame:
        """
        计算 VCP (Volatility Contraction Pattern) 评分。
        
        【核心逻辑 - 量价非线性交互】
        VCP = 价格波动率收缩 + 成交量萎缩
        
        【金融逻辑】
        - VCP 是强势上涨前的整理形态
        - 波动率收缩表明多空力量趋于平衡
        - 成交量萎缩表明抛压减轻
        
        【非线性特性】
        - 波动率越低且成交量越萎缩，VCP 评分越高
        - 是一种"压缩弹簧"效应
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0),
        ])
        
        # 计算振幅
        amplitude = (pl.col("high") - pl.col("low")) / (pl.col("close") + self.EPSILON)
        
        # 计算振幅标准差（波动率收缩）
        amplitude_std = amplitude.rolling_std(window_size=lookback, ddof=1)
        amplitude_mean = amplitude.rolling_mean(window_size=lookback)
        
        # 计算成交量相对水平
        volume_ma = pl.col("volume").rolling_mean(window_size=lookback)
        volume_ratio = pl.col("volume") / (volume_ma + self.EPSILON)
        
        # VCP 收缩 = 波动率收缩 × 成交量萎缩
        vcp_contraction = (amplitude_std / (amplitude_mean + self.EPSILON)) * volume_ratio
        
        # 归一化到 0-1（值越小表示收缩越明显）
        vcp_score = (2.0 - vcp_contraction.clip(0.0, 2.0)) / 2.0
        
        result = result.with_columns([
            vcp_score.alias("vcp_score"),
            vcp_contraction.alias("vcp_contraction"),
            amplitude.alias("price_amplitude"),
            amplitude_std.alias("amplitude_std"),
        ])
        
        logger.debug(f"[VCP Score] Computed with lookback={lookback}")
        return result
    
    def compute_volume_entropy(self, df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
        """
        计算成交量分布熵值因子。
        
        【核心逻辑 - 量价非线性交互】
        Entropy = -Σ(p * ln(p)), 其中 p = volume / rolling_sum(volume, window)
        
        【金融逻辑】
        - 熵值衡量成交量分布的"混乱程度"
        - 低熵值：成交量集中在少数交易日，可能有主力行为
        - 高熵值：成交量均匀分布，市场参与者分散
        
        【非线性特性】
        - 熵值与未来收益呈非线性关系
        - 极低熵值（主力高度控盘）和极高熵值（完全分散）都不是最优
        """
        result = df.clone().with_columns([
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0)
        ])
        
        # 计算成交量占比
        volume_sum = pl.col("volume").rolling_sum(window_size=window)
        p = pl.col("volume") / (volume_sum + self.EPSILON)
        
        # 计算熵值：-Σ(p * ln(p))
        p_log_p = (p * p.log()).fill_nan(0.0).fill_null(0.0)
        entropy = (-p_log_p.rolling_sum(window_size=window)).clip(0.0, 10.0)
        
        result = result.with_columns([
            entropy.alias("volume_entropy_20")
        ])
        
        logger.debug(f"[Volume Entropy] Computed with window={window}")
        return result
    
    def compute_turnover_stable(self, df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
        """
        计算换手率稳定性因子。
        
        Turnover_Stable = 1 / (1 + CV)
        其中 CV = Std(Turnover) / Mean(Turnover)
        
        【金融逻辑】
        - 换手率稳定的股票通常走势更稳健
        - 换手率波动过大表明筹码不稳定
        """
        result = df.clone().with_columns([
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0),
        ])
        
        # 如果没有 turnover_rate，用 volume 的变化率近似
        if "turnover_rate" in result.columns:
            turnover = pl.col("turnover_rate").fill_null(0.0)
        else:
            volume_ma = pl.col("volume").rolling_mean(window_size=5)
            turnover = (pl.col("volume") / (volume_ma + self.EPSILON) - 1.0).abs()
        
        # 计算变异系数 CV
        turnover_vol = turnover.rolling_std(window_size=lookback, ddof=1)
        turnover_mean = turnover.rolling_mean(window_size=lookback)
        turnover_cv = turnover_vol / (turnover_mean + self.EPSILON)
        
        # 稳定性 = 1 / (1 + CV)
        turnover_stable = 1.0 / (1.0 + turnover_cv)
        
        result = result.with_columns([
            turnover_stable.alias("turnover_stable"),
            turnover_cv.alias("turnover_cv"),
        ])
        
        logger.debug(f"[Turnover Stable] Computed with lookback={lookback}")
        return result
    
    # ==================== 技术指标因子 ====================
    
    def compute_rsi(self, df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
        """
        计算 RSI (Relative Strength Index)。
        
        RSI = 100 - 100 / (1 + RS)
        其中 RS = Avg(Gain, n) / Avg(Loss, n)
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        diff = pl.col("close").diff()
        gain = pl.when(diff > 0).then(diff).otherwise(0.0)
        loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
        
        avg_gain = gain.rolling_mean(window_size=period)
        avg_loss = loss.rolling_mean(window_size=period)
        
        rs = avg_gain / (avg_loss + self.EPSILON)
        rsi = (100.0 - 100.0 / (1.0 + rs)).clip(0.0, 100.0)
        
        result = result.with_columns([rsi.alias("rsi_14")])
        
        logger.debug(f"[RSI] Computed with period={period}")
        return result
    
    def compute_macd(self, df: pl.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pl.DataFrame:
        """
        计算 MACD 指标。
        
        DIF = EMA(fast) - EMA(slow)
        DEA = EMA(DIF, signal)
        MACD_Hist = 2 * (DIF - DEA)
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        ema_fast = pl.col("close").ewm_mean(span=fast, adjust=False)
        ema_slow = pl.col("close").ewm_mean(span=slow, adjust=False)
        
        dif = ema_fast - ema_slow
        dea = dif.ewm_mean(span=signal, adjust=False)
        macd_hist = 2.0 * (dif - dea)
        
        result = result.with_columns([
            dif.alias("macd"),
            dea.alias("macd_signal"),
            macd_hist.alias("macd_hist"),
        ])
        
        logger.debug(f"[MACD] Computed")
        return result
    
    # ==================== 标签计算 (T+1 收益) ====================
    
    def compute_t1_return(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 T+1 收益标签。
        
        【预测目标】
        T+1_Return = Close_{t+1} / Close_t - 1
        
        【截面排名】
        在每个交易日，对所有股票的 T+1 收益进行排名，
        归一化到 0-1 区间，作为预测目标。
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        # T+1 收益
        t1_return = pl.col("close").shift(-1) / (pl.col("close") + self.EPSILON) - 1.0
        
        result = result.with_columns([
            t1_return.alias("t1_return")
        ])
        
        logger.debug(f"[T+1 Return] Computed")
        return result
    
    def compute_cross_sectional_rank(self, df: pl.DataFrame, column: str = "t1_return") -> pl.DataFrame:
        """
        计算截面排名。
        
        【截面排名逻辑】
        在每个交易日，对所有股票的指定指标进行排名，
        归一化到 0-1 区间。
        
        Rank_Norm = (Rank - Min_Rank) / (Max_Rank - Min_Rank)
        """
        result = df.clone()
        
        if "trade_date" not in result.columns:
            logger.warning("Missing trade_date column, cannot compute cross-sectional rank")
            return result
        
        # 按日期分组计算排名
        rank = pl.col(column).rank("dense").over("trade_date").cast(pl.Float64)
        rank_min = rank.min().over("trade_date")
        rank_max = rank.max().over("trade_date")
        
        rank_norm = (rank - rank_min) / (rank_max - rank_min + self.EPSILON)
        
        result = result.with_columns([
            rank_norm.alias(f"{column}_rank"),
            rank.alias(f"{column}_rank_raw"),
        ])
        
        logger.debug(f"[Cross-Sectional Rank] Computed for {column}")
        return result
    
    # ==================== 因子预处理 ====================
    
    def winsorize(self, df: pl.DataFrame, columns: Optional[list[str]] = None,
                  lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> pl.DataFrame:
        """
        缩尾处理（去极值）。
        
        将超过百分位阈值的值截断到阈值处。
        """
        if columns is None:
            columns = self.get_factor_names()
        
        available_columns = [col for col in columns if col in df.columns]
        if not available_columns:
            return df
        
        result = df.clone()
        for col in available_columns:
            global_lower = result[col].quantile(lower_percentile / 100.0)
            global_upper = result[col].quantile(upper_percentile / 100.0)
            result = result.with_columns([
                pl.col(col).clip(lower_bound=global_lower, upper_bound=global_upper).alias(col)
            ])
        
        return result
    
    def normalize(self, df: pl.DataFrame, columns: Optional[list[str]] = None, method: str = "zscore") -> pl.DataFrame:
        """
        标准化处理。
        
        默认使用 Z-Score 标准化：
        X_norm = (X - Mean) / Std
        """
        if columns is None:
            columns = self.get_factor_names()
        
        available_columns = [col for col in columns if col in df.columns]
        if not available_columns:
            return df
        
        result = df.clone()
        
        if method == "zscore":
            # 按截面（日期）标准化
            if "trade_date" in result.columns:
                for col in available_columns:
                    result = result.with_columns([
                        ((pl.col(col) - pl.col(col).over("trade_date").mean()) / 
                         (pl.col(col).over("trade_date").std() + self.EPSILON)).alias(col)
                    ])
            else:
                # 全局标准化
                for col in available_columns:
                    mean_val = result[col].mean()
                    std_val = result[col].std()
                    result = result.with_columns([
                        ((pl.col(col) - mean_val) / (std_val + self.EPSILON)).alias(col)
                    ])
        
        return result
    
    def fill_null_values(self, df: pl.DataFrame, null_threshold: float = 0.30) -> pl.DataFrame:
        """
        智能填充空值。
        
        策略:
        1. 数值型因子：forward_fill -> backward_fill -> 列均值
        2. 缺失值超过阈值的因子：直接赋予 0 权重
        """
        result = df.clone()
        
        exclude_columns = {
            "t1_return", "t1_return_rank", "symbol", "trade_date", "ts_code"
        }
        
        factor_columns = [col for col in result.columns if col not in exclude_columns]
        
        total_rows = len(result)
        
        for col in factor_columns:
            if col not in result.columns:
                continue
            
            null_count = result[col].null_count()
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            
            if null_ratio > null_threshold:
                # 缺失值过多，直接填 0（0 权重）
                result = result.with_columns([pl.col(col).fill_null(0.0).alias(col)])
                logger.debug(f"[Fill Null] Factor '{col}' has {null_ratio:.1%} nulls, filled with 0")
            else:
                # 正常填充
                col_mean = result[col].mean()
                if col_mean is None or not np.isfinite(col_mean):
                    col_mean = 0.0
                
                result = result.with_columns([
                    pl.col(col).fill_null(strategy="forward")
                    .fill_null(strategy="backward")
                    .fill_null(col_mean)
                    .alias(col)
                ])
        
        return result
    
    # ==================== 预测评分 ====================
    
    def compute_predict_score(self, df: pl.DataFrame, weights: Optional[dict[str, float]] = None) -> pl.DataFrame:
        """
        计算综合预测评分。
        
        Predict_Score = Σ(Factor_i * Weight_i)
        
        【权重配置】
        - 基于因子历史 IC 表现动态调整
        - 量价交互因子权重较高
        """
        if weights is None:
            weights = self.FACTOR_WEIGHTS
        
        result = df.clone()
        
        # 计算加权评分
        raw_score = pl.lit(0.0)
        for factor_name, weight in weights.items():
            if factor_name in result.columns:
                raw_score = raw_score + pl.col(factor_name) * weight
        
        result = result.with_columns([raw_score.alias("raw_score")])
        
        # RSI 过滤（避免超买区域）
        if "rsi_14" in result.columns:
            rsi_filter = pl.when(pl.col("rsi_14") > 80.0).then(0.5).otherwise(1.0)
            result = result.with_columns([
                (pl.col("raw_score") * rsi_filter).alias("predict_score")
            ])
        else:
            result = result.with_columns([pl.col("raw_score").alias("predict_score")])
        
        # 量价健康度过滤
        if "volume_price_health" in result.columns:
            vp_filter = pl.when(pl.col("volume_price_health") < 0).then(0.7).otherwise(1.0)
            result = result.with_columns([
                (pl.col("predict_score") * vp_filter).alias("filtered_score")
            ])
        else:
            result = result.with_columns([pl.col("predict_score").alias("filtered_score")])
        
        # 清理中间列
        result = result.drop(["raw_score"])
        
        logger.debug(f"[Predict Score] Computed, factors used: {len(weights)}")
        return result
    
    # ==================== 因子计算主流程 ====================
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        【计算顺序】
        1. 基础动量因子
        2. 波动率因子
        3. 成交量因子
        4. 量价非线性交互因子 (核心)
        5. 技术指标因子
        6. T+1 收益标签
        7. 缺失值处理
        8. 标准化
        9. 预测评分
        """
        result = df.clone()
        
        # 1. 基础动量
        result = self.compute_momentum(result, periods=[5, 10, 20])
        
        # 2. 波动率
        result = self.compute_volatility(result, periods=[5, 20])
        
        # 3. 成交量
        result = self.compute_volume_ma_ratio(result, periods=[5, 20])
        
        # 4. 量价非线性交互 (核心)
        result = self.compute_volume_price_divergence(result, period=5)
        result = self.compute_volume_price_health(result)
        result = self.compute_vcp_score(result, lookback=10)
        result = self.compute_volume_entropy(result, window=20)
        result = self.compute_turnover_stable(result, lookback=20)
        
        # 5. 技术指标
        result = self.compute_rsi(result, period=14)
        result = self.compute_macd(result)
        
        # 6. T+1 收益标签
        result = self.compute_t1_return(result)
        result = self.compute_cross_sectional_rank(result, column="t1_return")
        
        # 7. 缺失值处理
        result = self.fill_null_values(result)
        
        # 8. 标准化（缩尾 + Z-Score）
        result = self.winsorize(result, lower_percentile=1.0, upper_percentile=99.0)
        result = self.normalize(result, method="zscore")
        
        # 9. 预测评分
        result = self.compute_predict_score(result)
        
        logger.info(f"[Compute Factors] Complete, total columns: {len(result.columns)}")
        return result
    
    def get_factor_names(self) -> list[str]:
        """获取配置的因子名称列表。"""
        return [f["name"] for f in self.factors]
    
    # ==================== IC 计算与自检 ====================
    
    def calculate_rank_ic(self, factor_values: pl.Series, label_values: pl.Series) -> float:
        """
        计算 Rank IC（Spearman 相关系数）。
        
        【损失函数】
        Spearman Rank IC 衡量因子值与未来收益排名的单调关系。
        IC > 0 表示正相关，IC < 0 表示负相关。
        """
        # 去除空值
        mask = factor_values.is_not_null() & label_values.is_not_null()
        factor_clean = factor_values.filter(mask)
        label_clean = label_values.filter(mask)
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算秩
        factor_ranks = factor_clean.rank(method='average')
        label_ranks = label_clean.rank(method='average')
        
        # 计算 Pearson 相关系数（在秩上）
        factor_np = factor_ranks.to_numpy()
        label_np = label_ranks.to_numpy()
        
        if np.std(factor_np) < 1e-10 or np.std(label_np) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(factor_np, label_np)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_t1_ic(self, df: pl.DataFrame, score_column: str = "predict_score") -> dict[str, Any]:
        """
        计算 T+1 Rank IC。
        
        【核心指标】
        - T+1 Rank IC: 预测评分与 T+1 收益排名的相关系数
        - IC > 0.05: 强预测能力
        - IC > 0.03: 中等预测能力
        - IC < 0.03: 触发 AlphaWeakWarning
        """
        if score_column not in df.columns:
            logger.error(f"Score column '{score_column}' not found")
            return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "num_days": 0}
        
        if "t1_return" not in df.columns:
            logger.error("t1_return column not found")
            return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "num_days": 0}
        
        if "trade_date" not in df.columns:
            logger.error("trade_date column not found")
            return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "num_days": 0}
        
        # 按日期分组计算 IC
        unique_dates = df["trade_date"].unique()
        ic_series = []
        
        for date in unique_dates:
            day_data = df.filter(pl.col("trade_date") == date)
            
            if len(day_data) < 10:
                continue
            
            score_values = day_data[score_column]
            label_values = day_data["t1_return"]
            
            ic = self.calculate_rank_ic(score_values, label_values)
            
            if ic != 0 or not np.isnan(ic):
                ic_series.append({
                    "trade_date": date,
                    "ic": ic,
                })
        
        if not ic_series:
            return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "num_days": 0}
        
        ic_df = pl.DataFrame(ic_series)
        ic_values = ic_df["ic"].to_numpy()
        
        mean_ic = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
        
        result = {
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "num_days": len(ic_values),
            "min_ic": float(np.min(ic_values)),
            "max_ic": float(np.max(ic_values)),
        }
        
        # 【自检机制】IC < 0.03 触发警告
        if mean_ic < 0.03:
            logger.warning(f"[AlphaWeakWarning] T+1 IC = {mean_ic:.4f} < 0.03")
            self._analyze_factor_contribution(df, score_column)
        
        return result
    
    def _analyze_factor_contribution(self, df: pl.DataFrame, score_column: str) -> None:
        """
        【自检机制】分析各因子贡献度。
        
        当 T+1 IC 低于阈值时，分析每个因子与 T+1 收益的相关性，
        找出失效因子。
        """
        logger.info("[因子贡献度分析] 开始分析各因子 IC 贡献...")
        
        factor_columns = list(self.FACTOR_WEIGHTS.keys())
        factor_ics = []
        
        for factor_name in factor_columns:
            if factor_name in df.columns and "t1_return" in df.columns:
                # 计算因子与 T+1 收益的 Rank IC
                ic = self.calculate_rank_ic(df[factor_name], df["t1_return"])
                factor_ics.append({
                    "factor": factor_name,
                    "ic": ic,
                    "weight": self.FACTOR_WEIGHTS.get(factor_name, 0),
                    "contribution": ic * self.FACTOR_WEIGHTS.get(factor_name, 0),
                })
        
        # 按 IC 排序
        factor_ics.sort(key=lambda x: abs(x["ic"]), reverse=True)
        
        logger.info("[因子贡献度分析] 结果:")
        for i, f in enumerate(factor_ics[:10], 1):  # 只显示前 10 个
            status = "✓" if abs(f["ic"]) > 0.03 else "✗"
            logger.info(f"  {i}. {f['factor']}: IC={f['ic']:.4f}, Weight={f['weight']:.2f}, Contribution={f['contribution']:.4f} {status}")
        
        # 找出失效因子
        failed_factors = [f for f in factor_ics if abs(f["ic"]) < 0.01]
        if failed_factors:
            logger.warning(f"[因子贡献度分析] 发现 {len(failed_factors)} 个失效因子:")
            for f in failed_factors:
                logger.warning(f"  - {f['factor']}: IC={f['ic']:.4f}")
    
    def calculate_factor_ic(self, df: pl.DataFrame, factor_name: str) -> float:
        """计算单个因子的 IC 值。"""
        if factor_name not in df.columns or "t1_return" not in df.columns:
            return 0.0
        
        return self.calculate_rank_ic(df[factor_name], df["t1_return"])
    
    def get_top_factor_ic(self, df: pl.DataFrame) -> dict[str, float]:
        """
        获取最高 IC 的因子。
        
        【验收指标】
        Top Factor IC > 0.04: 必须有至少一个核心因子具备独立战斗力
        """
        factor_columns = list(self.FACTOR_WEIGHTS.keys())
        factor_ics = {}
        
        for factor_name in factor_columns:
            if factor_name in df.columns:
                ic = self.calculate_factor_ic(df, factor_name)
                factor_ics[factor_name] = ic
        
        if not factor_ics:
            return {"factor": None, "ic": 0.0}
        
        top_factor = max(factor_ics, key=lambda k: abs(factor_ics[k]))
        
        result = {
            "factor": top_factor,
            "ic": factor_ics[top_factor],
            "all_ics": factor_ics,
        }
        
        # 【验收检查】
        if abs(result["ic"]) < 0.04:
            logger.warning(f"[验收警告] Top Factor IC = {abs(result['ic']):.4f} < 0.04")
        
        return result
    
    # ==================== 完整分析流程 ====================
    
    def run_alpha_analysis(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        运行完整的 Alpha 分析流程。
        
        Returns:
            dict: 分析结果，包括:
                - processed_df: 处理后的数据
                - t1_ic: T+1 IC 统计
                - top_factor: Top 因子 IC
                - passed: 是否通过验收
        """
        logger.info("=" * 60)
        logger.info("V101 Alpha Research - Prediction Analysis")
        logger.info("=" * 60)
        
        # 1. 计算因子
        processed_df = self.compute_factors(df)
        
        # 2. 计算 T+1 IC
        t1_ic = self.calculate_t1_ic(processed_df)
        
        logger.info(f"[T+1 IC] Mean={t1_ic['mean_ic']:.4f}, IR={t1_ic['ic_ir']:.2f}")
        
        # 3. 获取 Top 因子
        top_factor = self.get_top_factor_ic(processed_df)
        
        logger.info(f"[Top Factor] {top_factor['factor']}: IC={top_factor['ic']:.4f}")
        
        # 4. 验收判断
        passed = (
            t1_ic["mean_ic"] > 0.05 and  # T+1 IC > 0.05
            t1_ic["ic_ir"] > 0.6 and      # IC IR > 0.6
            abs(top_factor["ic"]) > 0.04  # Top Factor IC > 0.04
        )
        
        if passed:
            logger.info("[验收结果] PASSED - Alpha 预测能力达标")
        else:
            logger.warning("[验收结果] FAILED - Alpha 预测能力不足")
            if t1_ic["mean_ic"] <= 0.05:
                logger.warning(f"  - T+1 IC ({t1_ic['mean_ic']:.4f}) < 0.05")
            if t1_ic["ic_ir"] <= 0.6:
                logger.warning(f"  - IC IR ({t1_ic['ic_ir']:.2f}) <= 0.6")
            if abs(top_factor["ic"]) <= 0.04:
                logger.warning(f"  - Top Factor IC ({abs(top_factor['ic']):.4f}) <= 0.04")
        
        return {
            "processed_df": processed_df,
            "t1_ic": t1_ic,
            "top_factor": top_factor,
            "passed": passed,
        }


def get_alpha_research(config_path: str = "config/factors.yaml") -> AlphaResearch:
    """获取 AlphaResearch 实例。"""
    return AlphaResearch(config_path)