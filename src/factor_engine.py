"""
Factor Engine Module - Multi-Factor Model for Stock Selection.
【修复 - Iteration 18】添加 volume_entropy 因子计算支持。
"""

import yaml
import polars as pl
from pathlib import Path
from typing import Any, Optional, Tuple
from loguru import logger
import numpy as np
import math

pl.Config.set_streaming_chunk_size(10000)


class FactorEngine:
    """多因子模型引擎，使用 Polars 进行向量化计算。"""
    
    BASE_COLUMNS = {
        "open", "high", "low", "close", "pre_close",
        "change", "pct_chg", "volume", "amount",
        "pct_change", "adj_close", "adj_open", "adj_high", "adj_low",
        "symbol", "Date", "trade_date", "ts_code",
        "turnover_rate", "adj_factor",
    }
    
    FACTOR_WEIGHTS = {
        "momentum_5": 0.12, "momentum_10": 0.08, "momentum_20": 0.04,
        "volatility_5": -0.04, "volatility_20": -0.04,
        "volume_ma_ratio_5": 0.08, "volume_ma_ratio_20": 0.04,
        "rsi_14": 0.04, "macd": 0.12, "macd_signal": 0.08,
        "volume_price_divergence_5": 0.08,
        "vcp_score": 0.10, "turnover_stable": 0.08, "smart_money_signal": 0.10,
        "price_position_20": 0.04, "ma_deviation_5": 0.04,
        "volume_entropy_20": 0.05,  # 【新增 - Iteration 18】熵值因子权重
    }
    
    RSI_OVERBOUGHT_THRESHOLD = 80.0
    RSI_OVERSOLD_THRESHOLD = 20.0
    VOLUME_SHRINK_THRESHOLD = 0.8
    EPSILON = 1e-6
    
    def __init__(self, config_path: str, validate: bool = True) -> None:
        self.config_path = Path(config_path)
        self.factors: list[dict[str, Any]] = []
        self.label_config: dict[str, Any] | None = None
        self.validation_errors: list[str] = []
        self._load_config()
        if validate:
            self._validate_config()
    
    def _validate_config(self) -> None:
        logger.info("Validating factor configurations...")
        self.validation_errors = []
        
        test_df = pl.DataFrame({
            "open": [10.0, 10.5, 11.0], "high": [10.8, 11.2, 11.5],
            "low": [9.8, 10.2, 10.8], "close": [10.5, 11.0, 11.2],
            "pre_close": [10.0, 10.5, 11.0], "change": [0.5, 0.5, 0.2],
            "pct_chg": [0.05, 0.045, 0.018], "volume": [1000, 1200, 1100],
            "amount": [10000, 12000, 11000], "pct_change": [0.05, 0.045, 0.018],
            "adj_close": [10.5, 11.0, 11.2], "turnover_rate": [1.5, 2.0, 1.8],
            "symbol": ["A", "A", "A"], "trade_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })
        
        context = {col: test_df[col] for col in test_df.columns}
        context["pl"] = pl
        context["float"] = float
        context["log"] = math.log
        
        for factor in self.factors:
            factor_name = factor.get("name", "unknown")
            expression = factor.get("expression", "")
            try:
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min, "log": math.log}}
                eval(expression, eval_globals, context)
                logger.debug(f"Factor '{factor_name}' expression is valid")
            except Exception as e:
                self.validation_errors.append(f"Factor '{factor_name}': {e}")
                logger.error(f"Validation failed for factor '{factor_name}': {e}")
        
        if self.validation_errors:
            logger.warning(f"Configuration validation completed with {len(self.validation_errors)} error(s)")
        else:
            logger.info("Configuration validation passed: all expressions are valid")
    
    def _load_config(self) -> None:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.factors = config.get("factors", [])
            self.label_config = config.get("label", None)
            logger.info(f"Loaded {len(self.factors)} factor configurations")
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise
    
    def winsorize(self, df: pl.DataFrame, columns: Optional[list[str]] = None,
                  lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> pl.DataFrame:
        if columns is None:
            columns = self.get_factor_names()
        available_columns = [col for col in columns if col in df.columns]
        if not available_columns:
            return df
        result = df.clone()
        for col in available_columns:
            global_lower = result[col].quantile(lower_percentile / 100.0)
            global_upper = result[col].quantile(upper_percentile / 100.0)
            result = result.with_columns([pl.col(col).clip(lower_bound=global_lower, upper_bound=global_upper).alias(col)])
        return result
    
    def normalize(self, df: pl.DataFrame, columns: Optional[list[str]] = None, method: str = "zscore") -> pl.DataFrame:
        if columns is None:
            columns = self.get_factor_names()
        available_columns = [col for col in columns if col in df.columns]
        if not available_columns:
            return df
        result = df.clone()
        if method == "zscore":
            if "symbol" in result.columns:
                for col in available_columns:
                    result = result.with_columns([
                        ((pl.col(col) - pl.col(col).over("symbol").mean()) / 
                         (pl.col(col).over("symbol").std() + self.EPSILON)).alias(col)
                    ])
            else:
                for col in available_columns:
                    mean_val, std_val = result[col].mean(), result[col].std()
                    result = result.with_columns([((pl.col(col) - mean_val) / (std_val + self.EPSILON)).alias(col)])
        return result
    
    def preprocess(self, df: pl.DataFrame, columns: Optional[list[str]] = None,
                   winsorize_percentiles: Tuple[float, float] = (1.0, 99.0), normalize_method: str = "zscore") -> pl.DataFrame:
        exclude_columns = {"rsi_14", "rsi_7", "rsi_21", "sharpe_label", "future_max_return", 
                          "future_volatility", "future_return_5", "predict_score", "filtered_score",
                          "macd", "macd_signal", "macd_hist"}
        if columns is None:
            columns = [col for col in self.get_factor_names() if col not in exclude_columns]
        else:
            columns = [col for col in columns if col not in exclude_columns]
        df_processed = self.winsorize(df, columns=columns, lower_percentile=winsorize_percentiles[0], upper_percentile=winsorize_percentiles[1])
        df_processed = self.normalize(df_processed, columns=columns, method=normalize_method)
        return df_processed
    
    def normalize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        if "pct_chg" in result.columns and "pct_change" not in result.columns:
            result = result.with_columns([pl.col("pct_chg").alias("pct_change")])
        if "pct_change" in result.columns and "pct_chg" not in result.columns:
            result = result.with_columns([pl.col("pct_change").alias("pct_chg")])
        if "turnover_rate" not in result.columns and "volume" in result.columns:
            result = result.with_columns([
                (pl.col("volume") / (pl.col("volume").rolling_mean(window_size=20) + self.EPSILON)).alias("turnover_rate")
            ])
        return result
    
    def compute_bias(self, df: pl.DataFrame, period: int = 60, column: str = "close") -> pl.DataFrame:
        result = df.clone().with_columns([pl.col(column).cast(pl.Float64, strict=False)])
        ma_n = pl.col(column).rolling_mean(window_size=period)
        bias_raw = (pl.col(column) / (ma_n + self.EPSILON) - 1.0)
        bias_inverted = -bias_raw
        return result.with_columns([bias_inverted.alias(f"bias_{period}")])
    
    def compute_rsi(self, df: pl.DataFrame, period: int = 14, column: str = "close") -> pl.DataFrame:
        result = df.clone().with_columns([pl.col(column).cast(pl.Float64, strict=False)])
        diff = pl.col(column).diff()
        gain = pl.when(diff > 0).then(diff).otherwise(0.0)
        loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
        avg_gain, avg_loss = gain.rolling_mean(window_size=period), loss.rolling_mean(window_size=period)
        rs = avg_gain / (avg_loss + self.EPSILON)
        rsi = (100.0 - 100.0 / (1.0 + rs)).clip(0.0, 100.0)
        return result.with_columns([rsi.alias(f"rsi_{period}")])
    
    def compute_macd(self, df: pl.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = "close") -> pl.DataFrame:
        result = df.clone().with_columns([pl.col(column).cast(pl.Float64, strict=False)])
        ema_fast = pl.col(column).ewm_mean(span=fast_period, adjust=False)
        ema_slow = pl.col(column).ewm_mean(span=slow_period, adjust=False)
        dif = ema_fast - ema_slow
        dea = dif.ewm_mean(span=signal_period, adjust=False)
        macd_hist = 2.0 * (dif - dea)
        return result.with_columns([dif.alias("macd"), dea.alias("macd_signal"), macd_hist.alias("macd_hist")])
    
    def compute_volume_price_coordination(self, df: pl.DataFrame, volume_window: int = 5, price_window: int = 5) -> pl.DataFrame:
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        ])
        if "pct_change" not in result.columns:
            if "pct_chg" in result.columns:
                result = result.with_columns([pl.col("pct_chg").alias("pct_change")])
            else:
                result = result.with_columns([(pl.col("close") / pl.col("close").shift(1) - 1).alias("pct_change")])
        result = result.with_columns([pl.col("pct_change").cast(pl.Float64, strict=False)])
        volume_ma = pl.col("volume").rolling_mean(window_size=volume_window)
        volume_ratio = pl.col("volume") / (volume_ma + self.EPSILON)
        price_change = pl.col("close") / pl.col("close").shift(price_window) - 1
        volume_price_health = pl.when((price_change > 0) & (volume_ratio > 1.0)).then(1.0).when(
            (price_change > 0) & (volume_ratio <= 1.0)).then(-0.5).when(
            (price_change <= 0) & (volume_ratio <= 1.0)).then(-0.2).otherwise(-1.0)
        volume_shrink_flag = (volume_ratio < self.VOLUME_SHRINK_THRESHOLD).cast(pl.Float64)
        volume_change = pl.col("volume") / pl.col("volume").shift(price_window) - 1
        price_volume_divergence = price_change - volume_change
        return result.with_columns([
            volume_price_health.alias("volume_price_health"),
            volume_shrink_flag.alias("volume_shrink_flag"),
            price_volume_divergence.alias("price_volume_divergence"),
        ])
    
    def compute_vcp(self, df: pl.DataFrame, lookback: int = 10) -> pl.DataFrame:
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        ])
        amplitude = (pl.col("high") - pl.col("low")) / (pl.col("close") + self.EPSILON)
        amplitude_std = amplitude.rolling_std(window_size=lookback, ddof=1)
        amplitude_mean = amplitude.rolling_mean(window_size=lookback)
        volume_ma = pl.col("volume").rolling_mean(window_size=lookback)
        volume_ratio = pl.col("volume") / (volume_ma + self.EPSILON)
        vcp_contraction = (amplitude_std / (amplitude_mean + self.EPSILON)) * volume_ratio
        vcp_score = vcp_contraction.clip(0.0, 2.0) / 2.0
        return result.with_columns([
            vcp_score.alias("vcp_score"), vcp_contraction.alias("vcp_contraction"),
            amplitude.alias("price_amplitude"), amplitude_std.alias("amplitude_std"),
        ])
    
    def compute_turnover_vol(self, df: pl.DataFrame, lookback: int = 20) -> pl.DataFrame:
        result = df.clone().with_columns([pl.col("volume").cast(pl.Float64, strict=False)])
        if "turnover_rate" in result.columns:
            turnover = pl.col("turnover_rate").fill_null(0.0)
        else:
            volume_ma = pl.col("volume").rolling_mean(window_size=5)
            turnover = (pl.col("volume") / (volume_ma + self.EPSILON) - 1.0).abs()
        turnover_vol = turnover.rolling_std(window_size=lookback, ddof=1)
        turnover_mean = turnover.rolling_mean(window_size=lookback)
        turnover_cv = turnover_vol / (turnover_mean + self.EPSILON)
        turnover_stable = 1.0 / (1.0 + turnover_cv)
        return result.with_columns([
            turnover_vol.alias("turnover_vol"), turnover_cv.alias("turnover_cv"),
            turnover_stable.alias("turnover_stable"), turnover_mean.alias("turnover_mean"),
        ])
    
    def compute_volume_entropy(self, df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
        """
        计算成交量分布熵值因子 (Volume Entropy) - 【修复 - Iteration 18】。
        使用纯 Polars 原生函数实现熵值计算，避免 __import__ 在 eval 中被禁用。
        Entropy = -Σ(p * ln(p)), 其中 p = volume / rolling_sum(volume, window)
        """
        result = df.clone().with_columns([
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0).alias("volume_filled")
        ])
        volume_sum = pl.col("volume_filled").rolling_sum(window_size=window)
        p = pl.col("volume_filled") / (volume_sum + self.EPSILON)
        p_log_p = (p * p.log()).fill_nan(0.0).fill_null(0.0)
        entropy = (-p_log_p.rolling_sum(window_size=window)).clip(0.0, 10.0)
        result = result.with_columns([entropy.alias("volume_entropy_20")])
        cols_to_drop = [col for col in ["volume_filled"] if col in result.columns]
        if cols_to_drop:
            result = result.drop(cols_to_drop)
        logger.debug(f"[Volume Entropy] Computed with window={window}, output rows={len(result)}")
        return result
    
    def compute_vol_price_resilience(self, df: pl.DataFrame, lookback: int = 10) -> pl.DataFrame:
        """
        【V2.3 增强】计算价格 - 成交量 Resilience 因子（洗盘结束信号）
        
        【金融逻辑】
        - 当价格回撤时，如果成交量显著萎缩，表明抛压减弱
        - 成交量萎缩越明显，洗盘结束的可能性越大
        
        【V2.3 修复】
        - 所有除法操作加入 1e-6 保护
        - 使用 clip 确保值在有效范围内
        
        Args:
            df: 输入 DataFrame
            lookback: 滚动平均窗口（默认 10）
            
        Returns:
            包含 vol_price_resilience 列的 DataFrame
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        ])
        
        # 计算价格回撤（5 日）- 加入除零保护
        price_pullback = pl.col("close") / (pl.col("close").shift(5) + self.EPSILON) - 1.0
        
        # 计算成交量相对水平 - 加入除零保护
        volume_ma_20 = pl.col("volume").rolling_mean(window_size=20)
        volume_ratio = pl.col("volume") / (volume_ma_20 + self.EPSILON)
        
        # 计算 resilience 信号：价格回撤时的成交量萎缩程度
        # 值越大，表明洗盘越充分
        vol_shrink_intensity = pl.when(price_pullback < 0).then(
            (1.0 - volume_ratio).clip(0.0, 1.0)  # 成交量萎缩时为正
        ).otherwise(
            0.0  # 价格上涨时不计入
        )
        
        # 滚动平均，平滑信号
        vol_price_resilience = vol_shrink_intensity.rolling_mean(window_size=lookback)
        
        result = result.with_columns([
            vol_price_resilience.alias("vol_price_resilience"),
            price_pullback.alias("price_pullback"),
            volume_ratio.alias("volume_ratio_during_pullback"),
        ])
        
        logger.debug(f"[Vol Price Resilience] Computed with lookback={lookback}, output rows={len(result)}")
        return result
    
    def compute_relative_strength_sector(self, df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
        """
        【V2.4 优化】计算相对强度因子 (Relative Strength vs Sector)
        
        【V2.4 优化】
        - 增加对全市场 Top 10% 标的的偏好权重
        - 相对强度计算时，对 Top 10% 标的给予额外 10% 权重提升
        
        【金融逻辑】
        - 计算标的相对于所属行业（或全样本均值）的相对强度
        - 用于在震荡市筛选领涨品种
        - RS = 标的 N 日收益率 / 行业 N 日收益率
        
        Args:
            df: 输入 DataFrame
            period: 计算周期（默认 20 日）
            
        Returns:
            包含 relative_strength_sector 列的 DataFrame
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 计算标的 N 日收益率 - 加入除零保护
        stock_return = pl.col("close") / (pl.col("close").shift(period) + self.EPSILON) - 1.0
        
        # 按行业计算平均收益率
        if "industry" in result.columns:
            # 有行业信息时，计算行业平均
            industry_return = stock_return.over("industry").mean()
            relative_strength = stock_return / (industry_return + self.EPSILON)
        else:
            # 无行业信息时，使用全样本均值
            mean_return = stock_return.mean()
            relative_strength = stock_return / (mean_return + self.EPSILON)
        
        # V2.4: 对 Top 10% 标的给予额外权重
        # 计算全市场收益率排名
        if len(result) > 0:
            # 使用 dense 排名，然后归一化到 0-1
            return_rank = stock_return.rank(method="dense", descending=True)
            return_percentile = return_rank / return_rank.max()
            
            # 对 Top 10% (percentile >= 0.9) 给予 10% 额外权重
            top_10_bonus = pl.when(return_percentile >= 0.9).then(1.1).otherwise(1.0)
            relative_strength = relative_strength * top_10_bonus
        
        # 限制相对强度在合理范围内 - V2.4 放宽上限至 2.5
        relative_strength_clipped = relative_strength.clip(0.5, 2.5)
        
        result = result.with_columns([
            relative_strength_clipped.alias("relative_strength_sector"),
            stock_return.alias(f"return_{period}d"),
        ])
        
        logger.debug(f"[Relative Strength] Computed with period={period}, output rows={len(result)}")
        return result
    
    def compute_momentum_squeeze(self, df: pl.DataFrame, momentum_period: int = 5, vol_period: int = 20) -> pl.DataFrame:
        """
        【V2.4 新增】计算 Momentum Squeeze 因子
        
        【金融逻辑】
        - 捕捉那些低波动上涨、即将进入主升浪的标的
        - Momentum Squeeze = Momentum_5 / Volatility_20
        - 值越大，表明单位波动下的动量收益越高，即" squeeze"状态
        
        Args:
            df: 输入 DataFrame
            momentum_period: 动量周期（默认 5 日）
            vol_period: 波动率周期（默认 20 日）
            
        Returns:
            包含 momentum_squeeze 列的 DataFrame
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 计算 N 日动量
        momentum_n = pl.col("close") / (pl.col("close").shift(momentum_period) + self.EPSILON) - 1.0
        
        # 计算 N 日波动率
        returns = pl.col("close").pct_change().fill_null(0)
        volatility_n = returns.rolling_std(window_size=vol_period, ddof=1)
        
        # 计算 Momentum Squeeze
        # 除以波动率，得到单位波动下的动量收益
        momentum_squeeze = momentum_n / (volatility_n + self.EPSILON)
        
        # 限制在合理范围内
        momentum_squeeze_clipped = momentum_squeeze.clip(-5.0, 5.0)
        
        result = result.with_columns([
            momentum_squeeze_clipped.alias("momentum_squeeze"),
            momentum_n.alias(f"momentum_{momentum_period}d"),
            volatility_n.alias(f"volatility_{vol_period}d"),
        ])
        
        logger.debug(f"[Momentum Squeeze] Computed, output rows={len(result)}")
        return result
    
    def compute_smart_money(self, df: pl.DataFrame, lookback: int = 10) -> pl.DataFrame:
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        ])
        if "pct_change" not in result.columns:
            if "pct_chg" in result.columns:
                result = result.with_columns([pl.col("pct_chg").alias("pct_change")])
            else:
                result = result.with_columns([(pl.col("close") / pl.col("close").shift(1) - 1).alias("pct_change")])
        result = result.with_columns([pl.col("pct_change").cast(pl.Float64, strict=False)])
        is_up = (pl.col("pct_change") > 0).cast(pl.Float64)
        is_down = (pl.col("pct_change") < 0).cast(pl.Float64)
        up_volume, down_volume = is_up * pl.col("volume"), is_down * pl.col("volume")
        up_volume_sum = up_volume.rolling_sum(window_size=lookback)
        down_volume_sum = down_volume.rolling_sum(window_size=lookback)
        up_days = is_up.rolling_sum(window_size=lookback)
        down_days = is_down.rolling_sum(window_size=lookback)
        avg_up_volume = up_volume_sum / (up_days + self.EPSILON)
        avg_down_volume = down_volume_sum / (down_days + self.EPSILON)
        smart_money_ratio = avg_down_volume / (avg_up_volume + self.EPSILON)
        smart_money_signal = 1.0 / (1.0 + smart_money_ratio)
        return result.with_columns([
            smart_money_ratio.alias("smart_money_ratio"), smart_money_signal.alias("smart_money_signal"),
            avg_up_volume.alias("avg_up_volume"), avg_down_volume.alias("avg_down_volume"),
        ])
    
    def _safe_drop_columns(self, df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        cols_to_drop = [col for col in columns if col in df.columns]
        if cols_to_drop:
            return df.drop(cols_to_drop)
        return df
    
    def compute_predict_score(self, df: pl.DataFrame, weights: Optional[dict[str, float]] = None, use_preprocessed: bool = True) -> pl.DataFrame:
        if weights is None:
            weights = self.FACTOR_WEIGHTS
        result = df.clone()
        raw_score = pl.lit(0.0)
        for factor_name, weight in weights.items():
            if factor_name in result.columns:
                raw_score = raw_score + pl.col(factor_name) * weight
        result = result.with_columns([raw_score.alias("raw_score")])
        rsi_column = "rsi_14" if "rsi_14" in result.columns else None
        if rsi_column:
            rsi_filter = pl.when(pl.col(rsi_column) > self.RSI_OVERBOUGHT_THRESHOLD).then(0.5).otherwise(1.0)
            result = result.with_columns([(pl.col("raw_score") * rsi_filter).alias("score_after_rsi")])
        else:
            result = result.with_columns([pl.col("raw_score").alias("score_after_rsi")])
        if "volume_price_health" in result.columns:
            vp_filter = pl.when(pl.col("volume_price_health") < 0).then(0.7).otherwise(1.0)
            result = result.with_columns([(pl.col("score_after_rsi") * vp_filter).alias("predict_score")])
        else:
            result = result.with_columns([pl.col("score_after_rsi").alias("predict_score")])
        result = self._safe_drop_columns(result, ["raw_score", "score_after_rsi"])
        return result
    
    def apply_rsi_filter(self, df: pl.DataFrame, rsi_threshold: float = 80.0, discount_factor: float = 0.5) -> pl.DataFrame:
        result = df.clone()
        rsi_column = "rsi_14" if "rsi_14" in result.columns else None
        if rsi_column and "predict_score" in result.columns:
            rsi_filter = pl.when(pl.col(rsi_column) > rsi_threshold).then(discount_factor).otherwise(1.0)
            result = result.with_columns([(pl.col("predict_score") * rsi_filter).alias("filtered_score")])
        elif "predict_score" in result.columns:
            result = result.with_columns([pl.col("predict_score").alias("filtered_score")])
        return result
    
    def compute_hist_sharpe(self, df: pl.DataFrame, window: int = 20) -> pl.DataFrame:
        """计算历史夏普比率因子。【修复】确保 symbol 列存在。"""
        result = df.clone().with_columns([pl.col("close").cast(pl.Float64, strict=False)])
        if "pct_chg" not in result.columns:
            if "pct_change" in result.columns:
                result = result.with_columns([pl.col("pct_change").alias("pct_chg")])
            else:
                result = result.with_columns([(pl.col("close") / pl.col("close").shift(1) - 1).alias("pct_chg")])
        result = result.with_columns([pl.col("pct_chg").cast(pl.Float64, strict=False)])
        has_symbol = "symbol" in result.columns
        if not has_symbol:
            raise ValueError("Missing required column 'symbol' for hist_sharpe computation.")
        result = result.with_columns([
            pl.col("pct_chg").fill_null(strategy="forward").over("symbol")
            .fill_null(strategy="backward").over("symbol").fill_null(0.0).alias("pct_chg_filled")
        ])
        cumulative_return = (pl.col("close") / pl.col("close").shift(window).over("symbol") - 1.0).alias("cumulative_return")
        volatility = pl.col("pct_chg_filled").rolling_std(window_size=window, ddof=1, min_samples=window).over("symbol").alias("volatility_raw")
        result = result.with_columns([cumulative_return, volatility])
        result = result.with_columns([
            pl.when(pl.col("volatility_raw").is_null()).then(0.0).otherwise(pl.col("volatility_raw")).alias("volatility_filled"),
            pl.when(pl.col("cumulative_return").is_null()).then(0.0).otherwise(pl.col("cumulative_return")).alias("cumulative_return_filled"),
        ])
        hist_sharpe = (pl.col("cumulative_return_filled") / (pl.col("volatility_filled") + self.EPSILON)).fill_nan(0.0).fill_null(0.0).clip(-10.0, 10.0)
        result = result.with_columns([hist_sharpe.alias("hist_sharpe_20d")])
        result = self._safe_drop_columns(result, ["pct_chg_filled", "cumulative_return", "cumulative_return_filled", "volatility_raw", "volatility_filled"])
        return result
    
    def compute_label_5d(self, df: pl.DataFrame, future_window: int = 5, use_quantile: bool = True, index_return: Optional[pl.Series] = None) -> pl.DataFrame:
        """计算未来 5 日趋势标签。【修复】确保 trade_date 列存在。"""
        result = df.clone().with_columns([pl.col("close").cast(pl.Float64, strict=False)])
        if "high" in result.columns:
            result = result.with_columns([pl.col("high").cast(pl.Float64, strict=False)])
        future_return_5d = (pl.col("close").shift(-future_window) / (pl.col("close").shift(-1) + self.EPSILON) - 1.0).alias("future_return_5d")
        if "high" in result.columns:
            future_highs = [pl.col("high").shift(-i) for i in range(1, future_window + 1)]
            future_max_price = pl.max_horizontal(future_highs)
        else:
            future_closes = [pl.col("close").shift(-i) for i in range(1, future_window + 1)]
            future_max_price = pl.max_horizontal(future_closes)
        future_max_return_5d = (future_max_price / (pl.col("close") + self.EPSILON) - 1.0).alias("future_max_return_5d")
        result = result.with_columns([future_return_5d, future_max_return_5d])
        if index_return is not None:
            excess_return = (pl.col("future_return_5d") - index_return).alias("excess_return_5d")
            result = result.with_columns([excess_return])
        if use_quantile:
            has_trade_date = "trade_date" in result.columns
            if not has_trade_date:
                raise ValueError("Missing required column 'trade_date' for label computation.")
            label_rank = pl.col("future_return_5d").rank("dense").over("trade_date").cast(pl.Float64)
            label_rank_norm = (label_rank - label_rank.min().over("trade_date")) / (label_rank.max().over("trade_date") - label_rank.min().over("trade_date") + self.EPSILON)
            label_5d = pl.when(label_rank_norm >= 0.8).then(2).when(label_rank_norm <= 0.2).then(0).otherwise(1)
            result = result.with_columns([label_5d.alias("label_5d"), label_rank_norm.alias("label_rank_norm")])
        else:
            result = result.with_columns([pl.col("future_return_5d").alias("label_5d")])
        return result
    
    def compute_sharpe_target(self, df: pl.DataFrame, future_window: int = 3, min_samples: int = 1) -> pl.DataFrame:
        result = df.clone().with_columns([pl.col("close").cast(pl.Float64, strict=False)])
        price_col = "high" if "high" in result.columns else "close"
        if "high" in result.columns:
            result = result.with_columns([pl.col("high").cast(pl.Float64, strict=False)])
        future_highs = [pl.col(price_col).shift(-i) for i in range(1, future_window + 1)]
        future_max = pl.max_horizontal(future_highs)
        future_max_return = (future_max / pl.col("close") - 1.0).alias("future_max_return_target")
        future_returns = [(pl.col("close").shift(-i) / pl.col("close").shift(-(i-1)) - 1.0) for i in range(1, future_window + 1)]
        if future_returns:
            mean_return = sum(future_returns) / len(future_returns)
            variance = sum((ret - mean_return) ** 2 for ret in future_returns) / len(future_returns)
            future_volatility = (variance + self.EPSILON).pow(0.5).alias("future_volatility_target")
        else:
            future_volatility = pl.lit(self.EPSILON).alias("future_volatility_target")
        sharpe_target = (future_max_return / (future_volatility + self.EPSILON)).alias("sharpe_target")
        return result.with_columns([future_max_return, future_volatility, sharpe_target])
    
    def compute_label(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        result = self.compute_label_5d(result, future_window=5, use_quantile=True)
        result = self.compute_sharpe_target(result)
        if self.label_config:
            label_name = self.label_config["name"]
            expression = self.label_config["expression"]
            try:
                context = {col: result[col] for col in result.columns}
                context["pl"] = pl
                context["float"] = float
                context["log"] = math.log
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min, "log": math.log}}
                label_values = eval(expression, eval_globals, context)
                if not label_name.endswith("_target"):
                    label_name = label_name + "_target"
                result = result.with_columns([pl.Series(label_name, label_values)])
            except Exception as e:
                logger.error(f"Failed to compute traditional label: {e}")
        return result
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算所有配置的因子，并进行预处理和评分。
        
        【修复 - Iteration 19】
        1. 增加 fill_null 策略，减少因单个因子缺失导致整行数据被剔除
        2. 优化 dropna 逻辑，使用更宽松的 null 处理
        3. 添加 volume_entropy 计算
        """
        result = self.normalize_column_names(df)
        result = self._compute_base_factors(result)
        result = self.compute_rsi(result, period=14)
        result = self.compute_macd(result)
        result = self.compute_bias(result, period=60)
        result = self.compute_volume_price_coordination(result)
        result = self.compute_vcp(result, lookback=10)
        result = self.compute_turnover_vol(result, lookback=20)
        result = self.compute_smart_money(result, lookback=10)
        result = self.compute_volume_entropy(result, window=20)  # 【新增 - Iteration 18】
        result = self.compute_hist_sharpe(result, window=20)
        
        # 【修复 - Iteration 19】在预处理前填充空值
        result = self._fill_null_values(result)
        
        result = self.preprocess(result)
        result = self.compute_predict_score(result)
        result = self.apply_rsi_filter(result)
        result = self.compute_label(result)
        return result
    
    def _fill_null_values(self, df: pl.DataFrame, null_threshold: float = 0.30) -> pl.DataFrame:
        """
        【新增 - Iteration 19】【增强 - Iteration 22】智能填充空值，减少数据丢失
        
        策略:
        1. 数值型因子：使用 forward_fill -> backward_fill -> 列均值 的级联填充
        2. 对于 symbol 分组数据：按 symbol 分别填充
        3. 【Iteration 22 新增】对缺失值超过 30% 的因子直接赋予 0 权重（用 0 填充）
        
        Args:
            df: 输入 DataFrame
            null_threshold: 缺失值阈值，超过此比例的因子将被赋予 0 权重 (默认 30%)
        """
        result = df.clone()
        
        # 需要填充的因子列（排除标签列和特殊列）
        exclude_columns = {
            "sharpe_label", "future_max_return", "future_volatility",
            "future_return_5d", "future_return_5d", "label_5d",
            "predict_score", "filtered_score", "raw_score", "score_after_rsi",
            "symbol", "Date", "trade_date", "ts_code"
        }
        
        factor_columns = [col for col in result.columns if col not in exclude_columns]
        
        # 按 symbol 分组填充（如果存在 symbol 列）
        has_symbol = "symbol" in result.columns
        total_rows = len(result)
        
        # 【Iteration 22】统计缺失值超过阈值的因子
        high_null_factors = []
        normal_factors = []
        
        for col in factor_columns:
            if col not in result.columns:
                continue
            
            null_count = result[col].null_count()
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            
            if null_ratio > null_threshold:
                high_null_factors.append((col, null_ratio))
            else:
                normal_factors.append(col)
        
        # 【Iteration 22】对缺失值超过阈值的因子赋予 0 权重
        for col, null_ratio in high_null_factors:
            result = result.with_columns([
                pl.col(col).fill_null(0.0).alias(col)
            ])
            logger.info(f"[缺失值处理] 因子 '{col}' 缺失值比例 {null_ratio:.1%} > 阈值 {null_threshold:.0%}, 赋予 0 权重")
        
        # 对正常因子进行常规填充
        for col in normal_factors:
            if col not in result.columns:
                continue
            
            null_count = result[col].null_count()
            if null_count == 0:
                continue
            
            # 计算列均值用于最终填充
            col_mean = result[col].mean()
            if col_mean is None or not np.isfinite(col_mean):
                col_mean = 0.0
            
            if has_symbol:
                # 按 symbol 分组填充
                result = result.with_columns([
                    pl.col(col).fill_null(strategy="forward").over("symbol")
                    .fill_null(strategy="backward").over("symbol")
                    .fill_null(col_mean)
                    .alias(col)
                ])
            else:
                # 全局填充
                result = result.with_columns([
                    pl.col(col).fill_null(strategy="forward")
                    .fill_null(strategy="backward")
                    .fill_null(col_mean)
                    .alias(col)
                ])
        
        logger.debug(f"[Fill Null] 高缺失值因子：{len(high_null_factors)}, 正常填充因子：{len(normal_factors)}")
        return result
    
    def _compute_base_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        result = df.clone()
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "turnover_rate", "pre_close", "change", "pct_chg", "pct_change"]
        for col in numeric_columns:
            if col in result.columns:
                result = result.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        for factor in self.factors:
            factor_name = factor["name"]
            expression = factor["expression"]
            try:
                context = {col: result[col] for col in result.columns}
                context["pl"] = pl
                context["float"] = float
                context["log"] = math.log
                eval_globals = {"__builtins__": {"float": float, "abs": abs, "max": max, "min": min, "log": math.log}}
                factor_values = eval(expression, eval_globals, context)
                result = result.with_columns([pl.Series(factor_name, factor_values)])
            except Exception as e:
                logger.error(f"Failed to compute factor {factor_name}: {e}")
                continue
        return result
    
    def get_factor_names(self) -> list[str]:
        return [f["name"] for f in self.factors]
    
    def get_feature_columns(self) -> list[str]:
        columns = self.get_factor_names()
        columns.extend(["rsi_14", "macd", "macd_signal", "macd_hist"])
        columns.extend(["volume_price_health", "volume_shrink_flag", "price_volume_divergence"])
        columns.extend(["vcp_score", "vcp_contraction", "price_amplitude", "amplitude_std"])
        columns.extend(["turnover_vol", "turnover_cv", "turnover_stable", "turnover_mean"])
        columns.extend(["smart_money_ratio", "smart_money_signal", "avg_up_volume", "avg_down_volume", "price_volume_correlation"])
        if self.label_config:
            columns.append(self.label_config["name"])
        columns.extend(["sharpe_label", "future_max_return", "future_volatility"])
        columns.extend(["predict_score", "filtered_score"])
        return columns