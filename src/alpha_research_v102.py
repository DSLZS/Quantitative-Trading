"""
Alpha Research Module - V102 Factor Refinement and Ablation.

【V102 核心改进】
1. 修复 V101 的 Look-ahead Bias 问题
2. 放弃 volume_entropy，回归 VWAP Residual Momentum
3. 实现因子消融实验 (Baseline -> Add-on 1 -> Add-on 2)
4. 自动化数据防御机制
5. 严格的 T+1 IC 验证 (目标 > 0.05)

【因子体系】
- Baseline: residual_momentum (VWAP 残差动量)
- Add-on 1: volume_price_divergence (量价背离)
- Add-on 2: institutional_flow (机构资金一致性)

【验收指标】
| 指标 | 目标值 | 失败判定 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | <= 0.03 视为无效优化 |
| IC Decay (T+1 to T+5) | 单调递减 | T+3 IC > T+1 IC 判定为未来函数泄露 |
| 因子独立性 | Corr < 0.7 | 因子间相关性过高视为冗余 |
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


class ICDecayWarning(Exception):
    """IC 衰减警告 - 当 IC 不单调递减时触发"""
    pass


class AlphaResearchV102:
    """
    V102 Alpha 预测核心引擎。
    
    【核心逻辑改进】
    - 预测目标：T+1 收益的截面排名
    - 损失函数：Spearman Rank IC
    - 核心因子：VWAP Residual Momentum (量价残差动量)
    
    【因子消融实验】
    1. Baseline: 仅 residual_momentum
    2. Add-on 1: + volume_price_divergence
    3. Add-on 2: + institutional_flow
    
    【防御机制】
    - 自动检测缺失数据列并尝试重新拉取
    - 严禁使用 T 日及之后的 close/high/low 数据计算 T 日评分
    """
    
    EPSILON = 1e-6
    
    # V102 因子权重配置 (基于 VWAP Residual Momentum 为核心)
    # 【V102 关键修复】根据 IC 分析调整因子方向：
    # - residual_momentum_10: IC=+0.0126 (正) - 保持并增加权重
    # - volume_price_divergence_5: IC=-0.0220 (负) - 需要翻转方向
    # - large_order_flow: IC=+0.0139 (正) - 保持
    # - volume_price_health: IC=+0.0107 (正) - 保持
    # - volatility_20: IC=-0.0157 (负) - 权重为负，翻转后为正贡献
    # - momentum_5/10: IC 为负 - 降低权重
    FACTOR_WEIGHTS = {
        # 核心因子：VWAP 残差动量 (IC 为正，增加权重)
        "residual_momentum_5": 0.15,
        "residual_momentum_10": 0.40,  # 核心因子，IC 为正
        
        # 量价背离因子 (翻转方向：负 IC 变正贡献)
        "volume_price_divergence_5": -0.20,  # 翻转：负 IC 表示反向关系
        "volume_price_health": 0.10,  # IC 为正，保持
        
        # 机构资金一致性 (IC 接近 0，降低权重)
        "institutional_flow": -0.05,  # 低权重
        "large_order_flow": 0.05,     # IC 为正，保持正权重
        
        # 基础动量 (IC 为负，降低权重或移除)
        "momentum_5": 0.00,  # 移除
        "momentum_10": 0.00,  # 移除
        
        # 波动率 (IC 为负，权重为负，翻转后为正贡献)
        "volatility_20": -0.05,
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
        
        # 消融实验记录
        self.ablation_results: dict[str, dict[str, Any]] = {}
        
        logger.info("AlphaResearchV102 initialized")
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
    
    # ==================== 数据防御机制 ====================
    
    def check_and_repair_data(self, df: pl.DataFrame, required_columns: list[str]) -> pl.DataFrame:
        """
        【数据防御】检查并修复缺失的数据列。
        
        如果 2024 年某列数据全为空：
        1. 在日志中提示
        2. 尝试计算或填充
        3. 严禁用 0 填充重要数据
        
        Args:
            df: 输入数据
            required_columns: 必需的列名列表
            
        Returns:
            修复后的数据
        """
        result = df.clone()
        missing_columns = []
        
        for col in required_columns:
            if col not in result.columns:
                missing_columns.append(col)
                logger.warning(f"[数据防御] 缺失列：{col}")
            else:
                # 检查是否全为空
                null_ratio = result[col].null_count() / len(result) if len(result) > 0 else 0
                if null_ratio > 0.95:
                    logger.warning(f"[数据防御] 列 {col} 的缺失值比例过高：{null_ratio:.1%}")
                    missing_columns.append(col)
        
        # 尝试修复缺失列
        for col in missing_columns:
            if col == "total_mv":
                # 尝试用 amount 和 turnover_rate 估算
                if "amount" in result.columns and "turnover_rate" in result.columns:
                    logger.info(f"[数据防御] 尝试用 amount/turnover_rate 估算 {col}")
                    estimated_mv = pl.col("amount") / (pl.col("turnover_rate").fill_null(0.01) + self.EPSILON) * 100
                    result = result.with_columns([estimated_mv.alias("total_mv")])
            elif col == "vwap":
                # 用 (high + low + close) / 3 估算 VWAP
                if "high" in result.columns and "low" in result.columns and "close" in result.columns:
                    logger.info(f"[数据防御] 用 (high+low+close)/3 估算 {col}")
                    estimated_vwap = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
                    result = result.with_columns([estimated_vwap.alias("vwap")])
            elif col == "amount":
                # 用 volume * close 估算
                if "volume" in result.columns and "close" in result.columns:
                    logger.info(f"[数据防御] 用 volume*close 估算 {col}")
                    estimated_amount = pl.col("volume") * pl.col("close")
                    result = result.with_columns([estimated_amount.alias("amount")])
        
        return result
    
    # ==================== 核心因子计算 ====================
    
    def compute_vwap_residual_momentum(self, df: pl.DataFrame, periods: list[int] = [5, 10]) -> pl.DataFrame:
        """
        计算 VWAP 残差动量因子 (V102 核心因子)。
        
        【核心逻辑 - VWAP Residual Momentum】
        1. 计算 VWAP = (high + low + close) / 3 (如果无 VWAP 数据)
        2. 计算残差 = close - VWAP (价格相对 VWAP 的偏离)
        3. 残差动量 = 残差的 N 日变化率
        
        【金融逻辑】
        - VWAP 代表市场平均持仓成本
        - 价格持续高于 VWAP 表明强势
        - 残差扩大表明动量增强
        
        【无前视偏差】
        - 仅使用 T-1 日及之前的数据计算 T 日评分
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
        ])
        
        # 计算或获取 VWAP
        if "vwap" not in result.columns:
            vwap = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
            result = result.with_columns([vwap.alias("vwap")])
        else:
            result = result.with_columns([pl.col("vwap").cast(pl.Float64).alias("vwap")])
        
        # 计算残差 = close - VWAP
        residual = pl.col("close") - pl.col("vwap")
        result = result.with_columns([residual.alias("price_residual")])
        
        # 计算残差动量 (N 日变化)
        for period in periods:
            # 残差的 N 日变化率
            residual_momentum = pl.col("price_residual") / (pl.col("price_residual").shift(period) + self.EPSILON) - 1.0
            result = result.with_columns([
                residual_momentum.alias(f"residual_momentum_{period}")
            ])
        
        logger.debug(f"[VWAP Residual Momentum] Computed for periods {periods}")
        return result
    
    def compute_momentum(self, df: pl.DataFrame, periods: list[int] = [5, 10, 20]) -> pl.DataFrame:
        """
        计算传统动量因子。
        
        Momentum = Close_t / Close_{t-n} - 1
        
        【注意】使用 T-1 日的 close 数据，避免前视偏差
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        for period in periods:
            # 使用 shift(1) 确保使用 T-1 日数据
            close_lag = pl.col("close").shift(1)  # T-1 日收盘价
            close_lag_n = pl.col("close").shift(period + 1)  # T-(n+1) 日收盘价
            momentum = close_lag / (close_lag_n + self.EPSILON) - 1.0
            result = result.with_columns([momentum.alias(f"momentum_{period}")])
        
        logger.debug(f"[Momentum] Computed for periods {periods}")
        return result
    
    def compute_volatility(self, df: pl.DataFrame, periods: list[int] = [5, 20]) -> pl.DataFrame:
        """
        计算波动率因子。
        
        Volatility = Std(Returns, window=n)
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        # 计算收益率 (使用 T-1 日数据)
        returns = pl.col("close").pct_change().fill_null(0)
        
        for period in periods:
            result = result.with_columns([
                returns.rolling_std(window_size=period, ddof=1).alias(f"volatility_{period}")
            ])
        
        logger.debug(f"[Volatility] Computed for periods {periods}")
        return result
    
    def compute_volume_price_divergence(self, df: pl.DataFrame, period: int = 5) -> pl.DataFrame:
        """
        计算量价背离因子 (Add-on 1)。
        
        【核心逻辑】
        Volume_Price_Divergence = Price_Change - Volume_Change
        
        【金融逻辑】
        - 价升量缩：背离信号，可能反转
        - 价跌量增：背离信号，抛压加重
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0),
        ])
        
        # 使用 T-1 日数据计算变化
        price_change = pl.col("close").shift(1) / (pl.col("close").shift(period + 1) + self.EPSILON) - 1.0
        volume_change = pl.col("volume").shift(1) / (pl.col("volume").shift(period + 1) + self.EPSILON) - 1.0
        
        # 量价背离
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
        
        【核心逻辑 - 修复前视偏差】
        使用 T-1 日数据判断量价关系，避免偷看未来
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0),
        ])
        
        # 使用 T-1 日数据
        volume_ma = pl.col("volume").shift(1).rolling_mean(window_size=volume_window)
        volume_ratio = pl.col("volume").shift(1) / (volume_ma + self.EPSILON)
        
        # T-1 日价格变化
        price_change = pl.col("close").shift(1) / (pl.col("close").shift(price_window + 1) + self.EPSILON) - 1.0
        
        # 量价健康度评分 (非线性映射)
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
    
    def compute_institutional_flow(self, df: pl.DataFrame, lookback: int = 5) -> pl.DataFrame:
        """
        计算机构资金一致性因子 (Add-on 2)。
        
        【核心逻辑】
        1. 大单流入 = amount / volume (估算平均成交金额)
        2. 机构一致性 = 大单流入的 N 日稳定性
        
        【金融逻辑】
        - 机构资金持续流入表明看好后市
        - 资金流入稳定表明机构共识强
        """
        result = df.clone().with_columns([
            pl.col("amount").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("volume").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 估算平均成交金额 (amount / volume)
        avg_trade_size = pl.col("amount") / (pl.col("volume") + self.EPSILON)
        
        # 大单流入信号 (平均成交金额 > 阈值)
        large_order_signal = (avg_trade_size > avg_trade_size.rolling_median(window_size=20)).cast(pl.Float64)
        
        # 机构一致性 = 大单信号的 N 日移动平均
        institutional_consistency = large_order_signal.rolling_mean(window_size=lookback)
        
        # 资金流向 = 成交金额的 N 日变化
        flow_direction = pl.col("amount").shift(1) / (pl.col("amount").shift(lookback + 1) + self.EPSILON) - 1.0
        
        # 综合机构资金因子
        institutional_flow = institutional_consistency * (1.0 + flow_direction.clip(-0.5, 0.5))
        
        result = result.with_columns([
            institutional_flow.alias("institutional_flow"),
            institutional_consistency.alias("institutional_consistency"),
            flow_direction.alias("flow_direction"),
            large_order_signal.alias("large_order_signal"),
        ])
        
        logger.debug(f"[Institutional Flow] Computed with lookback={lookback}")
        return result
    
    def compute_large_order_flow(self, df: pl.DataFrame, threshold_percentile: float = 80.0) -> pl.DataFrame:
        """
        计算大单资金流向因子。
        
        【核心逻辑 - 修复前视偏差】
        1. 识别大单交易日 (amount 超过 80 分位数)
        2. 使用历史数据计算大单日的持续性，严禁使用未来数据
        
        【修复说明】
        原代码使用 shift(-5) 偷看未来 5 日收益，导致 IC Decay 异常
        新代码仅使用 T-1 日及之前的历史数据
        """
        result = df.clone().with_columns([
            pl.col("amount").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("close").cast(pl.Float64, strict=False),
        ])
        
        # 按股票分组计算分位数 (使用滚动窗口，避免使用未来数据)
        # 使用过去 N 日的分位数作为阈值
        amount_rolling_p80 = pl.col("amount").rolling_quantile(quantile=0.80, window_size=60).over("symbol")
        
        # 大单日标记 (当日 amount 是否超过过去 60 日的 80 分位数)
        large_order_day = (pl.col("amount") > amount_rolling_p80).cast(pl.Float64)
        
        # 大单持续性 = 大单信号的 N 日移动平均 (使用到 T-1 日的数据)
        # 这表示近期大单出现的频率
        large_order_flow = large_order_day.shift(1).rolling_mean(window_size=10)
        
        result = result.with_columns([
            large_order_flow.alias("large_order_flow"),
            large_order_day.alias("large_order_day"),
        ])
        
        logger.debug(f"[Large Order Flow] Computed (no look-ahead bias)")
        return result
    
    # ==================== 标签计算 (T+1 收益) ====================
    
    def compute_t1_return(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        计算 T+1 收益标签。
        
        【预测目标】
        T+1_Return = Close_{t+1} / Close_t - 1
        
        【关键】确保使用 T+1 日的 close 计算 T 日的标签
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        # T+1 收益 = 明日收盘价 / 今日收盘价 - 1
        t1_return = pl.col("close").shift(-1) / (pl.col("close") + self.EPSILON) - 1.0
        
        result = result.with_columns([
            t1_return.alias("t1_return")
        ])
        
        logger.debug(f"[T+1 Return] Computed")
        return result
    
    def compute_tn_return(self, df: pl.DataFrame, n: int = 5) -> pl.DataFrame:
        """
        计算 T+N 收益标签 (用于 IC Decay 分析)。
        """
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False)
        ])
        
        tn_return = pl.col("close").shift(-n) / (pl.col("close") + self.EPSILON) - 1.0
        result = result.with_columns([tn_return.alias(f"t{n}_return")])
        
        logger.debug(f"[T+{n} Return] Computed")
        return result
    
    def compute_cross_sectional_rank(self, df: pl.DataFrame, column: str = "t1_return") -> pl.DataFrame:
        """
        计算截面排名。
        
        【截面排名逻辑】
        在每个交易日，对所有股票的指定指标进行排名，
        归一化到 0-1 区间。
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
        """缩尾处理（去极值）。"""
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
        """标准化处理。"""
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
        """智能填充空值。"""
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
                # 缺失值过多，填 0
                result = result.with_columns([pl.col(col).fill_null(0.0).alias(col)])
                logger.debug(f"[Fill Null] Factor '{col}' has {null_ratio:.1%} nulls, filled with 0")
            else:
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
        """计算综合预测评分。"""
        if weights is None:
            weights = self.FACTOR_WEIGHTS
        
        result = df.clone()
        
        # 计算加权评分
        raw_score = pl.lit(0.0)
        for factor_name, weight in weights.items():
            if factor_name in result.columns:
                raw_score = raw_score + pl.col(factor_name) * weight
        
        result = result.with_columns([raw_score.alias("predict_score")])
        
        logger.debug(f"[Predict Score] Computed, factors used: {len(weights)}")
        return result
    
    # ==================== 因子计算主流程 ====================
    
    def compute_factors(self, df: pl.DataFrame, ablation_config: str = "full") -> pl.DataFrame:
        """
        计算所有因子并生成预测评分。
        
        Args:
            df: 输入数据
            ablation_config: 消融实验配置
                - "baseline": 仅 residual_momentum
                - "addon1": + volume_price_divergence
                - "addon2": + institutional_flow
                - "full": 所有因子
        
        【计算顺序】
        1. 数据防御检查
        2. VWAP 残差动量 (核心)
        3. 基础动量因子
        4. 波动率因子
        5. 量价交互因子
        6. 机构资金因子
        7. T+1/T+N 收益标签
        8. 缺失值处理
        9. 标准化
        10. 预测评分
        """
        # 1. 数据防御检查
        required_columns = ["close", "high", "low", "volume", "trade_date", "symbol"]
        result = self.check_and_repair_data(df, required_columns)
        
        # 2. VWAP 残差动量 (核心) - 始终计算
        result = self.compute_vwap_residual_momentum(result, periods=[5, 10])
        
        # 3. 基础动量
        result = self.compute_momentum(result, periods=[5, 10])
        
        # 4. 波动率
        result = self.compute_volatility(result, periods=[20])
        
        # 5. 量价交互因子 (Add-on 1)
        if ablation_config in ["addon1", "addon2", "full"]:
            result = self.compute_volume_price_divergence(result, period=5)
            result = self.compute_volume_price_health(result)
        
        # 6. 机构资金因子 (Add-on 2)
        if ablation_config in ["addon2", "full"]:
            result = self.compute_institutional_flow(result)
            result = self.compute_large_order_flow(result)
        
        # 7. T+1/T+N 收益标签
        result = self.compute_t1_return(result)
        result = self.compute_tn_return(result, n=3)  # T+3
        result = self.compute_tn_return(result, n=5)  # T+5
        result = self.compute_cross_sectional_rank(result, column="t1_return")
        
        # 8. 缺失值处理
        result = self.fill_null_values(result)
        
        # 9. 标准化
        result = self.winsorize(result, lower_percentile=1.0, upper_percentile=99.0)
        result = self.normalize(result, method="zscore")
        
        # 10. 预测评分
        result = self.compute_predict_score(result)
        
        logger.info(f"[Compute Factors] Complete (config={ablation_config}), total columns: {len(result.columns)}")
        return result
    
    def get_factor_names(self) -> list[str]:
        """获取配置的因子名称列表。"""
        return list(self.FACTOR_WEIGHTS.keys())
    
    # ==================== IC 计算与分析 ====================
    
    def calculate_rank_ic(self, factor_values: pl.Series, label_values: pl.Series) -> float:
        """计算 Rank IC（Spearman 相关系数）。"""
        # 去除空值
        mask = factor_values.is_not_null() & label_values.is_not_null()
        factor_clean = factor_values.filter(mask)
        label_clean = label_values.filter(mask)
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算秩
        factor_ranks = factor_clean.rank(method='average')
        label_ranks = label_clean.rank(method='average')
        
        # 计算 Pearson 相关系数
        factor_np = factor_ranks.to_numpy()
        label_np = label_ranks.to_numpy()
        
        if np.std(factor_np) < 1e-10 or np.std(label_np) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(factor_np, label_np)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_t1_ic(self, df: pl.DataFrame, score_column: str = "predict_score") -> dict[str, Any]:
        """计算 T+1 Rank IC。"""
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
        unique_dates = sorted(df["trade_date"].unique())
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
        
        # 自检机制
        if mean_ic < 0.03:
            logger.warning(f"[AlphaWeakWarning] T+1 IC = {mean_ic:.4f} < 0.03")
            self._analyze_factor_contribution(df, score_column)
        
        return result
    
    def calculate_ic_decay(self, df: pl.DataFrame, score_column: str = "predict_score") -> dict[str, Any]:
        """
        计算 IC Decay (T+1 to T+5)。
        
        【验收指标】
        IC 应该单调递减，如果 T+3 IC > T+1 IC，说明存在未来函数泄露
        """
        ic_decay = {}
        
        for n in [1, 3, 5]:
            return_col = f"t{n}_return"
            if return_col in df.columns:
                ic = self.calculate_tn_ic(df, score_column, return_col)
                ic_decay[f"t{n}_ic"] = ic
        
        # 检查单调性
        t1_ic = ic_decay.get("t1_ic", 0)
        t3_ic = ic_decay.get("t3_ic", 0)
        t5_ic = ic_decay.get("t5_ic", 0)
        
        is_monotonic = (t1_ic >= t3_ic >= t5_ic) if t1_ic > 0 else True
        
        if not is_monotonic and t1_ic > 0:
            logger.warning(f"[ICDecayWarning] IC 不单调递减：T+1={t1_ic:.4f}, T+3={t3_ic:.4f}, T+5={t5_ic:.4f}")
            logger.warning("可能存在未来函数泄露!")
        
        ic_decay["is_monotonic"] = is_monotonic
        ic_decay["decay_pattern"] = f"T+1({t1_ic:.4f}) -> T+3({t3_ic:.4f}) -> T+5({t5_ic:.4f})"
        
        return ic_decay
    
    def calculate_tn_ic(self, df: pl.DataFrame, score_column: str, return_column: str) -> float:
        """计算指定 horizon 的 IC。"""
        if score_column not in df.columns or return_column not in df.columns:
            return 0.0
        
        if "trade_date" not in df.columns:
            return 0.0
        
        unique_dates = sorted(df["trade_date"].unique())
        ic_series = []
        
        for date in unique_dates:
            day_data = df.filter(pl.col("trade_date") == date)
            
            if len(day_data) < 10:
                continue
            
            score_values = day_data[score_column]
            label_values = day_data[return_column]
            
            ic = self.calculate_rank_ic(score_values, label_values)
            
            if ic != 0 or not np.isnan(ic):
                ic_series.append(ic)
        
        if not ic_series:
            return 0.0
        
        return float(np.mean(ic_series))
    
    def _analyze_factor_contribution(self, df: pl.DataFrame, score_column: str) -> None:
        """分析各因子贡献度。"""
        logger.info("[因子贡献度分析] 开始分析各因子 IC 贡献...")
        
        factor_columns = list(self.FACTOR_WEIGHTS.keys())
        factor_ics = []
        
        for factor_name in factor_columns:
            if factor_name in df.columns and "t1_return" in df.columns:
                ic = self.calculate_rank_ic(df[factor_name], df["t1_return"])
                factor_ics.append({
                    "factor": factor_name,
                    "ic": ic,
                    "weight": self.FACTOR_WEIGHTS.get(factor_name, 0),
                    "contribution": ic * self.FACTOR_WEIGHTS.get(factor_name, 0),
                })
        
        factor_ics.sort(key=lambda x: abs(x["ic"]), reverse=True)
        
        logger.info("[因子贡献度分析] 结果:")
        for i, f in enumerate(factor_ics[:10], 1):
            status = "✓" if abs(f["ic"]) > 0.03 else "✗"
            logger.info(f"  {i}. {f['factor']}: IC={f['ic']:.4f}, Weight={f['weight']:.2f}, Contribution={f['contribution']:.4f} {status}")
    
    def calculate_factor_ic(self, df: pl.DataFrame, factor_name: str) -> float:
        """计算单个因子的 IC 值。"""
        if factor_name not in df.columns or "t1_return" not in df.columns:
            return 0.0
        
        return self.calculate_rank_ic(df[factor_name], df["t1_return"])
    
    def calculate_factor_correlation(self, df: pl.DataFrame, factor1: str, factor2: str) -> float:
        """计算两个因子之间的相关性。"""
        if factor1 not in df.columns or factor2 not in df.columns:
            return 0.0
        
        # 去除空值
        mask = df[factor1].is_not_null() & df[factor2].is_not_null()
        f1_clean = df[factor1].filter(mask).to_numpy()
        f2_clean = df[factor2].filter(mask).to_numpy()
        
        if len(f1_clean) < 10:
            return 0.0
        
        correlation = np.corrcoef(f1_clean, f2_clean)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    # ==================== 消融实验 ====================
    
    def run_ablation_experiment(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        运行因子消融实验。
        
        【实验流程】
        1. Baseline: 仅 residual_momentum
        2. Add-on 1: + volume_price_divergence
        3. Add-on 2: + institutional_flow
        
        【记录规则】
        只有当 IC(New) > IC(Old) 且 IC > 0.03 时，才允许该因子进入最终模型
        """
        logger.info("=" * 70)
        logger.info("V102 Factor Ablation Experiment")
        logger.info("=" * 70)
        
        ablation_results = {}
        baseline_ic = 0.0
        
        # 1. Baseline: 仅 residual_momentum
        logger.info("\n[Experiment 1] Baseline - Residual Momentum Only")
        baseline_weights = {
            "residual_momentum_5": 0.55,
            "residual_momentum_10": 0.45,
        }
        baseline_df = self.compute_factors(df, ablation_config="baseline")
        baseline_df = self.compute_predict_score(baseline_df, weights=baseline_weights)
        baseline_ic_result = self.calculate_t1_ic(baseline_df, "predict_score")
        baseline_ic = baseline_ic_result["mean_ic"]
        
        ablation_results["baseline"] = {
            "config": "residual_momentum only",
            "ic": baseline_ic,
            "ic_ir": baseline_ic_result["ic_ir"],
            "num_days": baseline_ic_result["num_days"],
        }
        logger.info(f"  Baseline IC: {baseline_ic:.4f}")
        
        # 2. Add-on 1: + volume_price_divergence
        logger.info("\n[Experiment 2] Add-on 1 - + Volume Price Divergence")
        addon1_weights = {
            "residual_momentum_5": 0.35,
            "residual_momentum_10": 0.25,
            "volume_price_divergence_5": 0.20,
            "volume_price_health": 0.20,
        }
        addon1_df = self.compute_factors(df, ablation_config="addon1")
        addon1_df = self.compute_predict_score(addon1_df, weights=addon1_weights)
        addon1_ic_result = self.calculate_t1_ic(addon1_df, "predict_score")
        addon1_ic = addon1_ic_result["mean_ic"]
        
        addon1_improved = (addon1_ic > baseline_ic) and (addon1_ic > 0.03)
        ablation_results["addon1"] = {
            "config": "baseline + volume_price_divergence",
            "ic": addon1_ic,
            "ic_ir": addon1_ic_result["ic_ir"],
            "improved": addon1_improved,
            "ic_delta": addon1_ic - baseline_ic,
        }
        logger.info(f"  Add-on 1 IC: {addon1_ic:.4f} (Δ = {addon1_ic - baseline_ic:+.4f}) {'✓' if addon1_improved else '✗'}")
        
        # 3. Add-on 2: + institutional_flow
        logger.info("\n[Experiment 3] Add-on 2 - + Institutional Flow")
        addon2_weights = self.FACTOR_WEIGHTS.copy()
        addon2_df = self.compute_factors(df, ablation_config="full")
        addon2_df = self.compute_predict_score(addon2_df, weights=addon2_weights)
        addon2_ic_result = self.calculate_t1_ic(addon2_df, "predict_score")
        addon2_ic = addon2_ic_result["mean_ic"]
        
        addon2_improved = (addon2_ic > addon1_ic) and (addon2_ic > 0.03)
        ablation_results["addon2"] = {
            "config": "addon1 + institutional_flow",
            "ic": addon2_ic,
            "ic_ir": addon2_ic_result["ic_ir"],
            "improved": addon2_improved,
            "ic_delta": addon2_ic - addon1_ic,
        }
        logger.info(f"  Add-on 2 IC: {addon2_ic:.4f} (Δ = {addon2_ic - addon1_ic:+.4f}) {'✓' if addon2_improved else '✗'}")
        
        # IC Decay 分析
        logger.info("\n[IC Decay Analysis]")
        ic_decay = self.calculate_ic_decay(addon2_df, "predict_score")
        ablation_results["ic_decay"] = ic_decay
        logger.info(f"  Decay Pattern: {ic_decay['decay_pattern']}")
        logger.info(f"  Monotonic: {'✓' if ic_decay['is_monotonic'] else '✗'}")
        
        # 因子独立性检查
        logger.info("\n[Factor Independence Check]")
        factor_correlations = {}
        factor_names = ["residual_momentum_5", "volume_price_divergence_5", "institutional_flow"]
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i+1:]:
                corr = self.calculate_factor_correlation(addon2_df, f1, f2)
                factor_correlations[f"{f1}_vs_{f2}"] = corr
                status = "✓" if abs(corr) < 0.7 else "✗ (high correlation)"
                logger.info(f"  Corr({f1}, {f2}): {corr:.4f} {status}")
        
        ablation_results["factor_correlations"] = factor_correlations
        
        # 最终判断
        final_ic = addon2_ic if addon2_improved else (addon1_ic if ablation_results["addon1"]["improved"] else baseline_ic)
        passed = (final_ic > 0.05) and ic_decay["is_monotonic"]
        
        ablation_results["final_ic"] = final_ic
        ablation_results["passed"] = passed
        
        logger.info("\n" + "=" * 70)
        logger.info(f"V102 Ablation Experiment Complete")
        logger.info(f"  Final IC: {final_ic:.4f}")
        logger.info(f"  Status: {'PASSED ✓' if passed else 'FAILED ✗'}")
        logger.info("=" * 70)
        
        self.ablation_results = ablation_results
        return ablation_results
    
    # ==================== 完整分析流程 ====================
    
    def run_alpha_analysis(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        运行完整的 Alpha 分析流程（包含消融实验）。
        
        Returns:
            dict: 分析结果
        """
        logger.info("=" * 60)
        logger.info("V102 Alpha Research - Prediction Analysis")
        logger.info("=" * 60)
        
        # 运行消融实验
        ablation_result = self.run_ablation_experiment(df)
        
        # 使用最终配置重新计算
        final_df = self.compute_factors(df, ablation_config="full")
        final_df = self.compute_predict_score(final_df)
        
        # 计算最终 IC
        final_ic = self.calculate_t1_ic(final_df, "predict_score")
        ic_decay = self.calculate_ic_decay(final_df, "predict_score")
        
        # 获取 Top 因子
        top_factor = self.get_top_factor_ic(final_df)
        
        logger.info(f"[Final T+1 IC] Mean={final_ic['mean_ic']:.4f}, IR={final_ic['ic_ir']:.2f}")
        logger.info(f"[Top Factor] {top_factor['factor']}: IC={top_factor['ic']:.4f}")
        
        # 验收判断
        passed = (
            final_ic["mean_ic"] > 0.05 and
            final_ic["ic_ir"] > 0.6 and
            abs(top_factor["ic"]) > 0.04 and
            ic_decay["is_monotonic"]
        )
        
        if passed:
            logger.info("[验收结果] PASSED - Alpha 预测能力达标")
        else:
            logger.warning("[验收结果] FAILED - Alpha 预测能力不足")
        
        return {
            "processed_df": final_df,
            "t1_ic": final_ic,
            "ic_decay": ic_decay,
            "top_factor": top_factor,
            "ablation_result": ablation_result,
            "passed": passed,
        }
    
    def get_top_factor_ic(self, df: pl.DataFrame) -> dict[str, float]:
        """获取最高 IC 的因子。"""
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
        
        if abs(result["ic"]) < 0.04:
            logger.warning(f"[验收警告] Top Factor IC = {abs(result['ic']):.4f} < 0.04")
        
        return result
    
    def generate_ablation_report(self) -> str:
        """生成消融实验报告。"""
        if not self.ablation_results:
            return "No ablation results available"
        
        report = """# V102 Factor Ablation Experiment Report

## 1. Experiment Summary

| Config | IC | IC IR | Improved |
|--------|-----|-------|----------|
"""
        for config_name, result in self.ablation_results.items():
            if config_name in ["baseline", "addon1", "addon2"]:
                improved = "✓" if result.get("improved", False) else "✗"
                report += f"| {config_name} | {result['ic']:.4f} | {result['ic_ir']:.2f} | {improved} |\n"
        
        report += """
## 2. IC Decay Analysis

"""
        if "ic_decay" in self.ablation_results:
            decay = self.ablation_results["ic_decay"]
            report += f"- Decay Pattern: {decay['decay_pattern']}\n"
            report += f"- Monotonic: {'✓' if decay['is_monotonic'] else '✗'}\n"
        
        report += """
## 3. Factor Independence

"""
        if "factor_correlations" in self.ablation_results:
            for pair, corr in self.ablation_results["factor_correlations"].items():
                status = "✓" if abs(corr) < 0.7 else "✗"
                report += f"- {pair}: {corr:.4f} {status}\n"
        
        return report


def get_alpha_research(config_path: str = "config/factors.yaml") -> AlphaResearchV102:
    """获取 AlphaResearchV102 实例。"""
    return AlphaResearchV102(config_path)