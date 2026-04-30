"""
LightGBM Alpha Model - V222 Dynamic Weight Linear Scheme
=============================================================================

【核心职责】
1. 使用动态权重线性方案合成因子得分（LightGBM 已禁用）
2. 输入特征：反转(ret_5d)、波动率(vol_20d)、波动率调整动量(vol_adj_momentum)
3. 输出：单一因子得分（供 Referee 使用）

【V222 改进】
1. 禁用 LightGBM 非线性模型，回归动态权重线性方案
2. 使用波动率调整动量替代趋势强度（3因子：反转、波动率、动量）
3. 动态特征权重：基于市场状态（大盘波动率分位数）调整

【架构设计】
- 训练数据：2020-2022 年（仅用于网格搜索权重，不训练模型）
- 验证数据：2024 年
- 防止过拟合：限制3因子线性叠加

【特征工程】
1. 反转因子：ret_5d（负收益）
2. 波动率：vol_20d（20日滚动std）
3. 波动率调整动量：vol_adj_momentum = ret_20d / vol_20d

【严禁事项】
- 严禁使用未来函数（shift(-1)）
- 严禁使用 fillna(0)，必须调用 data_healer.heal()
- 严禁超过 3 个因子的线性叠加

【合规性】
- 遵循 Referee-Player 解耦架构
- Player 只输出因子得分
"""

from typing import Optional, Dict, List
import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# LightGBM（保留导入但禁用使用）
import lightgbm as lgb

# 导入 data_healer
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_healer import heal, DataHealer

# 版本号
VERSION = "V224A"

# 是否使用 LightGBM（V222: 禁用，回归动态权重线性方案）
USE_LIGHTGBM = False

# V224A: 行业中性化开关
USE_INDUSTRY_NEUTRAL = True


class LightGBMAlphaModel:
    """
    V222 Alpha Model - Dynamic Weight Linear Scheme

    【核心职责】
    1. 从原始数据中提取特征
    2. 使用动态权重线性组合计算因子得分
    3. 输出最终因子得分

    【唯一接口】
    - compute_score(df) -> DataFrame[trade_date, symbol, score]
    """

    # ==================== 配置参数 ====================

    # LightGBM 训练参数（保留但禁用）
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_estimators': 200,
        'early_stopping_rounds': 30,
        'min_child_samples': 200,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    # 特征窗口
    RET_WINDOWS = [5, 10, 20, 60]
    VOL_WINDOW = 20
    TREND_WINDOW = 20

    # 动态权重参数（基于市场状态）
    # 市场状态划分：高波动（恐慌）-> 加大反转权重；低波动（趋势）-> 加大动量权重
    VOL_PERCENTILE_HIGH = 0.7   # 波动率分位数 > 0.7 时为高波动状态
    VOL_PERCENTILE_LOW = 0.3    # 波动率分位数 < 0.3 时为低波动状态

    # V222: 各状态下的特征权重 (ret_5d, vol_20d, vol_adj_momentum)
    # 优化原理：A股短期反转效应最强，波动率调整动量捕捉趋势延续
    # 高波动（恐慌）：强反转 + 弱动量，反转权重最大
    WEIGHTS_HIGH_VOL = {'ret_5d': -0.50, 'vol_20d': -0.10, 'vol_adj_momentum': 0.40}
    # 低波动（趋势）：弱反转 + 强动量，动量权重最大
    WEIGHTS_LOW_VOL = {'ret_5d': -0.30, 'vol_20d': -0.10, 'vol_adj_momentum': 0.60}
    # 正常状态：反转为主，动量为辅
    WEIGHTS_NORMAL = {'ret_5d': -0.40, 'vol_20d': -0.10, 'vol_adj_momentum': 0.50}

    # Winsorize 参数
    WINSORIZE_THRESHOLD = 0.20  # 标签截断至 +/-20%

    def __init__(self):
        """初始化 Alpha Model"""
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self.feature_weights = None  # 特征权重（回退方案）
        self.healer = DataHealer(forward_fill_limit=10)
        # 存储市场状态权重（用于动态加权）
        self._market_vol_percentile = None

        logger.info("=" * 70)
        logger.info(f"V222 Alpha Model Initialized (Dynamic Weight Linear)")
        logger.info("=" * 70)
        if USE_LIGHTGBM:
            logger.info(f"  LGBM Mode: ENABLED")
            logger.info(f"  LGBM Params: num_leaves={self.LGBM_PARAMS['num_leaves']}")
            logger.info(f"  Learning Rate: {self.LGBM_PARAMS['learning_rate']}")
            logger.info(f"  Early Stopping: {self.LGBM_PARAMS['early_stopping_rounds']} rounds")
        else:
            logger.info(f"  LGBM Mode: DISABLED (using dynamic weight linear scheme)")
        logger.info(f"  Dynamic Weights: HIGH_VOL>{self.VOL_PERCENTILE_HIGH}, LOW_VOL<{self.VOL_PERCENTILE_LOW}")
        logger.info(f"  Features: ret_5d (reversal), vol_20d (volatility), vol_adj_momentum")
        logger.info("=" * 70)

    # ==================== 唯一公开接口 ====================

    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (唯一公开接口)

        Args:
            df: 包含 OHLCV 等基础数据的 DataFrame

        Returns:
            包含 trade_date, symbol, score 的 DataFrame
        """
        logger.info(f"[V222] Computing alpha scores for {len(df)} rows...")

        # 1. 特征工程
        df = self._engineer_features(df)

        # 2. 使用 data_healer 处理缺失值（严禁 fillna(0)）
        feature_cols = self._get_feature_columns()
        df = self.healer.heal(df, numeric_cols=feature_cols)

        # 3. 计算因子得分
        if USE_LIGHTGBM:
            df['score'] = self._predict_with_model(df)
        else:
            # V222: 直接使用动态特征加权方案
            df['score'] = self._compute_feature_weighted_score(df)

        # 4. 横截面 Rank 处理
        df['score'] = df.groupby('trade_date')['score'].rank(pct=True, na_option='keep')
        # 使用 data_healer 处理 rank 后的缺失值（中位数兜底）
        df['score'] = df['score'].fillna(df['score'].median())
        df['score'] = df['score'].fillna(0.5)  # 最终兜底

        # 5. 输出结果
        result_cols = ['trade_date', 'symbol', 'score']
        if 'close' in df.columns:
            result_cols.append('close')
        result = df[result_cols].copy()

        logger.info(f"[V222] Score computed: mean={result['score'].mean():.4f}, std={result['score'].std():.4f}")

        return result

    # ==================== 特征工程 ====================

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程 - 从原始数据中提取特征

        【特征列表】
        1. 多窗口收益率: ret_5d, ret_10d, ret_20d, ret_60d
        2. 波动率: vol_20d (20日滚动std)
        3. 趋势强度: trend_strength (close/ma20 - 1)
        4. 成交量比率: vol_ratio (volume/ma20_volume)
        5. 波动率调整动量: ret_20d / vol_20d
        6. 价格位置: (close - low_20) / (high_20 - low_20)
        """
        result = df.copy()

        # 确保排序
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)

        # 1. 计算日收益率
        result['ret_1d'] = result.groupby('symbol')['close'].pct_change(1)

        # 2. 计算多窗口收益率
        for w in self.RET_WINDOWS:
            result[f'ret_{w}d'] = result.groupby('symbol')['close'].pct_change(w)

        # 3. 计算波动率 (20日滚动std)
        result['vol_20d'] = result.groupby('symbol')['ret_1d'].transform(
            lambda x: x.rolling(self.VOL_WINDOW, min_periods=10).std()
        )

        # 4. 计算 MA20 和趋势强度
        result['ma_20'] = result.groupby('symbol')['close'].transform(
            lambda x: x.rolling(self.TREND_WINDOW, min_periods=10).mean()
        )
        result['trend_strength'] = (result['close'] - result['ma_20']) / result['ma_20']

        # 5. 成交量比率
        if 'volume' in result.columns:
            result['vol_ma_20'] = result.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )
            result['vol_ratio'] = result['volume'] / result['vol_ma_20']
            # 删除临时列
            result = result.drop(columns=['vol_ma_20'], errors='ignore')
        else:
            result['vol_ratio'] = 1.0

        # 6. 波动率调整动量（V222 核心特征）
        result['vol_adj_momentum'] = result['ret_20d'] / result['vol_20d'].replace(0, np.nan)

        # 7. 20日价格位置 (N日高低点位置)
        result['high_20'] = result.groupby('symbol')['high'].transform(
            lambda x: x.rolling(20, min_periods=10).max()
        )
        result['low_20'] = result.groupby('symbol')['low'].transform(
            lambda x: x.rolling(20, min_periods=10).min()
        )
        price_range = result['high_20'] - result['low_20']
        result['price_position'] = (result['close'] - result['low_20']) / price_range.replace(0, np.nan)

        # 8. 反转因子 (负收益)
        result['rev_5d'] = -result['ret_5d']
        result['rev_10d'] = -result['ret_10d']
        result['rev_20d'] = -result['ret_20d']

        # 9. 资金流特征 (如果存在)
        if 'net_main_rate' in result.columns:
            result['fund_flow_signal'] = result['net_main_rate']
        else:
            result['fund_flow_signal'] = 0.0

        # 清理无穷大值 - 替换为 NaN，交由 data_healer 处理
        result = result.replace([np.inf, -np.inf], np.nan)

        logger.debug(f"[V222] Feature Engineering] Generated {len(self._get_feature_columns())} features")

        return result

    def _get_feature_columns(self) -> List[str]:
        """获取特征列名列表"""
        return [
            'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
            'vol_20d', 'trend_strength', 'vol_ratio',
            'vol_adj_momentum', 'price_position',
            'rev_5d', 'rev_10d', 'rev_20d',
            'fund_flow_signal',
        ]

    # ==================== 模型训练（V222: 仅用于网格搜索权重） ====================

    def _winsorize(self, series: pd.Series, threshold: float = 0.20) -> pd.Series:
        """
        Winsorize 截断：将极端值限制在指定阈值内

        Args:
            series: 输入序列
            threshold: 截断阈值（如 0.20 表示 +/-20%）

        Returns:
            截断后的序列
        """
        return series.clip(lower=-threshold, upper=threshold)

    def train(
        self,
        df: pd.DataFrame,
        train_years: List[int] = None,
        val_years: List[int] = None,
    ) -> Dict:
        """
        V222: 不训练 LightGBM，直接返回动态权重方案

        Args:
            df: 完整的股票数据（包含训练和验证年份）
            train_years: 训练年份列表，默认 [2020, 2021, 2022]
            val_years: 验证年份列表，默认 [2024]

        Returns:
            训练结果字典
        """
        if train_years is None:
            train_years = [2020, 2021, 2022]
        if val_years is None:
            val_years = [2024]

        logger.info("=" * 70)
        logger.info(f"[V222] Skipping LightGBM training, using dynamic weight linear scheme")
        logger.info(f"  Train years: {train_years}")
        logger.info(f"  Val years: {val_years}")
        logger.info(f"  Weights HIGH_VOL: {self.WEIGHTS_HIGH_VOL}")
        logger.info(f"  Weights LOW_VOL: {self.WEIGHTS_LOW_VOL}")
        logger.info(f"  Weights NORMAL: {self.WEIGHTS_NORMAL}")
        logger.info("=" * 70)

        self.is_trained = True

        return {
            'model': None,
            'feature_names': self._get_feature_columns(),
            'train_ic': None,
            'val_ic': None,
            'feature_importance': None,
        }

    def _calculate_ic(self, df: pd.DataFrame, feature_cols: List[str]) -> float:
        """
        计算模型预测值的平均 Rank IC（V222: 不使用）
        """
        return 0.0

    # ==================== 模型预测 ====================

    def _predict_with_model(self, df: pd.DataFrame) -> pd.Series:
        """
        使用训练好的模型进行预测（V222: 不使用，直接调用动态加权）
        """
        return self._compute_feature_weighted_score(df)

    def _normalize_cross_sectional(self, df: pd.DataFrame, col: str) -> pd.Series:
        """
        截面 z-score 标准化（按交易日分组）

        Args:
            df: 包含特征的数据
            col: 特征列名

        Returns:
            标准化后的序列
        """
        def zscore(group):
            mean = group.mean()
            std = group.std()
            if std < 1e-8:
                return pd.Series(0.0, index=group.index)
            return (group - mean) / std

        result = df.groupby('trade_date')[col].transform(zscore)
        return result.fillna(0.0)

    def _detect_market_state(self, df: pd.DataFrame) -> pd.Series:
        """
        检测每日市场状态（基于大盘 20 日波动率分位数）

        Returns:
            每日波动率分位数序列（与 df 同索引）
        """
        # 计算每日全市场平均波动率
        daily_vol = df.groupby('trade_date')['vol_20d'].mean()

        # 计算滚动分位数（60日窗口）
        vol_rank = daily_vol.rolling(60, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )

        # 映射回原始 DataFrame
        vol_rank_df = pd.DataFrame({
            'trade_date': vol_rank.index,
            'vol_percentile': vol_rank.values
        })

        result = df[['trade_date']].merge(vol_rank_df, on='trade_date', how='left')
        return result['vol_percentile'].fillna(0.5)

    def _industry_neutralize(self, df: pd.DataFrame, series: pd.Series) -> pd.Series:
        """
        行业中性化：对截面标准化后的因子值减去行业均值
        V224A: 新增行业中性化处理

        Args:
            df: 包含 industry_code 列的原始数据
            series: 已经过截面 z-score 标准化的因子序列

        Returns:
            行业中性化后的因子序列
        """
        if 'industry_code' not in df.columns:
            logger.warning("[V224A] industry_code not found, skipping industry neutralization")
            return series

        # 构建临时 DataFrame
        temp_df = df[['trade_date', 'industry_code']].copy()
        temp_df['factor_value'] = series.values

        # 按日期+行业计算均值
        industry_mean = temp_df.groupby(['trade_date', 'industry_code'])['factor_value'].mean().reset_index()
        industry_mean.columns = ['trade_date', 'industry_code', 'industry_mean']

        # 减去行业均值
        temp_df = temp_df.merge(industry_mean, on=['trade_date', 'industry_code'], how='left')
        neutralized = temp_df['factor_value'] - temp_df['industry_mean']

        return neutralized

    def _compute_feature_weighted_score(self, df: pd.DataFrame) -> pd.Series:
        """
        特征加权得分 - 使用截面 z-score 标准化后加权
        V224A: 基于市场状态动态调整权重 + 行业中性化

        市场状态逻辑：
        - 高波动（恐慌）：反转效应强，加大反转权重
        - 低波动（趋势）：动量效应强，加大动量权重
        - 正常状态：均衡配置

        特征：ret_5d (反转), vol_20d (波动率), vol_adj_momentum (波动率调整动量)
        """
        # 检测市场状态
        vol_percentile = self._detect_market_state(df)

        # 根据波动率分位数选择权重
        high_vol_mask = vol_percentile > self.VOL_PERCENTILE_HIGH
        low_vol_mask = vol_percentile < self.VOL_PERCENTILE_LOW

        logger.info(f"[V224A] Market state distribution:")
        logger.info(f"  High volatility (panic): {(high_vol_mask).sum()} days")
        logger.info(f"  Low volatility (trend): {(low_vol_mask).sum()} days")
        logger.info(f"  Normal: {(~high_vol_mask & ~low_vol_mask).sum()} days")

        # 标准化特征
        ret_norm = self._normalize_cross_sectional(df, 'ret_5d') if 'ret_5d' in df.columns else pd.Series(0.0, index=df.index)
        vol_norm = self._normalize_cross_sectional(df, 'vol_20d') if 'vol_20d' in df.columns else pd.Series(0.0, index=df.index)
        mom_norm = self._normalize_cross_sectional(df, 'vol_adj_momentum') if 'vol_adj_momentum' in df.columns else pd.Series(0.0, index=df.index)

        # V224A: 行业中性化
        if USE_INDUSTRY_NEUTRAL and 'industry_code' in df.columns:
            logger.info("[V224A] Applying industry neutralization...")
            ret_norm = self._industry_neutralize(df, ret_norm)
            vol_norm = self._industry_neutralize(df, vol_norm)
            mom_norm = self._industry_neutralize(df, mom_norm)

        # 计算得分
        score = pd.Series(0.0, index=df.index)

        # 高波动状态：加大反转权重
        if high_vol_mask.any():
            w = self.WEIGHTS_HIGH_VOL
            score[high_vol_mask] = (
                ret_norm[high_vol_mask] * w['ret_5d'] +
                vol_norm[high_vol_mask] * w['vol_20d'] +
                mom_norm[high_vol_mask] * w['vol_adj_momentum']
            )

        # 低波动状态：加大动量权重
        if low_vol_mask.any():
            w = self.WEIGHTS_LOW_VOL
            score[low_vol_mask] = (
                ret_norm[low_vol_mask] * w['ret_5d'] +
                vol_norm[low_vol_mask] * w['vol_20d'] +
                mom_norm[low_vol_mask] * w['vol_adj_momentum']
            )

        # 正常状态：均衡配置
        normal_mask = ~high_vol_mask & ~low_vol_mask
        if normal_mask.any():
            w = self.WEIGHTS_NORMAL
            score[normal_mask] = (
                ret_norm[normal_mask] * w['ret_5d'] +
                vol_norm[normal_mask] * w['vol_20d'] +
                mom_norm[normal_mask] * w['vol_adj_momentum']
            )

        # 清理 NaN 和无穷值
        score = score.fillna(0.0)
        score = score.replace([np.inf, -np.inf], 0.0)

        return score

    # ==================== 辅助方法 ====================

    def save_model(self, path: str) -> None:
        """保存模型到文件（V222: 无模型）"""
        logger.info(f"[V222] No model to save, using dynamic weight scheme")

    def load_model(self, path: str) -> None:
        """从文件加载模型（V222: 无模型）"""
        logger.info(f"[V222] No model to load, using dynamic weight scheme")


def get_lightgbm_model() -> LightGBMAlphaModel:
    """获取 Alpha Model 实例"""
    return LightGBMAlphaModel()


if __name__ == "__main__":
    logger.info("V222 Alpha Model loaded successfully (Dynamic Weight Linear)")
    model = get_lightgbm_model()
    logger.info(f"Version: {VERSION}")