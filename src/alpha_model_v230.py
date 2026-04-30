"""
Alpha Model V230 - LightGBM with Strictly Limited Features (≤5)
=================================================================

【核心架构 - Referee-Player】
这是 Player (选手) 模块，唯一职责是计算 Alpha 评分。

【唯一接口】
- compute_score(df: pd.DataFrame) -> pd.DataFrame
  输入: 包含 OHLCV 等基础数据
  输出: 包含 trade_date, symbol, score 的 DataFrame

【V230 核心假设 H3 - LightGBM 限制特征≤5】
1. 假设: LightGBM 在严格控制特征数量（≤5）的情况下，
   可以捕捉非线性关系而不过拟合。

2. 核心改进:
   a. 严格限制 5 个特征:
      - ret_5d: 5日收益率（反转信号）
      - vol_20d: 20日波动率（风险信号）
      - net_main_rate: 主力净流入率（资金流信号）
      - industry_rel_20d: 行业相对20日收益率（行业信号）
      - volume_surge: 成交量放量倍数（异动信号）
   
   b. LightGBM 参数极度保守:
      - num_leaves=8（极低复杂度）
      - learning_rate=0.01（极慢学习）
      - min_child_samples=500（极多样本要求）
      - n_estimators=50（极少树）
      - 无早停，固定迭代

3. 训练策略:
   - 使用分类标签（T+5收益符号）替代回归
   - Winsorize 截断 ±10%
   - 截面 rank 标准化

【合规锁定】
- 严禁任何 shift(-1) 或未来引用
- 所有因子仅使用 T 日及之前数据
- 严禁接触回测逻辑
- 严格使用 data_healer 处理缺失值
- 特征数量严格 ≤5
"""

from typing import Dict, Optional, List
import sys
import os

import pandas as pd
import numpy as np
from loguru import logger
import lightgbm as lgb

# 内存优化
pd.options.mode.chained_assignment = None

# 版本号
VERSION = "V230"

# 是否使用 LightGBM
USE_LIGHTGBM = True


class AlphaModelV230:
    """
    V230 Alpha Model - Player (选手)
    
    【核心职责】
    使用 LightGBM（限制特征≤5）合成因子得分
    
    【唯一接口】
    - compute_score(df) -> DataFrame[trade_date, symbol, score]
    """
    
    # ==================== 配置参数 ====================
    # LightGBM 参数（极度保守，防止过拟合）
    LGBM_PARAMS = {
        'objective': 'binary',  # 分类问题（预测涨跌）
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 8,  # 极低复杂度
        'learning_rate': 0.01,  # 极慢学习
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_estimators': 50,  # 极少树
        'min_child_samples': 500,  # 极多样本要求
        'reg_alpha': 0.5,  # 强 L1 正则
        'reg_lambda': 0.5,  # 强 L2 正则
    }
    
    # 特征窗口
    RET_WINDOW = 5
    VOL_WINDOW = 20
    INDUSTRY_WINDOW = 20
    VOLUME_WINDOW = 20
    
    def __init__(self):
        """初始化 Alpha Model"""
        self.model = None
        self.feature_names = None
        self.is_trained = False
        self._market_state_weights = None
        
        logger.info("=" * 70)
        logger.info("V230 Alpha Model Initialized (LightGBM ≤5 Features)")
        logger.info("=" * 70)
        logger.info(f"  Use LightGBM: {USE_LIGHTGBM}")
        logger.info(f"  Objective: binary (classification)")
        logger.info(f"  Num Leaves: {self.LGBM_PARAMS['num_leaves']}")
        logger.info(f"  Learning Rate: {self.LGBM_PARAMS['learning_rate']}")
        logger.info(f"  Min Child Samples: {self.LGBM_PARAMS['min_child_samples']}")
        logger.info(f"  N Estimators: {self.LGBM_PARAMS['n_estimators']}")
        logger.info(f"  Features: ret_5d, vol_20d, net_main_rate, industry_rel_20d, volume_surge")
        logger.info("=" * 70)
    
    # ==================== 唯一公开接口 ====================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (唯一公开接口)
        
        【接口规范】
        - 输入: pd.DataFrame with columns [trade_date, symbol, open, high, low, close, volume, amount, industry_code]
        - 输出: pd.DataFrame with columns [trade_date, symbol, score]
        
        【计算流程】
        1. 特征工程
        2. 数据预处理 (data_healer)
        3. 使用 LightGBM 预测（或回退到线性加权）
        4. 截面百分位排名标准化
        """
        from src.data_healer import heal
        
        logger.info(f"[V230] Computing alpha scores for {len(df)} rows...")
        
        # 1. 特征工程
        df = self._engineer_features(df)
        
        # 2. 获取特征列
        feature_cols = self._get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]
        
        # 3. 使用 data_healer 处理缺失值
        df = heal(df, numeric_cols=available_cols)
        
        # 4. 计算得分
        if USE_LIGHTGBM and self.model is not None:
            logger.info("[V230] Using LightGBM model for prediction...")
            df['score'] = self._predict_with_model(df)
        else:
            logger.info("[V230] Using linear fallback for prediction...")
            df['score'] = self._compute_linear_score(df)
        
        # 5. 截面百分位排名标准化
        df['score'] = df.groupby('trade_date')['score'].rank(pct=True, na_option='keep')
        df['score'] = df['score'].fillna(0.5)
        
        # 6. 输出结果
        result_cols = ['trade_date', 'symbol', 'score']
        if 'close' in df.columns:
            result_cols.append('close')
        result = df[result_cols].copy()
        
        logger.info(f"[V230] Score computed: mean={result['score'].mean():.4f}, std={result['score'].std():.4f}")
        
        return result
    
    # ==================== 特征工程 ====================
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征工程 - 严格限制 5 个特征
        """
        from src.data_healer import heal
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算日收益率
        result['daily_ret'] = result.groupby('symbol')['close'].pct_change()
        
        # 特征 1: 5 日收益率 (反转信号)
        result['ret_5d'] = result.groupby('symbol')['close'].transform(
            lambda x, w=self.RET_WINDOW: x.pct_change(w)
        )
        
        # 特征 2: 20 日波动率 (风险信号)
        result['vol_20d'] = result.groupby('symbol')['daily_ret'].transform(
            lambda x, w=self.VOL_WINDOW: x.rolling(w, min_periods=10).std()
        )
        
        # 特征 3: 主力净流入率 (资金流信号)
        if 'net_main_rate' in result.columns:
            result['net_main_rate'] = result['net_main_rate']
        else:
            result['net_main_rate'] = 0.0
        
        # 特征 4: 行业相对 20 日收益率 (行业信号)
        result = self._compute_industry_relative(result)
        
        # 特征 5: 成交量放量倍数 (异动信号)
        result[f'vol_ma_{self.VOLUME_WINDOW}d'] = result.groupby('symbol')['volume'].transform(
            lambda x, w=self.VOLUME_WINDOW: x.rolling(w, min_periods=10).mean()
        )
        result['volume_surge'] = result['volume'] / result[f'vol_ma_{self.VOLUME_WINDOW}d'].replace(0, np.nan)
        
        # 替换 inf 为 NaN
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 使用 data_healer 处理缺失值
        feature_cols = self._get_feature_columns()
        available_cols = [c for c in feature_cols if c in result.columns]
        if available_cols:
            result = heal(result, numeric_cols=available_cols)
        
        logger.debug(f"[V230] Feature Engineering] Generated {len(available_cols)} features: {available_cols}")
        
        return result
    
    def _compute_industry_relative(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算行业相对收益率
        """
        if 'industry_code' not in df.columns or 'daily_ret' not in df.columns:
            df['industry_rel_20d'] = 0.0
            return df
        
        # 计算行业每日中位数收益率
        industry_daily = df.groupby(['trade_date', 'industry_code'])['daily_ret'].median().reset_index()
        industry_daily.columns = ['trade_date', 'industry_code', 'industry_idx_ret']
        
        # 合并行业指数回原数据
        df = df.merge(industry_daily, on=['trade_date', 'industry_code'], how='left')
        
        # 计算个股相对行业的超额收益
        df['industry_rel_ret'] = df['daily_ret'] - df['industry_idx_ret']
        
        # 计算行业相对累计收益（窗口期内）
        df['industry_rel_20d'] = df.groupby('symbol')['industry_rel_ret'].transform(
            lambda x, d=self.INDUSTRY_WINDOW: x.rolling(d, min_periods=10).sum()
        )
        
        return df
    
    def _get_feature_columns(self) -> List[str]:
        """获取特征列名列表（严格 5 个）"""
        return ['ret_5d', 'vol_20d', 'net_main_rate', 'industry_rel_20d', 'volume_surge']
    
    # ==================== 模型训练 ====================
    
    def train(self, df: pd.DataFrame) -> Dict:
        """
        训练 LightGBM 模型
        
        Args:
            df: 训练数据（已包含特征工程后的列）
        
        Returns:
            训练结果字典
        """
        if not USE_LIGHTGBM:
            logger.info("[V230] LightGBM disabled, skipping training")
            return {'model': None}
        
        logger.info("=" * 70)
        logger.info("[V230] Training LightGBM model...")
        logger.info("=" * 70)
        
        # 1. 特征工程
        df = self._engineer_features(df)
        
        # 2. 计算标签: 未来 5 日收益符号 (分类问题)
        df['future_5d_ret'] = df.groupby('symbol')['close'].transform(
            lambda x, w=5: x.shift(-w) / x - 1
        )
        df['label'] = (df['future_5d_ret'] > 0).astype(int)
        
        # 3. 获取特征
        feature_cols = self._get_feature_columns()
        available_cols = [c for c in feature_cols if c in df.columns]
        
        if len(available_cols) < 3:
            logger.warning(f"[V230] Not enough features available ({available_cols}), falling back to linear")
            return {'model': None}
        
        # 4. 准备训练数据
        train_data = df[available_cols + ['label']].copy()
        
        # 删除有 NaN 的行
        train_data = train_data.dropna()
        
        if len(train_data) < 10000:
            logger.warning(f"[V230] Not enough training data ({len(train_data)}), falling back to linear")
            return {'model': None}
        
        logger.info(f"[V230] Training data: {len(train_data)} samples, {len(available_cols)} features")
        
        # 5. 计算标签分布
        label_dist = train_data['label'].value_counts()
        logger.info(f"[V230] Label distribution: {label_dist.to_dict()}")
        
        # 6. 训练模型
        X = train_data[available_cols].values
        y = train_data['label'].values
        
        train_dataset = lgb.Dataset(X, label=y, feature_name=available_cols)
        
        self.model = lgb.train(
            self.LGBM_PARAMS,
            train_dataset,
            valid_sets=[train_dataset],
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        self.feature_names = available_cols
        self.is_trained = True
        
        # 7. 特征重要性
        importance = self.model.feature_importance(importance_type='gain')
        logger.info(f"[V230] Feature importance:")
        for name, imp in zip(available_cols, importance):
            logger.info(f"  {name}: {imp:.0f}")
        
        return {
            'model': self.model,
            'feature_names': self.feature_names,
            'feature_importance': dict(zip(available_cols, importance.tolist())),
        }
    
    # ==================== 模型预测 ====================
    
    def _predict_with_model(self, df: pd.DataFrame) -> pd.Series:
        """
        使用 LightGBM 模型预测
        """
        if self.model is None or self.feature_names is None:
            return self._compute_linear_score(df)
        
        feature_cols = [c for c in self.feature_names if c in df.columns]
        
        if len(feature_cols) < 3:
            return self._compute_linear_score(df)
        
        X = df[feature_cols].values
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, index=df.index)
    
    def _compute_linear_score(self, df: pd.DataFrame) -> pd.Series:
        """
        线性加权得分（回退方案）
        """
        weights = {
            'ret_5d': -0.4,
            'vol_20d': -0.2,
            'net_main_rate': 0.2,
            'industry_rel_20d': -0.1,
            'volume_surge': 0.1,
        }
        
        score = pd.Series(0.0, index=df.index)
        
        for col, w in weights.items():
            if col in df.columns:
                # 截面标准化
                norm = df.groupby('trade_date')[col].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 1e-8 else 0
                )
                score += w * norm.fillna(0)
        
        return score
    
    # ==================== 辅助方法 ====================
    
    def get_factor_ics(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取各因子的 IC"""
        ics = {}
        if 't1_return' not in df.columns:
            return ics
        
        for col in self._get_feature_columns():
            if col in df.columns:
                ic = self._calculate_daily_ic(df, col, 't1_return')
                ics[col] = ic
        return ics
    
    def _calculate_daily_ic(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str
    ) -> float:
        """计算日度平均 IC"""
        ic_values = []
        
        for date in sorted(df['trade_date'].unique()):
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 10:
                continue
            
            factor_vals = day_data[factor_col]
            return_vals = day_data[return_col]
            
            mask = factor_vals.notna() & return_vals.notna()
            if mask.sum() < 10:
                continue
            
            corr = factor_vals[mask].corr(return_vals[mask], method='spearman')
            if not np.isnan(corr):
                ic_values.append(corr)
        
        return float(np.mean(ic_values)) if ic_values else 0.0
    
    def save_model(self, path: str) -> None:
        """保存模型到文件"""
        if self.model is not None:
            self.model.save_model(path)
            logger.info(f"[V230] Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """从文件加载模型"""
        if os.path.exists(path):
            self.model = lgb.Booster(model_file=path)
            self.is_trained = True
            logger.info(f"[V230] Model loaded from {path}")


def get_alpha_model() -> AlphaModelV230:
    """获取 Alpha Model 实例"""
    return AlphaModelV230()


if __name__ == "__main__":
    logger.info("V230 Alpha Model loaded successfully")
    model = get_alpha_model()
    logger.info(f"Version: {VERSION}")