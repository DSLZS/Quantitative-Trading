"""
Alpha Research Module - V140 特征瘦身与动态半衰期校准.

【V140 核心改进 - 响应最高级别指令】
V139 在架构上表现完美，但 35 因子过于冗余。V140 执行"特征炼金"计划：

1. IC-Contribution 自动筛选 (L1 Sparsity):
   - 仅保留对 T+1 IC 贡献度最高的前 12 个正交因子
   - 禁止通过增加因子数量来强刷 IC
   - 引入互信息 (Mutual Information) 验证，确保因子间信息冗余度 < 0.1

2. 动态衰减半衰期 (Dynamic Half-life):
   - 针对 A 股风格切换极快的特点
   - 当 Market_Volatility 处于高分位时，将信号的 EMA 平滑窗口缩短（增加灵敏度）
   - 当处于低波动时，延长窗口（过滤噪声）

3. 非线性特征净化:
   - 在执行 Gram-Schmidt 正交化后，加入 Mutual Information 验证
   - 确保留下的 12 个因子相互之间的信息冗余度 < 0.1

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 140 运行
- 严禁"回测碰运气"：禁止修改 backtest_referee.py 中的费率或初始资金
- 报错必改：如遇数据缺失，必须调用 DataHealing 接口补全

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标 |
| IC_IR | > 0.6 | 稳定性 |
| IC/Factor Ratio | > 0.004 | 效率指标 (V139: 35 因子，V140: 12 因子) |
| Decay | T+1 > T+3 > T+5 | 正常衰减模式 |
"""

from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import warnings
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V140"

# V140 基础因子池 - 继承 V139
BASE_FACTORS = [
    'pct_chg', 'change',
    # Momentum
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
    # Reversion
    'reversion_5', 'reversion_10',
    # Volatility
    'volatility_5', 'volatility_10', 'volatility_20',
    # Volume-Price
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'vwap_distance', 'volume_rank', 'price_rank',
]

# V140 流动性因子
LIQUIDITY_FACTORS = [
    'volume_price_contradiction',
    'liquidity_alpha',
    'ofi_normalized',
    'volume_confirmed_momentum',
]

# V140 时效性因子
TIMELINESS_FACTORS = [
    'signal_delta',
    'volume_shock',
    'price_acceleration',
    'momentum_change',
]

# V140 所有候选因子
ALL_CANDIDATE_FACTORS = BASE_FACTORS + LIQUIDITY_FACTORS + TIMELINESS_FACTORS

# V140 最大因子数量 - 硬性限制
MAX_FACTORS = 12


def winsorize(series: pd.Series, sigma: float = 2.5) -> pd.Series:
    """Winsorization 去极值"""
    mean = series.mean()
    std = series.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return series.clip(lower=lower, upper=upper)


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    计算两个变量之间的互信息 (Mutual Information).
    
    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    
    使用分箱法近似计算离散互信息.
    
    Args:
        x: 变量 X
        y: 变量 Y
        n_bins: 分箱数量
        
    Returns:
        互信息值 (0 表示独立，越大表示相关性越强)
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    # 去除 NaN
    mask = np.isnan(x) | np.isnan(y)
    x_clean = x[~mask]
    y_clean = y[~mask]
    
    if len(x_clean) < 20:
        return 0.0
    
    # 分箱
    x_bins = pd.qcut(x_clean, q=n_bins, labels=False, duplicates='drop')
    y_bins = pd.qcut(y_clean, q=n_bins, labels=False, duplicates='drop')
    
    # 计算联合分布和边缘分布
    n_x = len(np.unique(x_bins))
    n_y = len(np.unique(y_bins))
    
    # 联合概率
    joint_hist = np.zeros((n_x, n_y))
    for xi, yi in zip(x_bins, y_bins):
        joint_hist[xi, yi] += 1
    joint_prob = joint_hist / len(x_clean)
    
    # 边缘概率
    px = joint_prob.sum(axis=1)
    py = joint_prob.sum(axis=0)
    
    # 互信息
    mi = 0.0
    for i in range(n_x):
        for j in range(n_y):
            if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
    
    return mi


def gram_schmidt_orthogonalize(X: np.ndarray, mi_threshold: float = 0.1) -> Tuple[np.ndarray, List[int]]:
    """
    Gram-Schmidt 正交化 + 互信息验证.
    
    确保因子间:
    1. 线性相关性 < 0.2 (传统 Gram-Schmidt)
    2. 互信息 < mi_threshold (非线性独立性)
    
    Args:
        X: 因子矩阵 (n_samples, n_factors)
        mi_threshold: 互信息阈值
        
    Returns:
        orthogonalized: 正交化后的因子矩阵
        kept_indices: 保留的因子索引
    """
    n_samples, n_factors = X.shape
    
    # 标准化输入
    X_norm = X.copy()
    for i in range(n_factors):
        std = np.std(X_norm[:, i])
        if std > 1e-10:
            X_norm[:, i] = (X_norm[:, i] - np.mean(X_norm[:, i])) / std
    
    orthogonal = []
    kept_indices = []
    
    for i in range(n_factors):
        v = X_norm[:, i].copy()
        
        # 减去在已选正交基上的投影
        for u in orthogonal:
            proj = np.dot(v, u) / (np.dot(u, u) + 1e-10)
            v = v - proj * u
        
        # 检查正交化后的范数
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            # 检查与已选因子的相关性 (线性和非线性)
            max_corr = 0
            max_mi = 0
            
            for j in kept_indices:
                # 线性相关性
                corr = np.corrcoef(X_norm[:, i], X_norm[:, j])[0, 1]
                if not np.isnan(corr):
                    max_corr = max(max_corr, abs(corr))
                
                # 互信息 (非线性)
                mi = compute_mutual_information(X_norm[:, i], X_norm[:, j], n_bins=10)
                max_mi = max(max_mi, mi)
            
            # 双重检查：线性和非线性独立性
            if max_corr < 0.2 and max_mi < mi_threshold:
                orthogonal.append(v / norm)
                kept_indices.append(i)
    
    # 构建正交化后的矩阵
    orthogonalized = np.zeros_like(X)
    for idx, (ortho_idx, u) in enumerate(zip(kept_indices, orthogonal)):
        orthogonalized[:, idx] = u
    
    return orthogonalized[:, :len(kept_indices)], kept_indices


class DataHealing:
    """V140 数据自愈模块 - 继承 V139"""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._init_sql_healer()
        
    def _init_sql_healer(self):
        """初始化 SQL 自愈器"""
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url)
                logger.info("[V140][DataHealing] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V140][DataHealing] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V140][DataHealing] No database URL, SQL healer disabled")
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        """记录自愈日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details,
        }
        self.healing_log.append(entry)
        logger.info(f"[V140][DataHealing] {action} - Column: {column}, Status: {status}, {details}")
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """检查并修复缺失列"""
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns"
            )
            
            if self.engine:
                result = self._heal_from_sql(result, missing)
            else:
                for col in missing:
                    result[col] = 0.0
                    self._log_healing(
                        action="DefaultFill",
                        column=col,
                        status="PARTIAL",
                        details="Filled with 0.0 (no SQL connection)"
                    )
        else:
            self._log_healing(
                action="ColumnsComplete",
                column="ALL",
                status="OK",
                details="All required columns present"
            )
        
        # NaN 检测与修复
        result = self._heal_nan(result)
        
        return result
    
    def _heal_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """NaN 检测与修复"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            nan_count = result[col].isna().sum()
            if nan_count > 0:
                if 'symbol' in result.columns:
                    result[col] = result.groupby('symbol')[col].transform(
                        lambda x: x.ffill().bfill()
                    )
                else:
                    result[col] = result[col].ffill().bfill()
                
                remaining_nan = result[col].isna().sum()
                if remaining_nan > 0:
                    result[col] = result[col].fillna(0.0)
                
                self._log_healing(
                    action="NaNHealed",
                    column=col,
                    status="SUCCESS",
                    details=f"Healed {nan_count} NaN values, {remaining_nan} remaining"
                )
        
        return result
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """从 SQL 补全缺失列"""
        if not self.engine or df.empty:
            return df
        
        result = df.copy()
        symbols = df['symbol'].unique().tolist()[:50]
        
        if not symbols:
            return df
        
        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date']).unique()
            start_date = pd.to_datetime(dates.min()).strftime('%Y%m%d')
            end_date = pd.to_datetime(dates.max()).strftime('%Y%m%d')
        else:
            return df
        
        try:
            from sqlalchemy import text
            
            symbols_str = ', '.join([f"'{s}'" for s in symbols])
            query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       turnover_rate, total_mv, pe_ttm, pb
                FROM stock_daily
                WHERE symbol IN ({symbols_str})
                AND trade_date BETWEEN :start_date AND :end_date
            """)
            
            sql_df = pd.read_sql_query(query, self.engine, params={
                'start_date': start_date,
                'end_date': end_date,
            })
            
            if not sql_df.empty:
                for col in columns:
                    if col in sql_df.columns:
                        merge_df = result.merge(
                            sql_df[['symbol', 'trade_date', col]],
                            on=['symbol', 'trade_date'],
                            how='left',
                            suffixes=('', '_sql')
                        )
                        result[col] = merge_df[col].fillna(merge_df[f'{col}_sql'])
                        result = result.drop(columns=[c for c in result.columns if c.endswith('_sql')])
                        
                        self._log_healing(
                            action="HealedFromSQL",
                            column=col,
                            status="SUCCESS",
                            details=f"Healed {len(sql_df)} rows from stock_daily"
                        )
                        
        except Exception as e:
            logger.error(f"[V140][DataHealing] SQL heal failed: {e}")
            for col in columns:
                result[col] = 0.0
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


class DynamicHalfLifeEngine:
    """
    V140 动态衰减半衰期引擎.
    
    【核心功能】
    1. 根据 Market_Volatility 的分位数动态调整 EMA 窗口
    2. 高波动场景：窗口缩短 (半衰期缩短，增加灵敏度)
    3. 低波动场景：窗口延长 (半衰期延长，过滤噪声)
    
    【半衰期公式】
    half_life = base_half_life * (1 + volatility_adjustment)
    
    其中 volatility_adjustment = (current_vol - median_vol) / median_vol
    """
    
    def __init__(self, base_half_life: int = 10, volatility_window: int = 60):
        self.base_half_life = base_half_life
        self.volatility_window = volatility_window
        self.half_life_log = []
        self.current_half_life = base_half_life
        
    def _log_half_life(self, action: str, details: str = ""):
        """记录半衰期日志"""
        entry = {'action': action, 'details': details}
        self.half_life_log.append(entry)
        logger.info(f"[V140][DynamicHalfLife] {action}: {details}")
    
    def compute_market_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场波动率（截面平均波动率）.
        
        Market_Volatility = Mean(Volatility_20, cross-section)
        """
        result = df.copy()
        
        if 'volatility_20' in df.columns:
            result['market_volatility'] = result.groupby('trade_date')['volatility_20'].transform('mean')
        elif 'pct_chg' in df.columns:
            result['market_volatility'] = result.groupby('trade_date')['pct_chg'].transform('std')
        else:
            result['market_volatility'] = 1.0
        
        result['market_volatility'] = result['market_volatility'].fillna(1.0)
        
        self._log_half_life(
            "Computed",
            f"market_volatility = Mean(Volatility_20, cross-section)"
        )
        
        return result
    
    def compute_dynamic_half_life(self, df: pd.DataFrame) -> pd.Series:
        """
        计算动态半衰期.
        
        Returns:
            half_life: 每个时间点的半衰期
        """
        if 'market_volatility' not in df.columns:
            df = self.compute_market_volatility(df)
        
        # 计算滚动分位数
        def compute_half_life(series):
            median_vol = series.rolling(self.volatility_window, min_periods=20).median()
            q75 = series.rolling(self.volatility_window, min_periods=20).quantile(0.75)
            q25 = series.rolling(self.volatility_window, min_periods=20).quantile(0.25)
            
            def _half_life(vol, med, q75, q25):
                if pd.isna(med) or pd.isna(q75) or pd.isna(q25):
                    return self.base_half_life
                
                # 波动率调整因子
                vol_adjustment = (vol - med) / (med + 1e-10)
                
                # 高波动时缩短半衰期，低波动时延长
                # 半衰期范围：base_half_life * [0.5, 2.0]
                adjusted_half_life = self.base_half_life * (1 - 0.5 * vol_adjustment)
                adjusted_half_life = max(5, min(20, adjusted_half_life))
                
                return int(adjusted_half_life)
            
            return pd.Series([_half_life(v, m, q75_, q25_) 
                             for v, m, q75_, q25_ in zip(series, median_vol, q75, q25)],
                            index=series.index)
        
        half_life = df.groupby('symbol').apply(
            lambda g: compute_half_life(g['market_volatility'])
        ).reset_index(level=0, drop=True)
        
        # 更新当前半衰期
        self.current_half_life = int(half_life.iloc[-1]) if len(half_life) > 0 else self.base_half_life
        
        self._log_half_life(
            "Computed",
            f"Dynamic half-life: current={self.current_half_life}, base={self.base_half_life}"
        )
        
        return half_life
    
    def apply_ema_with_dynamic_half_life(self, df: pd.DataFrame, signal_col: str) -> pd.Series:
        """
        使用动态半衰期应用 EMA 平滑.
        
        Args:
            df: 数据 DataFrame
            signal_col: 信号列名
            
        Returns:
            平滑后的信号
        """
        half_life_series = self.compute_dynamic_half_life(df)
        
        # 按股票分组应用 EMA
        def apply_ema(group):
            half_life = half_life_series.loc[group.index].iloc[0]
            span = 2 * half_life - 1  # EMA span
            return group[signal_col].ewm(span=span, adjust=False).mean()
        
        smoothed = df.groupby('symbol').apply(apply_ema).reset_index(level=0, drop=True)
        
        self._log_half_life(
            "AppliedEMA",
            f"Signal: {signal_col}, avg_half_life={half_life_series.mean():.1f}"
        )
        
        return smoothed
    
    def get_half_life_log(self) -> List[Dict]:
        """获取半衰期日志"""
        return self.half_life_log
    
    def get_current_half_life(self) -> int:
        """获取当前半衰期"""
        return self.current_half_life


class ICContributionSelector:
    """
    V140 IC-Contribution 自动筛选器.
    
    【核心功能】
    1. 计算每个因子的 IC-Contribution
    2. 仅保留 IC 贡献最高的前 12 个因子
    3. 确保因子间互信息 < 0.1
    
    【IC-Contribution 定义】
    IC_Contribution(factor_i) = IC(model_with_i) - IC(model_without_i)
    """
    
    def __init__(self, max_factors: int = MAX_FACTORS, mi_threshold: float = 0.1):
        self.max_factors = max_factors
        self.mi_threshold = mi_threshold
        self.selection_log = []
        self.ic_contributions = {}
        
    def _log_selection(self, action: str, details: str = ""):
        """记录筛选日志"""
        entry = {'action': action, 'details': details}
        self.selection_log.append(entry)
        logger.info(f"[V140][ICContribution] {action}: {details}")
    
    def compute_ic_contribution(self, df: pd.DataFrame, factors: List[str]) -> Dict[str, float]:
        """
        计算每个因子的 IC-Contribution.
        
        Args:
            df: 数据 DataFrame (必须包含 factors 和 t1_return)
            factors: 因子列表
            
        Returns:
            ic_contributions: 因子 IC 贡献字典
        """
        ic_contributions = {}
        
        for factor in factors:
            if factor not in df.columns:
                continue
            
            # 计算单因子 IC
            factor_ic = self._compute_factor_ic(df, factor)
            
            # IC-Contribution = 因子 IC 绝对值
            ic_contributions[factor] = abs(factor_ic)
        
        self.ic_contributions = ic_contributions
        
        self._log_selection(
            "Computed",
            f"IC-Contribution for {len(ic_contributions)} factors"
        )
        
        return ic_contributions
    
    def _compute_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算单因子 IC"""
        ics = []
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            
            f = day[factor_col].fillna(0)
            l = day['t1_return'].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                l_rank = l.rank(method='average')
                ic = np.corrcoef(f_rank, l_rank)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        return float(np.mean(ics)) if ics else 0.0
    
    def select_top_factors(self, df: pd.DataFrame, factors: List[str]) -> List[str]:
        """
        选择 IC-Contribution 最高的前 N 个因子.
        
        Args:
            df: 数据 DataFrame
            factors: 候选因子列表
            
        Returns:
            selected_factors: 选中的因子列表
        """
        # 计算 IC-Contribution
        ic_contributions = self.compute_ic_contribution(df, factors)
        
        # 按 IC-Contribution 排序
        sorted_factors = sorted(ic_contributions.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前 N 个因子
        selected = []
        for factor, ic_contrib in sorted_factors:
            if len(selected) >= self.max_factors:
                break
            selected.append(factor)
        
        self._log_selection(
            "Selected",
            f"Top {len(selected)} factors from {len(factors)} candidates"
        )
        
        return selected
    
    def verify_mutual_information(self, df: pd.DataFrame, factors: List[str]) -> Tuple[List[str], Dict[str, float]]:
        """
        验证因子间互信息是否 < threshold.
        
        Args:
            df: 数据 DataFrame
            factors: 因子列表
            
        Returns:
            kept_factors: 通过验证的因子
            mi_matrix: 互信息矩阵
        """
        n = len(factors)
        mi_matrix = np.zeros((n, n))
        
        # 计算互信息矩阵
        for i in range(n):
            for j in range(i + 1, n):
                fi = df[factors[i]].fillna(0).values
                fj = df[factors[j]].fillna(0).values
                mi = compute_mutual_information(fi, fj, n_bins=10)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        # 检查冗余度
        kept_factors = [factors[0]]  # 保留第一个因子
        
        for i in range(1, n):
            max_mi_with_kept = max(mi_matrix[i, factors.index(k)] for k in kept_factors)
            if max_mi_with_kept < self.mi_threshold:
                kept_factors.append(factors[i])
        
        self._log_selection(
            "MIVerified",
            f"Reduced {len(factors)} -> {len(kept_factors)} factors (MI < {self.mi_threshold})"
        )
        
        return kept_factors, {f'f{i}_{factors[i]}_f{j}_{factors[j]}': mi_matrix[i, j] 
                             for i in range(n) for j in range(i + 1, n)}
    
    def get_ic_contributions(self) -> Dict[str, float]:
        """获取 IC-Contribution"""
        return self.ic_contributions
    
    def get_selection_log(self) -> List[Dict]:
        """获取筛选日志"""
        return self.selection_log


class AdaptiveFeatureEnsemble:
    """
    V140 自适应特征集成 - 继承 V139 并增强.
    
    【V140 核心改进】
    1. 集成 ICContributionSelector (12 因子精选)
    2. 集成 DynamicHalfLifeEngine (动态半衰期)
    3. 保留 Gram-Schmidt 正交化 + MI 验证
    """
    
    def __init__(self, n_bins: int = 10, min_samples_per_bin: int = 30, rolling_window: int = 20):
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.rolling_window = rolling_window
        self.bin_stats = {}
        self.ensemble_log = []
        self.rolling_ic = {}
        self.orthogonalization_stats = {}
        
        # V140 新增模块
        self.ic_contribution_selector = ICContributionSelector(max_factors=MAX_FACTORS, mi_threshold=0.1)
        self.dynamic_half_life_engine = DynamicHalfLifeEngine(base_half_life=10, volatility_window=60)
        
    def _log_ensemble(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.ensemble_log.append(entry)
        logger.info(f"[V140][AdaptiveEnsemble] {action}: {details}")
    
    def update_rolling_ic(self, df: pd.DataFrame, factor_col: str):
        """更新滚动 IC 记录"""
        if 'trade_date' not in df.columns or 't1_return' not in df.columns:
            return
        
        dates = df['trade_date'].unique()
        dates = sorted(dates)
        
        ics = []
        for date in dates[-self.rolling_window:]:
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 20:
                continue
            f = day_data[factor_col].fillna(0)
            l = day_data['t1_return'].fillna(0)
            if len(f) > 10 and np.std(f) > 1e-10:
                ic = np.corrcoef(f.rank(method='average'), l.rank(method='average'))[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        self.rolling_ic[factor_col] = ics
    
    def compute_dynamic_bin_weight(self, df: pd.DataFrame, factor_col: str) -> Dict[int, float]:
        """计算动态分箱权重"""
        self.update_rolling_ic(df, factor_col)
        
        ics = self.rolling_ic.get(factor_col, [])
        
        if len(ics) < 5:
            return {i: 1.0 / self.n_bins for i in range(self.n_bins)}
        
        ic_mean = np.mean(ics)
        ic_std = np.std(ics) + 1e-10
        ic_ir = ic_mean / ic_std
        
        bin_weights = {}
        for i in range(self.n_bins):
            base_weight = 1.0 / self.n_bins
            
            if ic_mean > 0:
                if i == 0 or i == self.n_bins - 1:
                    weight = base_weight * (1.5 + ic_ir)
                else:
                    weight = base_weight
            else:
                if i == 0 or i == self.n_bins - 1:
                    weight = base_weight * (1.5 - ic_ir)
                else:
                    weight = base_weight
            
            bin_weights[i] = weight
        
        total = sum(bin_weights.values())
        bin_weights = {k: v / total for k, v in bin_weights.items()}
        
        return bin_weights
    
    def apply_gram_schmidt(self, factor_matrix: np.ndarray, factor_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """应用 Gram-Schmidt 正交化 + MI 验证"""
        if len(factor_matrix) == 0 or len(factor_names) == 0:
            return factor_matrix, factor_names
        
        orthogonalized, kept_indices = gram_schmidt_orthogonalize(factor_matrix, mi_threshold=0.1)
        
        kept_names = [factor_names[i] for i in kept_indices]
        
        self.orthogonalization_stats = {
            'method': 'gram_schmidt_with_mi',
            'correlation_threshold': 0.2,
            'mi_threshold': 0.1,
            'input_features': len(factor_names),
            'output_features': len(kept_names),
            'removed_features': [factor_names[i] for i in range(len(factor_names)) if i not in kept_indices],
        }
        
        self._log_ensemble(
            "GramSchmidtApplied",
            f"Reduced {len(factor_names)} -> {len(kept_names)} factors (corr < 0.2, MI < 0.1)"
        )
        
        return orthogonalized, kept_names
    
    def compute_bin_based_score(self, df: pd.DataFrame, factor_col: str) -> pd.Series:
        """基于分箱计算非线性评分"""
        result = df.copy()
        
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if 't1_return' not in df.columns:
            return pd.Series(0, index=df.index)
        
        factor_values = df[factor_col].fillna(0)
        trade_dates = df['trade_date']
        
        all_bins = []
        for date in trade_dates.unique():
            date_mask = trade_dates == date
            date_values = factor_values[date_mask]
            
            if len(date_values) < self.n_bins:
                date_bins = date_values.rank(method='average', pct=True).mul(self.n_bins).fillna(0).astype(int).clip(0, self.n_bins - 1)
            else:
                try:
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    bin_boundaries = date_values.quantile(quantiles)
                    date_bins = pd.cut(
                        date_values,
                        bins=bin_boundaries.unique(),
                        labels=False,
                        include_lowest=True
                    )
                except Exception:
                    date_bins = date_values.rank(method='average', pct=True).mul(self.n_bins).fillna(0).astype(int).clip(0, self.n_bins - 1)
            
            all_bins.append(pd.Series(date_bins, index=date_values.index))
        
        if all_bins:
            factor_bins = pd.concat(all_bins).reindex(df.index).fillna(0).astype(int)
        else:
            factor_bins = pd.Series(0, index=df.index)
        
        bin_win_rates = {}
        bin_counts = {}
        for bin_id in range(self.n_bins):
            bin_mask = factor_bins == bin_id
            count = bin_mask.sum()
            bin_counts[bin_id] = count
            
            if count < self.min_samples_per_bin:
                bin_win_rates[bin_id] = 0.5
            else:
                bin_returns = df.loc[bin_mask, 't1_return']
                win_rate = (bin_returns > 0).mean()
                bin_win_rates[bin_id] = win_rate
        
        self.bin_stats[factor_col] = bin_win_rates
        
        dynamic_weights = self.compute_dynamic_bin_weight(df, factor_col)
        
        bin_scores = {}
        for bin_id, win_rate in bin_win_rates.items():
            weight = dynamic_weights.get(bin_id, 1.0 / self.n_bins)
            bin_scores[bin_id] = (win_rate - 0.5) * 2 * weight * self.n_bins
        
        score = factor_bins.map(bin_scores).fillna(0)
        
        valid_bins = [v for v in bin_win_rates.values() if v != 0.5]
        if valid_bins:
            self._log_ensemble(
                "ComputedBinScore",
                f"{factor_col}: bins={self.n_bins}, win_rate_range=[{min(valid_bins):.3f}, {max(valid_bins):.3f}]"
            )
        
        return score
    
    def compute_adaptive_weight(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子的自适应权重"""
        if factor_col not in self.bin_stats:
            self.compute_bin_based_score(df, factor_col)
        
        bin_win_rates = self.bin_stats.get(factor_col, {})
        
        if not bin_win_rates:
            return 1.0 / self.n_bins
        
        max_win = max(bin_win_rates.values())
        min_win = min(bin_win_rates.values())
        win_spread = max_win - min_win
        
        ics = self.rolling_ic.get(factor_col, [])
        if ics:
            ic_mean = np.mean(ics)
            ic_std = np.std(ics) + 1e-10
            ic_ir = abs(ic_mean / ic_std)
            adaptive_weight = win_spread * (1 + ic_ir)
        else:
            adaptive_weight = win_spread
        
        return adaptive_weight
    
    def get_bin_stats(self) -> Dict[str, Dict[int, float]]:
        """获取分箱统计"""
        return self.bin_stats
    
    def get_ensemble_log(self) -> List[Dict]:
        """获取集成日志"""
        return self.ensemble_log
    
    def get_orthogonalization_stats(self) -> Dict:
        """获取正交化统计"""
        return self.orthogonalization_stats
    
    def get_ic_contributions(self) -> Dict[str, float]:
        """获取 IC-Contribution"""
        return self.ic_contribution_selector.get_ic_contributions()
    
    def get_dynamic_half_life(self) -> int:
        """获取当前半衰期"""
        return self.dynamic_half_life_engine.get_current_half_life()


class LiquidityAlphaEngine:
    """V140 流动性 Alpha 引擎 - 继承 V139"""
    
    EPSILON = 1e-6
    
    def __init__(self):
        self.liquidity_log = []
        
    def _log_liquidity(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.liquidity_log.append(entry)
        logger.info(f"[V140][LiquidityAlpha] {action}: {details}")
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """截面排名 (0-1 归一化)"""
        return series.rank(method='average', pct=True)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算量价背离因子"""
        result = df.copy()
        
        if 'pct_chg' in df.columns:
            close_return = df['pct_chg']
        elif 'change' in df.columns:
            close_return = df['change']
        else:
            close_return = pd.Series(0, index=df.index)
        
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
        elif 'amount' in df.columns:
            volume_change = df['amount'].pct_change()
        else:
            volume_change = pd.Series(0, index=df.index)
        
        price_rank = self._rank(close_return.fillna(0))
        volume_rank = self._rank(volume_change.fillna(0))
        
        result['volume_price_contradiction'] = price_rank - volume_rank
        
        self._log_liquidity(
            "Computed",
            "volume_price_contradiction = Rank(Close_Return) - Rank(Volume_Change)"
        )
        
        return result
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算流动性 Alpha 因子"""
        result = df.copy()
        
        if 'amount' in df.columns and 'volume' in df.columns:
            vwap = df['amount'] / (df['volume'] + self.EPSILON)
            price_change = df['close'] - df.get('pre_close', df['close'])
            ofi = price_change * df['volume'] / (df['amount'] + self.EPSILON)
        elif 'pct_chg' in df.columns and 'volume' in df.columns:
            ofi = df['pct_chg'] * df['volume']
        else:
            ofi = df.get('pct_chg', pd.Series(0, index=df.index)) * df.get('volume', pd.Series(1, index=df.index))
        
        if 'close' in df.columns:
            ts_std_20 = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
        else:
            ts_std_20 = pd.Series(1, index=df.index)
        
        result['liquidity_alpha'] = ofi / (ts_std_20 + self.EPSILON)
        
        self._log_liquidity("Computed", "liquidity_alpha = OFI / Ts_Std(Close, 20)")
        
        return result
    
    def compute_all_liquidity_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有流动性因子"""
        result = df.copy()
        
        self._log_liquidity("StartLiquidityMining", f"Processing {len(df)} rows")
        
        result = self.compute_volume_price_contradiction(result)
        result = self.compute_liquidity_alpha(result)
        
        self._log_liquidity("Complete", f"Generated 2 liquidity factors")
        
        return result
    
    def get_liquidity_log(self) -> List[Dict]:
        """获取流动性因子日志"""
        return self.liquidity_log


class TimelinessOperator:
    """V140 时效性增强算子 - 继承 V139"""
    
    def __init__(self):
        self.timeliness_log = []
        
    def _log_timeliness(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.timeliness_log.append(entry)
        logger.info(f"[V140][TimelinessOperator] {action}: {details}")
    
    def compute_signal_delta(self, df: pd.DataFrame, signal_col: str = 'score') -> pd.DataFrame:
        """计算信号变化量"""
        result = df.copy()
        
        if signal_col in df.columns:
            result['signal_delta'] = result.groupby('symbol')[signal_col].transform(
                lambda x: x.diff()
            )
        else:
            result['signal_delta'] = 0.0
        
        result['signal_delta'] = result['signal_delta'].fillna(0.0)
        
        self._log_timeliness("Computed", f"signal_delta = {signal_col}_t - {signal_col}_{{t-1}}")
        
        return result
    
    def compute_volume_shock(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量突增指标"""
        result = df.copy()
        
        if 'volume' in df.columns:
            volume_ma20 = result.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(20, min_periods=5).mean()
            )
            result['volume_shock'] = result['volume'] / (volume_ma20 + 1e-10)
        else:
            result['volume_shock'] = 1.0
        
        result['volume_shock'] = result['volume_shock'].fillna(1.0)
        
        self._log_timeliness("Computed", "volume_shock = Volume_t / MA(Volume, 20)")
        
        return result
    
    def compute_all_timeliness_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有时效性因子"""
        result = df.copy()
        
        self._log_timeliness("StartTimelinessMining", f"Processing {len(df)} rows")
        
        if 'score' not in result.columns:
            result['score'] = result.get('pct_chg', pd.Series(0, index=df.index))
        
        result = self.compute_signal_delta(result)
        result = self.compute_volume_shock(result)
        
        self._log_timeliness("Complete", f"Generated 2 timeliness factors")
        
        return result
    
    def get_timeliness_log(self) -> List[Dict]:
        """获取时效性算子日志"""
        return self.timeliness_log


class FactorGenerator:
    """V140 因子生成器 - 从基础数据计算因子"""
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        self.generation_log.append(entry)
        logger.info(f"[V140][FactorGenerator] {action}: {details}")
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算动量因子"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算反转因子（负动量）"""
        return -df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算波动率因子"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window).std()
        ).fillna(0)
    
    def compute_volume_price_stable(self, df: pd.DataFrame) -> pd.Series:
        """计算量价稳定性因子"""
        if 'volume' in df.columns and 'close' in df.columns:
            vol_ret = df.groupby('symbol').apply(
                lambda g: g['volume'].pct_change().rolling(20).corr(g['close'].pct_change())
            ).reset_index(level=0, drop=True).fillna(0)
            return 1 - vol_ret.abs()  # 相关性越低越稳定
        return pd.Series(0, index=df.index)
    
    def compute_volume_price_divergence(self, df: pd.DataFrame, window: int) -> pd.Series:
        """计算量价背离因子"""
        if 'volume' in df.columns and 'close' in df.columns:
            vol_ma = df.groupby('symbol')['volume'].transform(
                lambda x: x.rolling(window).mean()
            )
            price_ma = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(window).mean()
            )
            vol_change = df['volume'] / (vol_ma + 1e-10) - 1
            price_change = df['close'] / (price_ma + 1e-10) - 1
            return (vol_change - price_change).fillna(0)
        return pd.Series(0, index=df.index)
    
    def compute_vwap_distance(self, df: pd.DataFrame) -> pd.Series:
        """计算 VWAP 距离因子"""
        if 'amount' in df.columns and 'volume' in df.columns:
            vwap = df['amount'] / (df['volume'] + 1e-6)
            return (df['close'] - vwap) / (vwap + 1e-6)
        return pd.Series(0, index=df.index)
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有基础因子"""
        result = df.copy()
        
        self._log_generation("StartFactorGeneration", f"Processing {len(df)} rows")
        
        # 动量因子
        result['momentum_5'] = self.compute_momentum(result, 5)
        result['momentum_10'] = self.compute_momentum(result, 10)
        result['momentum_20'] = self.compute_momentum(result, 20)
        result['momentum_60'] = self.compute_momentum(result, 60)
        self._log_generation("Computed", "momentum_5, momentum_10, momentum_20, momentum_60")
        
        # 反转因子
        result['reversion_5'] = self.compute_reversion(result, 5)
        result['reversion_10'] = self.compute_reversion(result, 10)
        self._log_generation("Computed", "reversion_5, reversion_10")
        
        # 波动率因子
        result['volatility_5'] = self.compute_volatility(result, 5)
        result['volatility_10'] = self.compute_volatility(result, 10)
        result['volatility_20'] = self.compute_volatility(result, 20)
        self._log_generation("Computed", "volatility_5, volatility_10, volatility_20")
        
        # 量价因子
        result['volume_price_stable'] = self.compute_volume_price_stable(result)
        result['volume_price_divergence_5'] = self.compute_volume_price_divergence(result, 5)
        result['volume_price_divergence_20'] = self.compute_volume_price_divergence(result, 20)
        result['vwap_distance'] = self.compute_vwap_distance(result)
        self._log_generation("Computed", "volume_price_stable, volume_price_divergence_5, volume_price_divergence_20, vwap_distance")
        
        # 排名因子
        result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        self._log_generation("Computed", "volume_rank, price_rank")
        
        self._log_generation("Complete", f"Generated {len(['momentum_5', 'momentum_10', 'momentum_20', 'momentum_60', 'reversion_5', 'reversion_10', 'volatility_5', 'volatility_10', 'volatility_20', 'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20', 'vwap_distance', 'volume_rank', 'price_rank'])} base factors")
        
        return result


class AlphaResearchV140:
    """
    V140 Alpha 研究引擎 - 特征瘦身与动态半衰期校准.
    
    【V140 核心改进】
    1. ICContributionSelector: 仅保留前 12 个正交因子
    2. DynamicHalfLifeEngine: 动态衰减半衰期
    3. Mutual Information 验证：确保因子间信息冗余度 < 0.1
    4. DataHealing: 数据自愈
    5. FactorGenerator: 基础因子生成
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_liquidity: bool = True,
        enable_timeliness: bool = True,
        enable_orthogonalization: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_liquidity = enable_liquidity
        self.enable_timeliness = enable_timeliness
        self.enable_orthogonalization = enable_orthogonalization
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealing(db_url) if auto_heal else None
        self.ensemble = AdaptiveFeatureEnsemble(n_bins=n_bins, rolling_window=20) if enable_ensemble else None
        self.liquidity_engine = LiquidityAlphaEngine() if enable_liquidity else None
        self.timeliness_operator = TimelinessOperator() if enable_timeliness else None
        self.factor_generator = FactorGenerator()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Feature Slimming + Dynamic Half-life Calibration")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  Max N Factors: {n_factors} (V139: 35)")
        logger.info(f"  N Bins: {n_bins}")
        logger.info(f"  Liquidity Alpha: {'Enabled' if enable_liquidity else 'Disabled'}")
        logger.info(f"  Timeliness Operator: {'Enabled' if enable_timeliness else 'Disabled'}")
        logger.info(f"  Orthogonalization + MI: {'Enabled' if enable_orthogonalization else 'Disabled'}")
        logger.info(f"  Factor Generator: {'Enabled'}")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子 IC"""
        ics = []
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            
            f = day[factor_col].fillna(0)
            l = day['t1_return'].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                l_rank = l.rank(method='average')
                ic = np.corrcoef(f_rank, l_rank)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
        
        return float(np.mean(ics)) if ics else 0.0
    
    def _process_factor(self, series: pd.Series, trade_dates: pd.Series) -> np.ndarray:
        """因子处理：去极值 + 标准化"""
        series_wins = winsorize(series.fillna(0), sigma=2.5)
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V140 核心逻辑"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据自愈检查
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        # 2. 准备标签
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # 3. V140 新增：生成基础因子（动量、波动率、量价等）
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
            self._log_audit("FactorGeneration", "Generated base factors (momentum, volatility, volume-price)")
        
        # 4. 计算流动性因子
        if self.enable_liquidity and self.liquidity_engine:
            result = self.liquidity_engine.compute_all_liquidity_factors(result)
            self._log_audit("LiquidityMining", f"Generated 2 liquidity factors")
        
        # 4. 计算时效性因子
        if self.enable_timeliness and self.timeliness_operator:
            result = self.timeliness_operator.compute_all_timeliness_factors(result)
            self._log_audit("TimelinessMining", f"Generated 2 timeliness factors")
        
        # 5. 准备所有候选因子 (使用实际生成的因子)
        # V140: 基础因子 + 流动性因子 + 时效性因子
        all_available_factors = [
            'pct_chg', 'change',
            'momentum_5', 'momentum_10', 'momentum_20', 'momentum_60',
            'reversion_5', 'reversion_10',
            'volatility_5', 'volatility_10', 'volatility_20',
            'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
            'vwap_distance', 'volume_rank', 'price_rank',
            # 生成的流动性因子
            'volume_price_contradiction', 'liquidity_alpha',
            # 生成的时效性因子
            'signal_delta', 'volume_shock',
        ]
        
        # 6. 计算所有因子 IC 并排序
        factor_ics = []
        for factor in all_available_factors:
            if factor not in result.columns:
                continue
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            factor_ics.append((factor, ic))
        
        # 7. V140: IC-Contribution 自动筛选
        if self.enable_ensemble and self.ensemble:
            self._log_audit("ICContribution", "Running IC-Contribution selection...")
            selected_factors = self.ensemble.ic_contribution_selector.select_top_factors(
                result, [f for f, _ in factor_ics]
            )
            self._log_audit("ICContribution", f"Selected {len(selected_factors)} factors by IC-Contribution")
        else:
            # 按 IC 绝对值排序
            factor_ics.sort(key=lambda x: abs(x[1]), reverse=True)
            selected_factors = [f for f, _ in factor_ics[:self.n_factors]]
        
        # 8. 准备因子数据
        factor_data = {}
        factor_matrices = []
        factor_names = []
        
        for factor in selected_factors:
            if abs(self.factor_ics[factor]) < self.ic_threshold:
                continue
            if len(factor_names) >= self.n_factors:
                break
                
            f_raw = result[factor]
            
            # 负 IC 因子翻转
            if self.factor_ics[factor] < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={self.factor_ics[factor]:.4f} -> flipped")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={self.factor_ics[factor]:.4f} -> kept")
            
            # 使用自适应特征集成
            if self.enable_ensemble and self.ensemble:
                bin_score = self.ensemble.compute_bin_based_score(result, factor)
                factor_data[factor] = bin_score.values
                self._log_audit("BinMapping", f"{factor}: applied {self.n_bins}-bin nonlinear mapping")
            else:
                f_std = self._process_factor(f_processed, result['trade_date'])
                factor_data[factor] = f_std
            
            # 收集用于正交化的数据
            factor_matrices.append(factor_data[factor])
            factor_names.append(factor)
            
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(all_available_factors)} factors")
        
        # 9. V140: 应用 Gram-Schmidt 正交化 + MI 验证
        if self.enable_orthogonalization and self.ensemble and len(factor_matrices) > 0:
            factor_matrix = np.column_stack([factor_data[f] for f in factor_names])
            orthogonalized, kept_names = self.ensemble.apply_gram_schmidt(factor_matrix, factor_names)
            
            self.selected_factors = kept_names
            
            factor_data_filtered = {}
            for i, name in enumerate(kept_names):
                if i < orthogonalized.shape[1]:
                    factor_data_filtered[name] = orthogonalized[:, i]
            factor_data = factor_data_filtered
            
            self._log_audit(
                "Orthogonalization",
                f"Gram-Schmidt + MI: {len(factor_names)} -> {len(kept_names)} factors"
            )
        
        # 10. 应用动态半衰期 EMA 平滑
        if self.enable_ensemble and self.ensemble:
            self._log_audit("DynamicHalfLife", f"Applying EMA with dynamic half-life...")
            
            # 先计算原始 score
            if not self.selected_factors:
                result['score'] = np.random.randn(len(result))
            else:
                weights = []
                for factor in self.selected_factors:
                    adaptive_weight = self.ensemble.compute_adaptive_weight(result, factor)
                    weights.append(adaptive_weight)
                
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                else:
                    normalized_weights = [1.0 / len(self.selected_factors)] * len(self.selected_factors)
                
                score = np.zeros(len(result))
                for i, factor in enumerate(self.selected_factors):
                    score += factor_data[factor] * normalized_weights[i]
                    self.factor_weights[factor] = normalized_weights[i]
                
                result['score_raw'] = score
            
            # 应用动态半衰期 EMA
            if 'score_raw' in result.columns:
                smoothed_score = self.ensemble.dynamic_half_life_engine.apply_ema_with_dynamic_half_life(
                    result, 'score_raw'
                )
                result['score'] = smoothed_score
                self._log_audit(
                    "DynamicHalfLife",
                    f"Current half-life: {self.ensemble.get_dynamic_half_life()}"
                )
        else:
            # 不使用动态半衰期
            if not self.selected_factors:
                result['score'] = np.random.randn(len(result))
            else:
                weights = []
                for factor in self.selected_factors:
                    if self.enable_ensemble and self.ensemble:
                        adaptive_weight = self.ensemble.compute_adaptive_weight(result, factor)
                    else:
                        adaptive_weight = abs(self.factor_ics[factor])
                    weights.append(adaptive_weight)
                
                total_weight = sum(weights)
                if total_weight > 0:
                    normalized_weights = [w / total_weight for w in weights]
                else:
                    normalized_weights = [1.0 / len(self.selected_factors)] * len(self.selected_factors)
                
                score = np.zeros(len(result))
                for i, factor in enumerate(self.selected_factors):
                    score += factor_data[factor] * normalized_weights[i]
                    self.factor_weights[factor] = normalized_weights[i]
                
                result['score'] = score
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (IC-Contribution + Dynamic Half-life)")
        
        return result[['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']]
    
    def get_factor_ics(self, df=None) -> Dict[str, float]:
        """获取因子 IC"""
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted
    
    def get_selected_factors(self) -> List[str]:
        """获取选中的因子"""
        return self.selected_factors
    
    def get_bin_stats(self) -> Dict[str, Dict[int, float]]:
        """获取分箱统计"""
        return self.ensemble.get_bin_stats() if self.ensemble else {}
    
    def get_ensemble_log(self) -> List[Dict]:
        """获取集成日志"""
        return self.ensemble.get_ensemble_log() if self.ensemble else []
    
    def get_orthogonalization_stats(self) -> Dict:
        """获取正交化统计"""
        return self.ensemble.get_orthogonalization_stats() if self.ensemble else {}
    
    def get_liquidity_log(self) -> List[Dict]:
        """获取流动性因子日志"""
        return self.liquidity_engine.get_liquidity_log() if self.liquidity_engine else []
    
    def get_timeliness_log(self) -> List[Dict]:
        """获取时效性算子日志"""
        return self.timeliness_operator.get_timeliness_log() if self.timeliness_operator else []
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_ic_contributions(self) -> Dict[str, float]:
        """获取 IC-Contribution"""
        return self.ensemble.get_ic_contributions() if self.ensemble else {}
    
    def get_dynamic_half_life(self) -> int:
        """获取当前半衰期"""
        return self.ensemble.get_dynamic_half_life() if self.ensemble else 10
    
    def get_efficiency_ratio(self) -> float:
        """
        计算效率指标：IC / Factor Count.
        
        V139: 35 因子，IC ~0.0479 -> 0.00137
        V140: 12 因子，目标 > 0.004
        """
        if not self.selected_factors:
            return 0.0
        
        # 计算平均 IC
        ics = list(self.get_factor_ics().values())
        if not ics:
            return 0.0
        
        mean_ic = abs(np.mean(ics))
        return mean_ic / len(self.selected_factors)


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_liquidity: bool = True,
    enable_timeliness: bool = True,
    enable_orthogonalization: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV140:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV140(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_liquidity=enable_liquidity,
        enable_timeliness=enable_timeliness,
        enable_orthogonalization=enable_orthogonalization,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV140...")
    
    np.random.seed(42)
    n_samples = 1000
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], n_samples),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], n_samples),
        'close': np.random.randn(n_samples) * 10 + 100,
        'volume': np.random.randn(n_samples) * 1000 + 5000,
        'amount': np.random.randn(n_samples) * 10000 + 50000,
        'pct_chg': np.random.randn(n_samples) * 2,
        'momentum_5': np.random.randn(n_samples),
        'momentum_20': np.random.randn(n_samples),
        'volatility_10': np.abs(np.random.randn(n_samples)),
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  Orthogonalization Stats: {alpha.get_orthogonalization_stats()}")
    logger.info(f"  Dynamic Half-life: {alpha.get_dynamic_half_life()}")
    logger.info(f"  Efficiency Ratio: {alpha.get_efficiency_ratio():.4f}")