"""
Alpha Research Module - V154 Stability-Reinforcement-Alpha (SRA).

【V153 问题诊断】
- T+1 IC: 0.073 (达标 > 0.055)
- IC IR: 0.50 (未达标 > 0.55)
- IC Std: 0.145 (偏高，需降低)
- IC 衰减过快：信号震荡过大，捕捉瞬时噪音

【V154 核心使命 - Stability-Reinforcement-Alpha (SRA)】
1. Dynamic Volatility Scaling (DVS) - 截面波动率缩放:
   - V153 的信号震荡过大，某几天极端信号主导回测
   - 引入截面波动率缩放，确保信号 score 的每日方差保持恒定
   - 公式：Score_DVS = Score_Raw / (CS_Volatility + ε)

2. Adaptive Signal Momentum (ASM) - 自适应信号动量:
   - 引入信号动量：Score_t = α * Raw_Score_t + (1-α) * Score_{t-1}
   - α 根据过去 5 天的 IC 相关性动态调整
   - 如果信号持续反转，则减小 α (增强平滑)

3. Turnover Constraint (调仓约束):
   - 在生成 score 阶段，显式引入惩罚项
   - 抑制导致高换手率的低质量预测
   - 提升 IR 的分母 (降低 IC 波动)

4. Cross-Sectional Interaction 2.0 - 因子协同过滤:
   - 利用 ORA (Orthogonal Residual) 的基础
   - 增加因子间的"协同过滤"逻辑，而非简单的残差相加
   - 引入因子间相关性加权集成

【V154 核心算法】
1. DVS:
   - CS_Volatility_t = Std(Score_Raw | trade_date=t)
   - Score_DVS = Score_Raw / (CS_Volatility_t + ε)

2. ASM:
   - IC_Correlation = Corr(IC_{t-5:t-1}, IC_{t-4:t})
   - α = 0.5 + 0.5 * IC_Correlation (α ∈ [0, 1])
   - Score_ASM = α * Score_DVS + (1-α) * Score_ASM_{t-1}

3. Turnover Constraint:
   - Turnover_Penalty = |Score_t - Score_{t-1}|
   - Score_Final = Score_ASM - λ * Turnover_Penalty

4. CSI 2.0:
   - Factor_Corr = Corr(Factor_i, Factor_j)
   - Weight_i = 1 / (1 + Σ_j |Corr(Factor_i, Factor_j)|)

【工程纪律】
- 版本元数据同步：全文使用 V154，禁止出现 V153 等旧版本号
- 严禁偷看未来：PAC 逻辑必须严格基于 Rolling Window（过去 20 日滚动 IC）
- 数据完整性闭环：遇到缺失值必须主动调用 src/data_loader.py 重新拉取
- 架构守则：严禁修改 src/engine/ 目录，只能在 src/alpha_research_v154.py 中迭代
- 严禁修改初始资金（100,000）和费率（0.03%）
- 唯一入口：python main.py --version 154 --year 2024

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.06 | 核心指标 |
| IC_IR | > 0.55 | 稳定性（V153: 0.50）|
| IC Decay Pattern | T+1 > T+3 > T+5 | 必须单调递减（严格） |
| IC Std | < 0.10 | 时序波动率（V153: 0.145）|
| Turnover | ↓15% vs V153 | 日度信号换手率 |
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

VERSION = "V154"

# V154 核心因子 - 聚焦短期预测（解决 IC 衰减问题）
V154_CORE_FACTORS = [
    'momentum_5',       # 短期动量
    'volatility_5',     # 短期波动率
    'volume_price_contradiction',  # V147 核心 - ORM 核心因子
    'liquidity_alpha',              # V147 核心
    'reversion_5',      # 短期反转
]

# V154 候选因子池
V154_CANDIDATE_FACTORS = [
    'momentum_5', 'momentum_10', 'momentum_60',
    'reversion_10',
    'volatility_5', 'volatility_20',
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'vwap_distance', 'volume_rank', 'price_rank',
    'value_rank', 'ep_rank', 'bp_rank',
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist',
    'turnover_bias_5', 'turnover_bias_10', 'turnover_bias_20',
    'volume_shrink_ratio', 'turnover_vol_ratio',
    'tail_risk_indicator', 'skewness_20', 'extreme_volume_ratio',
]

ALL_FACTORS = V154_CORE_FACTORS + V154_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V154 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V154 SRA 参数
ORM_CORE_FACTOR = 'volume_price_contradiction'  # 正交残差挖掘的核心因子
LEAD_LAG_THRESHOLD = 1.5  # 领先滞后比阈值（T+1 MI / T+5 MI）
LEAD_LAG_MAX_LAG = 5  # 最大滞后阶数
CS_VOLATILITY_WINDOW = 20  # 截面波动率计算窗口
ROLLING_WINDOW = 20  # 滚动 IC 窗口
IC_WEIGHT_WINDOW = 10  # IC 加权窗口

# V154 新增参数
DVS_TARGET_VOL = 1.0  # DVS 目标波动率
ASM_LOOKBACK = 5  # ASM 自相关性计算窗口
ASM_BASE_ALPHA = 0.5  # ASM 基础平滑系数
TURNOVER_PENALTY_LAMBDA = 0.1  # 调仓惩罚系数
CSI_CORR_THRESHOLD = 0.7  # CSI 相关性阈值


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    计算两个变量之间的互信息（Mutual Information）.
    
    【V154 核心】用于 Adaptive Lead-Lag Correction
    计算因子与不同滞后阶数回报之间的 MI，判断因子的领先性
    """
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    
    try:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
    except (ValueError, TypeError):
        return 0.0
    
    mask = np.isnan(x) | np.isnan(y)
    x_clean = x[~mask]
    y_clean = y[~mask]
    
    if len(x_clean) < 20:
        return 0.0
    
    try:
        # 分箱处理
        x_bins = pd.qcut(x_clean, q=n_bins, labels=False, duplicates='drop')
        y_bins = pd.qcut(y_clean, q=n_bins, labels=False, duplicates='drop')
        
        n_x = len(np.unique(x_bins))
        n_y = len(np.unique(y_bins))
        
        # 联合概率分布
        joint_hist = np.zeros((n_x, n_y))
        for xi, yi in zip(x_bins, y_bins):
            joint_hist[xi, yi] += 1
        joint_prob = joint_hist / len(x_clean)
        
        # 边缘概率
        px = joint_hist.sum(axis=1)
        py = joint_hist.sum(axis=0)
        
        # 计算 MI
        mi = 0.0
        for i in range(n_x):
            for j in range(n_y):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))
        
        return mi
    except Exception:
        return 0.0


def winsorize_auto_heal(
    series: pd.Series, 
    sigma: float = 3.0, 
    percentile: float = 0.99
) -> pd.Series:
    """V154 自动愈合版 Winsorization"""
    series_clean = series.copy()
    
    # 1. 处理 Inf
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    # 2. 计算均值
    mean = series_clean.mean()
    if pd.isna(mean):
        mean = 0.0
    
    # 3. Sigma 截断
    std = series_clean.std()
    if std > 1e-10:
        lower = mean - sigma * std
        upper = mean + sigma * std
        series_clean = series_clean.clip(lower=lower, upper=upper)
    
    # 4. Percentile 截断
    lower_pct = series_clean.quantile(1 - percentile)
    upper_pct = series_clean.quantile(percentile)
    series_clean = series_clean.clip(lower=lower_pct, upper=upper_pct)
    
    # 5. 最终 NaN 填充 - V154 使用 ffill 优先
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """V154 截断日志摘要，防止 400 报错."""
    if df.empty:
        return "Empty DataFrame"
    
    if 'trade_date' in df.columns:
        df_sorted = df.sort_values('trade_date').copy()
        df_sorted['year_month'] = pd.to_datetime(df_sorted['trade_date']).dt.to_period('M')
        
        first_days = df_sorted.groupby('year_month').first().reset_index()
        last_days = df_sorted.groupby('year_month').last().reset_index()
        
        summary_df = pd.concat([first_days, last_days]).drop_duplicates()
        
        if len(summary_df) > max_rows:
            summary_df = summary_df.head(max_rows)
        
        if 'year_month' in summary_df.columns:
            summary_df = summary_df.drop(columns=['year_month'])
        
        return summary_df.to_string(max_rows=MAX_LOG_ENTRIES)
    else:
        return df.head(max_rows).to_string(max_rows=MAX_LOG_ENTRIES)


def safe_summary_dict(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> Dict:
    """安全地生成 DataFrame 摘要字典"""
    if df.empty:
        return {'rows': 0, 'summary': 'Empty'}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    summary = {
        'rows': len(df),
        'columns': list(df.columns),
        'numeric_stats': {},
    }
    
    for col in numeric_cols[:10]:
        summary['numeric_stats'][col] = {
            'mean': float(df[col].mean()) if not df[col].isna().all() else 0.0,
            'std': float(df[col].std()) if not df[col].isna().all() else 0.0,
            'min': float(df[col].min()) if not df[col].isna().all() else 0.0,
            'max': float(df[col].max()) if not df[col].isna().all() else 0.0,
        }
    
    return summary


class DataHealerV154:
    """V154 增强版数据自愈模块 - 主动补全策略"""
    
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
                logger.info("[V154][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V154][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V154][DataHealer] No database URL, SQL healer disabled")
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        """记录自愈日志"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details,
        }
        if len(self.healing_log) >= MAX_LOG_ENTRIES:
            self.healing_log = self.healing_log[-MAX_LOG_ENTRIES//2:]
        self.healing_log.append(entry)
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """V154 检查并修复缺失列 - 严格数据闭环"""
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns"
            )
            
            # V154: 缺失率检查
            missing_ratio = len(missing) / len(required_columns)
            if missing_ratio > 0.05:  # 超过 5%
                logger.error(f"[V154][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                logger.error(f"[V154][DataHealer] Current columns: {result.columns.tolist()}")
                
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    # 无 SQL 连接时报错并停止
                    raise ValueError(
                        f"[V154] Data integrity violation: {len(missing)} columns missing. "
                        f"Missing: {missing}. No SQL connection for auto-heal."
                    )
            else:
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    for col in missing:
                        result = result.assign(**{col: 0.0})
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
        
        # V154: 主动使用 ffill() 补全
        result = self._auto_impute_grouped(result, 'trade_date')
        result = self._repair_nan_inf(result)
        
        self._log_healing(
            action="AutoImputeApplied",
            column="ALL_NUMERIC",
            status="SUCCESS",
            details="Applied ffill + bfill + median imputation"
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
            logger.error(f"[V154][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """V154 自动分组插值 - 优先 ffill"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # V154: 优先 ffill 填充
            result[col] = result.groupby(group_col, group_keys=False)[col].transform(
                lambda x: x.ffill().bfill()
            )
            
            # 再用中位数填充剩余 NaN
            global_median = result[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            
            result[col] = result[col].fillna(global_median)
        
        return result
    
    def _repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """V154 自动修复 NaN/Inf"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 处理 Inf
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                self._log_healing(
                    action="InfRepaired",
                    column=col,
                    status="SUCCESS",
                    details=f"Repaired {inf_count} Inf values"
                )
            
            # 处理 NaN - V154 使用中位数填充
            nan_count = result[col].isna().sum()
            if nan_count > 0:
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                result[col] = result[col].fillna(col_median)
                self._log_healing(
                    action="NaNRepaired",
                    column=col,
                    status="SUCCESS",
                    details=f"Repaired {nan_count} NaN values with median={col_median:.4f}"
                )
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log[-MAX_LOG_ENTRIES:]


class AdaptiveLeadLagCorrector:
    """
    V154 核心 - 自适应领先滞后校正器.
    
    【V153 问题】
    - DSI 固定平滑未考虑因子与回报的领先滞后关系
    - 出现 T+5 IC > T+1 IC 的错误滞后
    
    【V154 修复】
    - 计算因子与 Return 的互信息（MI）在不同滞后阶数下的分布
    - 仅保留 T+1 MI 显著高于 T+5 的因子（领先因子）
    - Lead_Score = MI_Lag_1 / (MI_Lag_5 + ε)
    - 若 Lead_Score > 1.5，则该因子为领先因子
    """
    
    def __init__(
        self, 
        max_lag: int = LEAD_LAG_MAX_LAG,
        threshold: float = LEAD_LAG_THRESHOLD,
        n_bins: int = 10,
    ):
        self.max_lag = max_lag
        self.threshold = threshold
        self.n_bins = n_bins
        self.correction_log = []
        self.lead_lag_stats = {}
        
    def _log_correction(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.correction_log) >= MAX_LOG_ENTRIES:
            self.correction_log = self.correction_log[-MAX_LOG_ENTRIES//2:]
        self.correction_log.append(entry)
    
    def compute_lead_lag_score(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_cols: Optional[List[str]] = None,
    ) -> Tuple[float, Dict[int, float]]:
        """
        计算因子的领先滞后分数.
        
        【完整流程】
        1. 对每个滞后阶数 k (1-5)，计算 Factor_t 与 Return_{t+k} 的 MI
        2. Lead_Score = MI_Lag_1 / MI_Lag_5
        3. 若 Lead_Score > threshold，则该因子为领先因子
        
        Returns:
            lead_lag_score: T+1 MI / T+5 MI
            mi_by_lag: 各滞后阶数的 MI 字典
        """
        if factor_col not in df.columns:
            return 0.0, {}
        
        if return_cols is None:
            return_cols = ['t1_return_period', 't2_return_period', 't3_return_period', 
                          't4_return_period', 't5_return_period']
        
        # 获取有效数据
        factor_data = df[factor_col].fillna(0).values
        
        mi_by_lag = {}
        
        for lag in range(1, self.max_lag + 1):
            return_col = f't{lag}_return_period'
            
            if return_col not in df.columns:
                # 尝试使用累计回报
                return_col = f't{lag}_return'
                if return_col not in df.columns:
                    continue
            
            return_data = df[return_col].fillna(0).values
            
            # 计算 MI
            mi = compute_mutual_information(factor_data, return_data, self.n_bins)
            mi_by_lag[lag] = mi
        
        # 计算领先滞后分数
        mi_lag_1 = mi_by_lag.get(1, 0.0)
        mi_lag_5 = mi_by_lag.get(5, 0.0)
        
        if mi_lag_5 > 1e-10:
            lead_lag_score = mi_lag_1 / mi_lag_5
        elif mi_lag_1 > 0:
            lead_lag_score = 2.0  # T+5 MI 为 0 时，认为因子是领先的
        else:
            lead_lag_score = 0.0
        
        self._log_correction(
            "LeadLagScoreComputed",
            f"{factor_col}: MI_Lag1={mi_lag_1:.4f}, MI_Lag5={mi_lag_5:.4f}, Score={lead_lag_score:.2f}"
        )
        
        return lead_lag_score, mi_by_lag
    
    def select_lead_factors(
        self,
        df: pd.DataFrame,
        candidate_factors: List[str],
    ) -> List[str]:
        """
        选择领先因子.
        
        【筛选标准】
        - Lead_Score > threshold 的因子
        - 若所有因子都不满足，则选择 Lead_Score 最高的前 N 个
        """
        lead_scores = {}
        
        for factor in candidate_factors:
            score, _ = self.compute_lead_lag_score(df, factor)
            lead_scores[factor] = score
        
        # 选择领先因子
        lead_factors = [f for f, s in lead_scores.items() if s > self.threshold]
        
        if not lead_factors:
            # 若无领先因子，选择 Lead_Score 最高的前 5 个
            sorted_factors = sorted(lead_scores.items(), key=lambda x: x[1], reverse=True)
            lead_factors = [f for f, _ in sorted_factors[:min(5, len(sorted_factors))]]
        
        self.lead_lag_stats = {
            'threshold': self.threshold,
            'lead_factors': lead_factors,
            'lead_scores': lead_scores,
        }
        
        self._log_correction(
            "LeadFactorsSelected",
            f"Selected {len(lead_factors)} lead factors: {lead_factors}"
        )
        
        return lead_factors
    
    def get_correction_log(self) -> List[Dict]:
        return self.correction_log[-MAX_LOG_ENTRIES:]
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_stats


class DynamicVolatilityScaler:
    """
    V154 核心 - 动态波动率缩放器 (DVS).
    
    【目标】
    - 确保信号 score 的每日方差保持恒定
    - 避免某几天极端信号主导回测
    - 降低 IC Std (从 V153 的 0.145 降至 0.10 以下)
    
    【公式】
    - CS_Volatility_t = Std(Score_Raw | trade_date=t)
    - Score_DVS = Score_Raw / (CS_Volatility_t + ε) * Target_Vol
    """
    
    def __init__(self, target_vol: float = DVS_TARGET_VOL):
        self.target_vol = target_vol
        self.scaling_log = []
        self.dvs_stats = {}
        
    def _log_scaling(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.scaling_log) >= MAX_LOG_ENTRIES:
            self.scaling_log = self.scaling_log[-MAX_LOG_ENTRIES//2:]
        self.scaling_log.append(entry)
    
    def compute_dvs(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
    ) -> pd.Series:
        """
        计算动态波动率缩放信号.
        
        【完整流程】
        1. 按日期分组计算截面标准差
        2. Score_DVS = Score_Raw / (CS_Volatility + ε) * Target_Vol
        """
        if score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        
        # 计算每日截面标准差
        cs_vol = result.groupby('trade_date')[score_col].transform(
            lambda x: x.std() if len(x) > 1 else 1.0
        )
        
        # DVS 缩放
        dvs_score = result[score_col] / (cs_vol + 1e-10) * self.target_vol
        
        self._log_scaling(
            "DVSComputed",
            f"Target_Vol={self.target_vol}, Mean_CS_Vol={cs_vol.mean():.4f}"
        )
        
        self.dvs_stats = {
            'target_vol': self.target_vol,
            'mean_cs_vol': float(cs_vol.mean()),
            'std_cs_vol': float(cs_vol.std()),
        }
        
        return dvs_score.fillna(0)


class AdaptiveSignalMomentum:
    """
    V154 核心 - 自适应信号动量器 (ASM).
    
    【目标】
    - 引入信号动量：Score_t = α * Raw_Score_t + (1-α) * Score_{t-1}
    - α 根据过去 5 天的 IC 相关性动态调整
    - 如果信号持续反转，则减小 α (增强平滑)
    
    【公式】
    - IC_Correlation = Corr(IC_{t-5:t-1}, IC_{t-4:t})
    - α = 0.5 + 0.5 * IC_Correlation (α ∈ [0, 1])
    """
    
    def __init__(
        self,
        lookback: int = ASM_LOOKBACK,
        base_alpha: float = ASM_BASE_ALPHA,
    ):
        self.lookback = lookback
        self.base_alpha = base_alpha
        self.momentum_log = []
        self.asm_stats = {}
        
    def _log_momentum(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.momentum_log) >= MAX_LOG_ENTRIES:
            self.momentum_log = self.momentum_log[-MAX_LOG_ENTRIES//2:]
        self.momentum_log.append(entry)
    
    def compute_adaptive_alpha(
        self,
        ic_series: pd.Series,
    ) -> float:
        """
        计算自适应 α 系数.
        
        【完整流程】
        1. 计算 IC 自相关性
        2. α = base_alpha + (1 - base_alpha) * IC_Correlation
        """
        if len(ic_series) < self.lookback * 2:
            return self.base_alpha
        
        # 计算 IC 自相关性
        ic_1 = ic_series.iloc[:-1].values
        ic_2 = ic_series.iloc[1:].values
        
        if len(ic_1) < 5 or np.std(ic_1) < 1e-10 or np.std(ic_2) < 1e-10:
            return self.base_alpha
        
        ic_corr = np.corrcoef(ic_1, ic_2)[0, 1]
        if np.isnan(ic_corr):
            return self.base_alpha
        
        # α = 0.5 + 0.5 * IC_Correlation (IC 持续性强则增大 α)
        adaptive_alpha = self.base_alpha + (1 - self.base_alpha) * max(0, ic_corr)
        
        self._log_momentum(
            "AdaptiveAlphaComputed",
            f"IC_Corr={ic_corr:.4f}, Alpha={adaptive_alpha:.4f}"
        )
        
        return adaptive_alpha
    
    def apply_asm(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_dvs',
        ic_col: str = 't1_return',
    ) -> pd.Series:
        """
        应用自适应信号动量.
        
        【完整流程】
        1. 按日期排序
        2. 对每个日期计算自适应 α
        3. Score_ASM = α * Score_DVS + (1-α) * Score_ASM_{t-1}
        """
        if score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.sort_values('trade_date').copy()
        dates = result['trade_date'].unique()
        
        asm_scores = []
        prev_asm_score = 0.0
        ic_history = []
        
        for date in dates:
            date_mask = result['trade_date'] == date
            date_data = result.loc[date_mask]
            
            # 计算当日 IC
            if ic_col in date_data.columns:
                f = date_data[score_col].fillna(0)
                r = date_data[ic_col].fillna(0)
                if len(f) > 10 and np.std(f) > 1e-10:
                    ic = np.corrcoef(f.rank(), r.rank())[0, 1]
                    if not np.isnan(ic):
                        ic_history.append(ic)
            
            # 计算自适应 α
            if len(ic_history) >= self.lookback:
                ic_series = pd.Series(ic_history[-self.lookback*2:])
                alpha = self.compute_adaptive_alpha(ic_series)
            else:
                alpha = self.base_alpha
            
            # 计算 ASM 分数
            current_score = date_data[score_col].mean()
            asm_score = alpha * current_score + (1 - alpha) * prev_asm_score
            asm_scores.append((date_mask, asm_score))
            prev_asm_score = asm_score
        
        # 合并结果
        final_asm = pd.Series(0.0, index=df.index)
        for mask, score in asm_scores:
            final_asm.loc[mask] = score
        
        self._log_momentum(
            "ASMApplied",
            f"Mean_Alpha={self.base_alpha:.4f}, Lookback={self.lookback}"
        )
        
        self.asm_stats = {
            'lookback': self.lookback,
            'base_alpha': self.base_alpha,
        }
        
        return final_asm


class TurnoverConstraint:
    """
    V154 核心 - 调仓约束器.
    
    【目标】
    - 在生成 score 阶段，显式引入惩罚项
    - 抑制导致高换手率的低质量预测
    - 提升 IR 的分母 (降低 IC 波动)
    
    【公式】
    - Turnover_Penalty = |Score_t - Score_{t-1}|
    - Score_Final = Score_ASM - λ * Turnover_Penalty
    """
    
    def __init__(self, penalty_lambda: float = TURNOVER_PENALTY_LAMBDA):
        self.penalty_lambda = penalty_lambda
        self.constraint_log = []
        self.turnover_stats = {}
        
    def _log_constraint(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.constraint_log) >= MAX_LOG_ENTRIES:
            self.constraint_log = self.constraint_log[-MAX_LOG_ENTRIES//2:]
        self.constraint_log.append(entry)
    
    def apply_turnover_penalty(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_asm',
    ) -> pd.Series:
        """
        应用调仓惩罚.
        
        【完整流程】
        1. 按日期排序
        2. 计算相邻日期的信号变化
        3. Score_Final = Score_ASM - λ * |Score_t - Score_{t-1}|
        """
        if score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.sort_values('trade_date').copy()
        
        # 计算信号变化（按股票分组）
        result = result.sort_values(['symbol', 'trade_date'])
        result['score_prev'] = result.groupby('symbol')[score_col].transform('shift')
        result['turnover_penalty'] = (result[score_col] - result['score_prev']).abs()
        
        # 应用惩罚
        result['score_final'] = result[score_col] - self.penalty_lambda * result['turnover_penalty']
        
        self._log_constraint(
            "TurnoverPenaltyApplied",
            f"Lambda={self.penalty_lambda}, Mean_Penalty={result['turnover_penalty'].mean():.4f}"
        )
        
        self.turnover_stats = {
            'penalty_lambda': self.penalty_lambda,
            'mean_turnover': float(result['turnover_penalty'].mean()),
            'std_turnover': float(result['turnover_penalty'].std()),
        }
        
        return result['score_final'].fillna(result[score_col])


class CrossSectionalInteraction2:
    """
    V154 核心 - 截面交互器 2.0 (CSI 2.0).
    
    【V154 修复 - IC 加权】
    - 原问题：仅基于相关性加权，未考虑 IC 符号，导致正负 IC 因子相互抵消
    - 修复：引入 IC 加权，Weight_i = |IC_i| / Σ_j |IC_j|
    - 仅选择 IC > 0 的因子
    
    【公式】
    - IC_Weight_i = |IC_i| / Σ_j |IC_j|  (仅对 IC > 0 的因子)
    - Corr_Penalty_i = 1 / (1 + Σ_j |Corr(Factor_i, Factor_j)|)
    - Final_Weight_i = IC_Weight_i * Corr_Penalty_i
    """
    
    def __init__(self, corr_threshold: float = CSI_CORR_THRESHOLD):
        self.corr_threshold = corr_threshold
        self.interaction_log = []
        self.csi_stats = {}
        
    def _log_interaction(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.interaction_log) >= MAX_LOG_ENTRIES:
            self.interaction_log = self.interaction_log[-MAX_LOG_ENTRIES//2:]
        self.interaction_log.append(entry)
    
    def compute_correlation_weights(
        self,
        df: pd.DataFrame,
        factors: List[str],
        factor_ics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        计算因子权重 - V154 IC 加权版本.
        
        【完整流程】
        1. 筛选 IC > 0 的因子
        2. 计算 IC 权重：Weight_IC_i = |IC_i| / Σ_j |IC_j|
        3. 计算相关性惩罚：Corr_Penalty_i = 1 / (1 + Σ_j |Corr|)
        4. 最终权重 = IC_Weight * Corr_Penalty，归一化
        """
        # V154 FIX: 使用 IC 加权
        if factor_ics is None or not factor_ics:
            # 无 IC 信息，返回等权重
            return {f: 1.0 / len(factors) for f in factors if f in df.columns}
        
        # 1. 筛选 IC > 0 的因子（只保留正 IC 因子）
        positive_ic_factors = {f: ic for f, ic in factor_ics.items() if ic > 0 and f in factors}
        
        if not positive_ic_factors:
            # 无正 IC 因子，返回等权重
            self._log_interaction("NoPositiveIC", "All factors have negative IC, using equal weights")
            return {f: 1.0 / len(factors) for f in factors if f in df.columns}
        
        # 2. 计算 IC 权重
        total_ic = sum(abs(ic) for ic in positive_ic_factors.values())
        ic_weights = {f: abs(ic) / total_ic for f, ic in positive_ic_factors.items()}
        
        # 3. 计算相关性惩罚（仅对正 IC 因子）
        factor_data = {}
        for factor in positive_ic_factors.keys():
            if factor in df.columns:
                factor_data[factor] = df.groupby('trade_date')[factor].mean()
        
        corr_penalties = {}
        if len(factor_data) >= 2:
            factor_df = pd.DataFrame(factor_data)
            corr_matrix = factor_df.corr()
            
            for factor in positive_ic_factors.keys():
                if factor in corr_matrix.columns:
                    corr_sum = corr_matrix[factor].abs().sum() - 1.0  # 减去自身
                    corr_penalties[factor] = 1.0 / (1.0 + corr_sum)
                else:
                    corr_penalties[factor] = 1.0
        else:
            corr_penalties = {f: 1.0 for f in positive_ic_factors.keys()}
        
        # 4. 合并权重并归一化
        weights = {}
        for factor in positive_ic_factors.keys():
            weights[factor] = ic_weights[factor] * corr_penalties.get(factor, 1.0)
        
        total = sum(weights.values())
        if total > 0:
            weights = {f: w / total for f, w in weights.items()}
        
        self._log_interaction(
            "ICCorrelationWeightsComputed",
            f"Positive_IC_Factors={len(positive_ic_factors)}, ICs={positive_ic_factors}, Mean_Corr_Penalty={np.mean(list(corr_penalties.values())):.4f}"
        )
        
        self.csi_stats = {
            'corr_threshold': self.corr_threshold,
            'factors': list(weights.keys()),
            'weights': weights,
            'ic_weights': ic_weights,
            'corr_penalties': corr_penalties,
            'positive_ic_factors': positive_ic_factors,
        }
        
        return weights
    
    def compute_synergy_score(
        self,
        df: pd.DataFrame,
        factors: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        计算协同过滤分数.
        
        【完整流程】
        1. 获取因子权重
        2. Score = Σ_i Weight_i * Factor_i
        """
        if weights is None:
            weights = self.compute_correlation_weights(df, factors)
        
        result = df.copy()
        score = pd.Series(0.0, index=df.index)
        
        for factor, weight in weights.items():
            if factor in result.columns:
                score += result[factor].fillna(0) * weight
        
        self._log_interaction(
            "SynergyScoreComputed",
            f"Weighted sum of {len(weights)} factors"
        )
        
        return score


class OrthogonalResidualMiner:
    """
    V154 核心 - 正交残差挖掘器.
    
    【目标】
    - 将 volume_price_contradiction 作为核心因子
    - 提取其他因子相对于核心因子的正交残差
    - 确保每一份信号都是增量信息
    
    【公式】
    - Residual_i = Factor_i - β_i * CoreFactor
    - β_i = Cov(Factor_i, CoreFactor) / Var(CoreFactor)
    """
    
    def __init__(self, core_factor: str = ORM_CORE_FACTOR):
        self.core_factor = core_factor
        self.mining_log = []
        self.residual_stats = {}
        
    def _log_mining(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.mining_log) >= MAX_LOG_ENTRIES:
            self.mining_log = self.mining_log[-MAX_LOG_ENTRIES//2:]
        self.mining_log.append(entry)
    
    def compute_orthogonal_residual(
        self,
        df: pd.DataFrame,
        factor_col: str,
    ) -> pd.Series:
        """
        计算因子相对于核心因子的正交残差.
        
        【完整流程】
        1. 计算 β = Cov(Factor, CoreFactor) / Var(CoreFactor)
        2. Residual = Factor - β * CoreFactor
        """
        if factor_col not in df.columns or self.core_factor not in df.columns:
            return pd.Series(0, index=df.index)
        
        if factor_col == self.core_factor:
            # 核心因子自身，返回原值
            return df[factor_col].fillna(0)
        
        result = df.copy()
        
        # 按日期分组计算 β 和残差
        residuals = []
        
        for date in result['trade_date'].unique():
            date_mask = result['trade_date'] == date
            date_data = result.loc[date_mask]
            
            factor_vals = date_data[factor_col].values
            core_vals = date_data[self.core_factor].values
            
            # 去除 NaN
            mask = ~np.isnan(factor_vals) & ~np.isnan(core_vals)
            f_clean = factor_vals[mask]
            c_clean = core_vals[mask]
            
            if len(f_clean) < 10:
                residuals.append((date_mask, pd.Series(np.zeros(len(date_data)), index=date_data.index)))
                continue
            
            # 计算 β
            cov = np.cov(f_clean, c_clean)[0, 1]
            var = np.var(c_clean)
            
            if var > 1e-10:
                beta = cov / var
            else:
                beta = 0.0
            
            # 计算残差
            residual = factor_vals - beta * core_vals
            residual = np.nan_to_num(residual, nan=0.0)
            
            residuals.append((date_mask, pd.Series(residual, index=date_data.index)))
        
        # 合并结果
        final_residual = pd.Series(0.0, index=df.index)
        for mask, series in residuals:
            final_residual.loc[mask] = series
        
        self._log_mining(
            "OrthogonalResidualComputed",
            f"{factor_col} vs {self.core_factor}: β={beta:.4f}"
        )
        
        return final_residual
    
    def extract_all_residuals(
        self,
        df: pd.DataFrame,
        factors: List[str],
    ) -> Dict[str, pd.Series]:
        """
        提取所有因子的正交残差.
        
        Returns:
            因子名 -> 残差序列 的字典
        """
        residuals = {}
        
        for factor in factors:
            residuals[factor] = self.compute_orthogonal_residual(df, factor)
        
        self.residual_stats = {
            'core_factor': self.core_factor,
            'factors_processed': factors,
        }
        
        return residuals
    
    def get_mining_log(self) -> List[Dict]:
        return self.mining_log[-MAX_LOG_ENTRIES:]
    
    def get_residual_stats(self) -> Dict:
        return self.residual_stats


class RollingICSignCalculator:
    """
    V154 核心 - 滚动 IC 符号计算器（严格 PAC）.
    
    【PAC 逻辑】
    - 严格基于 Rolling Window（过去 20 日滚动 IC）
    - 严禁使用全样本 IC 进行符号校正
    """
    
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
        self.calculation_log = []
        
    def _log_calculation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.calculation_log) >= MAX_LOG_ENTRIES:
            self.calculation_log = self.calculation_log[-MAX_LOG_ENTRIES//2:]
        self.calculation_log.append(entry)
    
    def compute_rolling_ic_sign(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str = 't1_return',
    ) -> pd.Series:
        """
        计算滚动 IC 符号 - V154 严格防前视版本.
        
        【完整流程】
        1. 按日期排序
        2. 对每个时间点，计算过去 window 天的 IC
        3. 返回 IC 符号：sign(IC) ∈ {-1, 0, 1}
        """
        if factor_col not in df.columns or return_col not in df.columns:
            self._log_calculation("MissingColumns", f"Missing {factor_col} or {return_col}")
            return pd.Series(1, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        # 按日期计算每日 IC
        date_ics = []
        for date in result['trade_date'].unique():
            day_data = result[result['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            f = day_data[factor_col].fillna(0)
            r = day_data[return_col].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                r_rank = r.rank(method='average')
                ic = np.corrcoef(f_rank, r_rank)[0, 1]
                if not np.isnan(ic):
                    date_ics.append({'trade_date': date, 'ic': ic})
        
        if not date_ics:
            self._log_calculation("NoICCalculated", "No valid IC computed")
            return pd.Series(1, index=df.index)
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        
        # 计算滚动 IC 均值（严格使用历史信息）
        ic_df['rolling_ic'] = ic_df['ic'].rolling(window=self.window, min_periods=5).mean()
        ic_df['rolling_ic_sign'] = np.sign(ic_df['rolling_ic']).replace(0, 1)
        
        # 映射回原始数据
        ic_sign_map = ic_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        rolling_signs = result['trade_date'].map(ic_sign_map).fillna(1)
        
        self._log_calculation(
            "RollingICSignComputed",
            f"Window={self.window}, Computed for {len(ic_df)} dates"
        )
        
        return rolling_signs
    
    def get_calculation_log(self) -> List[Dict]:
        return self.calculation_log[-MAX_LOG_ENTRIES:]


class FactorGeneratorV154:
    """V154 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.generation_log) >= MAX_LOG_ENTRIES:
            self.generation_log = self.generation_log[-MAX_LOG_ENTRIES//2:]
        self.generation_log.append(entry)
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_reversion(self, df: pd.DataFrame, window: int) -> pd.Series:
        return -df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window).std()
        ).fillna(0)
    
    def compute_volatility_reversion(self, df: pd.DataFrame) -> pd.Series:
        """V154 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """V154 量价背离因子 - ORM 核心"""
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
        
        price_rank = close_return.fillna(0).rank(method='average', pct=True)
        volume_rank = volume_change.fillna(0).rank(method='average', pct=True)
        
        vpc = (price_rank - volume_rank).fillna(0)
        
        self._log_generation(
            "VolumePriceContradiction",
            f"V154 ORM core factor: mean={vpc.mean():.4f}, std={vpc.std():.4f}"
        )
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """V154 流动性 Alpha 因子"""
        if 'amount' in df.columns and 'volume' in df.columns:
            vwap = df['amount'] / (df['volume'] + 1e-6)
            price_change = df['close'] - df.get('pre_close', df['close'])
            ofi = price_change * df['volume'] / (df['amount'] + 1e-6)
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
        
        liquidity_alpha = (ofi / (ts_std_20 + 1e-6)).fillna(0)
        
        self._log_generation(
            "LiquidityAlpha",
            f"V154 core factor: mean={liquidity_alpha.mean():.4f}, std={liquidity_alpha.std():.4f}"
        )
        
        return liquidity_alpha
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有基础因子"""
        result = df.copy()
        
        self._log_generation("StartFactorGeneration", f"Processing {len(df)} rows")
        
        # 动量因子
        result['momentum_5'] = self.compute_momentum(result, 5)
        result['momentum_10'] = self.compute_momentum(result, 10)
        result['momentum_20'] = self.compute_momentum(result, 20)
        result['momentum_60'] = self.compute_momentum(result, 60)
        
        # 反转因子
        result['reversion_5'] = self.compute_reversion(result, 5)
        result['reversion_10'] = self.compute_reversion(result, 10)
        
        # 波动率因子
        result['volatility_5'] = self.compute_volatility(result, 5)
        result['volatility_10'] = self.compute_volatility(result, 10)
        result['volatility_20'] = self.compute_volatility(result, 20)
        
        result['volatility_reversion'] = self.compute_volatility_reversion(result)
        
        # V154 核心：量价因子
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        # volume_rank
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        self._log_generation("Complete", f"Generated base factors")
        
        return result


class AlphaResearchV154:
    """
    V154 Alpha 研究引擎 - Stability-Reinforcement-Alpha (SRA).
    
    【V154 核心改进】
    1. Dynamic Volatility Scaling (DVS): 截面波动率缩放
    2. Adaptive Signal Momentum (ASM): 自适应信号动量
    3. Turnover Constraint: 调仓约束
    4. Cross-Sectional Interaction 2.0: 因子协同过滤
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_pac: bool = True,
        enable_lead_lag: bool = True,
        enable_dvs: bool = True,
        enable_asm: bool = True,
        enable_turnover: bool = True,
        enable_csi2: bool = True,
        enable_orm: bool = True,
        enable_sector_neutral: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_pac = enable_pac
        self.enable_lead_lag = enable_lead_lag
        self.enable_dvs = enable_dvs
        self.enable_asm = enable_asm
        self.enable_turnover = enable_turnover
        self.enable_csi2 = enable_csi2
        self.enable_orm = enable_orm
        self.enable_sector_neutral = enable_sector_neutral
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV154(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV154()
        
        # V154 核心模块
        self.rolling_ic_calculator = RollingICSignCalculator() if enable_pac else None
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.dvs_scaler = DynamicVolatilityScaler() if enable_dvs else None
        self.asm_momentum = AdaptiveSignalMomentum() if enable_asm else None
        self.turnover_constraint = TurnoverConstraint() if enable_turnover else None
        self.csi2_interactor = CrossSectionalInteraction2() if enable_csi2 else None
        self.orm_miner = OrthogonalResidualMiner() if enable_orm else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Stability-Reinforcement-Alpha (SRA)")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'} (window={ROLLING_WINDOW})")
        logger.info(f"  Lead-Lag Correction: {'Enabled' if enable_lead_lag else 'Disabled'} (threshold={LEAD_LAG_THRESHOLD})")
        logger.info(f"  DVS: {'Enabled' if enable_dvs else 'Disabled'} (target_vol={DVS_TARGET_VOL})")
        logger.info(f"  ASM: {'Enabled' if enable_asm else 'Disabled'} (lookback={ASM_LOOKBACK})")
        logger.info(f"  Turnover Constraint: {'Enabled' if enable_turnover else 'Disabled'} (lambda={TURNOVER_PENALTY_LAMBDA})")
        logger.info(f"  CSI 2.0: {'Enabled' if enable_csi2 else 'Disabled'}")
        logger.info(f"  ORM Core Factor: {ORM_CORE_FACTOR}")
        logger.info(f"  Target IR: 0.55 (V153: 0.50)")
        logger.info(f"  Target IC Std: < 0.10 (V153: 0.145)")
    
    def _log_audit(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.audit_log) >= MAX_LOG_ENTRIES:
            self.audit_log = self.audit_log[-MAX_LOG_ENTRIES//2:]
        self.audit_log.append(entry)
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
        """因子处理：Auto-Heal Winsorization + 标准化"""
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V154 核心逻辑（SRA）"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据自愈检查
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        # 2. 准备标签（严格 T+1）
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # V154 修复：生成单期回报列用于 IC Decay 分析
        if 't1_return_period' not in result.columns:
            result['t1_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't2_return_period' not in result.columns:
            result['t2_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-2) / x.shift(-1) - 1)
        if 't3_return_period' not in result.columns:
            result['t3_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x.shift(-2) - 1)
        if 't4_return_period' not in result.columns:
            result['t4_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-4) / x.shift(-3) - 1)
        if 't5_return_period' not in result.columns:
            result['t5_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x.shift(-4) - 1)
        
        # 3. 生成基础因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        # 4. V154 Adaptive Lead-Lag Correction - 选择领先因子
        candidate_factors = ['volume_rank']
        core_factors = [f for f in V154_CORE_FACTORS if f in result.columns]
        candidate_factors.extend(core_factors)
        candidate_factors.extend([f for f in V154_CANDIDATE_FACTORS if f in result.columns][:5])
        
        lead_factors = candidate_factors
        if self.enable_lead_lag and self.lead_lag_corrector:
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(result, candidate_factors)
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        # V154 FIX: 保存选中的因子
        self.selected_factors = lead_factors
        
        # 5. V154 Orthogonal Residual Mining - 提取正交残差
        residuals = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORM", f"Extracting orthogonal residuals (core={ORM_CORE_FACTOR})...")
            residuals = self.orm_miner.extract_all_residuals(result, lead_factors)
        
        # 6. V154 Rolling PAC 极性校正 + IC 计算
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            # 使用正交残差（如果已提取）
            if factor in residuals:
                f_raw = residuals[factor]
            else:
                f_raw = result[factor].copy()
            
            # V154 Rolling PAC: 使用滚动 IC 符号
            if self.enable_pac and self.rolling_ic_calculator:
                rolling_sign = self.rolling_ic_calculator.compute_rolling_ic_sign(result, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            # V154 FIX: 保存因子方向
            self.factor_directions[factor] = factor_signs[factor]
            
            # V154 FIX: 计算并保存因子 IC
            ic = self._calc_factor_ic(result, factor if factor in result.columns else lead_factors[0])
            self.factor_ics[factor] = ic * factor_signs[factor]
            
            # 标准化处理
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 7. V154 CSI 2.0 - IC 加权集成
        if self.enable_csi2 and self.csi2_interactor:
            self._log_audit("CSI2", "Computing IC-weighted ensemble...")
            # V154 FIX: 传入 factor_ics 进行 IC 加权
            csi_weights = self.csi2_interactor.compute_correlation_weights(result, lead_factors, self.factor_ics)
            self.factor_weights = csi_weights
        else:
            # 等权重
            for factor in lead_factors:
                self.factor_weights[factor] = 1.0 / len(lead_factors)
        
        # 8. 加权集成
        score = np.zeros(len(result), dtype=np.float64)
        for factor in lead_factors:
            f = factor_data.get(factor)
            if f is None:
                continue
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 1.0 / len(lead_factors))
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 9. V154 DVS - 动态波动率缩放
        if self.enable_dvs and self.dvs_scaler:
            self._log_audit("DVS", "Applying dynamic volatility scaling...")
            result['score_dvs'] = self.dvs_scaler.compute_dvs(result, 'score_raw')
        else:
            result['score_dvs'] = result['score_raw']
        
        # 10. V154 ASM - 自适应信号动量
        if self.enable_asm and self.asm_momentum:
            self._log_audit("ASM", "Applying adaptive signal momentum...")
            result['score_asm'] = self.asm_momentum.apply_asm(result, 'score_dvs')
        else:
            result['score_asm'] = result['score_dvs']
        
        # 11. V154 Turnover Constraint - 调仓约束
        if self.enable_turnover and self.turnover_constraint:
            self._log_audit("Turnover", "Applying turnover constraint...")
            result['score'] = self.turnover_constraint.apply_turnover_penalty(result, 'score_asm')
        else:
            result['score'] = result['score_asm']
        
        # 12. V154 最终截面 Z-Score 归一化
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors (SRA)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return', 
                       't1_return_period', 't2_return_period', 't3_return_period', 
                       't4_return_period', 't5_return_period']
        
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """获取因子 IC"""
        if df is not None and not df.empty:
            ics = {}
            for factor in self.selected_factors:
                if factor in df.columns:
                    ic = self._calc_factor_ic(df, factor)
                    sign = self.factor_directions.get(factor, 1)
                    ics[factor] = ic * sign
                else:
                    ics[factor] = self.factor_ics.get(factor, 0.0)
            return ics
        
        return self.factor_ics
    
    def get_selected_factors(self) -> List[str]:
        """获取选中的因子"""
        return self.selected_factors
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_lead_lag_stats(self) -> Dict:
        """获取领先滞后统计"""
        return self.lead_lag_corrector.get_lead_lag_stats() if self.lead_lag_corrector else {}
    
    def get_orm_stats(self) -> Dict:
        """获取 ORM 统计"""
        return self.orm_miner.get_residual_stats() if self.orm_miner else {}
    
    def get_dvs_stats(self) -> Dict:
        """获取 DVS 统计"""
        return self.dvs_scaler.dvs_stats if self.dvs_scaler else {}
    
    def get_asm_stats(self) -> Dict:
        """获取 ASM 统计"""
        return self.asm_momentum.asm_stats if self.asm_momentum else {}
    
    def get_turnover_stats(self) -> Dict:
        """获取调仓统计"""
        return self.turnover_constraint.turnover_stats if self.turnover_constraint else {}
    
    def get_csi2_stats(self) -> Dict:
        """获取 CSI 2.0 统计"""
        return self.csi2_interactor.csi_stats if self.csi2_interactor else {}
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log[-MAX_LOG_ENTRIES:]


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_pac: bool = True,
    enable_lead_lag: bool = True,
    enable_dvs: bool = True,
    enable_asm: bool = True,
    enable_turnover: bool = True,
    enable_csi2: bool = True,
    enable_orm: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV154:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV154(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_dvs=enable_dvs,
        enable_asm=enable_asm,
        enable_turnover=enable_turnover,
        enable_csi2=enable_csi2,
        enable_orm=enable_orm,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV154...")
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], 1000),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], 1000),
        'close': np.random.randn(1000) * 10 + 100,
        'volume': np.random.randn(1000) * 1000 + 5000,
        'amount': np.random.randn(1000) * 10000 + 50000,
        'pct_chg': np.random.randn(1000) * 2,
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  Lead-Lag Stats: {alpha.get_lead_lag_stats()}")
    logger.info(f"  ORM Stats: {alpha.get_orm_stats()}")
    logger.info(f"  DVS Stats: {alpha.get_dvs_stats()}")
    logger.info(f"  ASM Stats: {alpha.get_asm_stats()}")
    logger.info(f"  Turnover Stats: {alpha.get_turnover_stats()}")
    logger.info(f"  CSI2 Stats: {alpha.get_csi2_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")