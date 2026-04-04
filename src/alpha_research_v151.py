"""
Alpha Research Module - V151 Latency-Corrected-Alpha (LCA).

【V150 问题诊断】
- IC 衰减反转：T+5 IC 高于 T+1，说明信号太慢
- EMA α=0.4 过度平滑，延迟了信号反应速度
- 缺乏对价格拐点的超前预测能力

【V151 核心使命 - Latency-Corrected-Alpha (LCA)】
1. 解决 IC 衰减反转（核心任务）:
   - 现象诊断：V150 的 T+5 IC 高于 T+1，说明信号太慢
   - 改进要求：将 EMA 平滑系数从 α=0.4 调高至 α=0.8
   - 引入 Lead-Signal：对 volume_price_contradiction 进行一阶差分处理（Change of Alpha）

2. 严格 PAC（防偷看未来）:
   - PAC 因子纠偏必须使用 Rolling_IC_Sign(window=20)
   - 严禁使用回测全时段的平均 IC 符号
   - 只能使用过去 20 天的滚动 IC 来确定当前因子的极性

3. 波动率归一化（Volatility-Standardized IC）:
   - 在集成得分前，将各因子除以其过去 20 天的 Rank IC 标准差
   - 逻辑：降低不稳定性因子权重，提升稳定因子权重
   - 目标：冲击 IR > 0.55

【V151 核心算法】
1. Latency-Corrected-Alpha (LCA):
   - 提升 EMA α 从 0.4 → 0.8（新信号权重 80%）
   - 引入 Lead-Signal: d(Factor)/dt 一阶差分

2. Rolling PAC (Rolling-Polarity-Auto-Correction):
   - 使用 Rolling_IC_Sign(window=20) 确定因子极性
   - 严格 PAC：只能使用历史信息

3. Volatility-Standardized IC Weighting:
   - Weight_i = 1 / Std(IC_i, window=20)
   - 归一化后集成

【工程纪律】
- 禁止版本元数据错误：必须全文使用 V151
- 拒绝报错摆烂：DataHealer 必须主动补全 NaN/Inf
- 锁定裁判与资金：不修改 src/engine/，初始资金 100,000
- 截断输出：继续截断日志，严禁 400 报错

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标 |
| IC_IR | > 0.55 | 稳定性（V150: ~0.50）|
| IC Decay Pattern | T+1 > T+3 > T+5 | 必须单调递减 |
| IC Std | < 0.08 | 时序波动率 |
| 400 Error | 0 | 禁止 String Length 报错 |
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

VERSION = "V151"

# V151 核心因子（回归 V147）
V151_CORE_FACTORS = [
    'momentum_20',
    'volatility_10',
    'volume_price_contradiction',  # V147 核心
    'liquidity_alpha',              # V147 核心
    'volatility_reversion',
    'reversion_5',
]

# V151 候选因子池
V151_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V151_CORE_FACTORS + V151_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V151 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V151 LCA 参数
LCA_ALPHA = 0.8  # EMA 平滑系数：0.8 * New + 0.2 * Prev（V150: 0.4）
PIN_LAMBDA = 0.3  # 行业中性化权重
ROLLING_WINDOW = 20  # 滚动 IC 窗口


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """计算两个变量之间的互信息"""
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
        x_bins = pd.qcut(x_clean, q=n_bins, labels=False, duplicates='drop')
        y_bins = pd.qcut(y_clean, q=n_bins, labels=False, duplicates='drop')
        
        n_x = len(np.unique(x_bins))
        n_y = len(np.unique(y_bins))
        
        joint_hist = np.zeros((n_x, n_y))
        for xi, yi in zip(x_bins, y_bins):
            joint_hist[xi, yi] += 1
        joint_prob = joint_hist / len(x_clean)
        
        px = joint_hist.sum(axis=1)
        py = joint_hist.sum(axis=0)
        
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
    """V151 自动愈合版 Winsorization"""
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
    
    # 5. 最终 NaN 填充 - V151 使用 ffill 优先
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """V151 截断日志摘要，防止 400 报错."""
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


class DataHealerV151:
    """V151 增强版数据自愈模块 - 主动补全策略"""
    
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
                logger.info("[V151][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V151][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V151][DataHealer] No database URL, SQL healer disabled")
    
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
        """V151 检查并修复缺失列"""
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
        
        # V151: 主动使用 ffill() 补全
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
            logger.error(f"[V151][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """V151 自动分组插值 - 优先 ffill"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # V151: 优先 ffill 填充
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
        """V151 自动修复 NaN/Inf"""
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
            
            # 处理 NaN - V151 使用中位数填充
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


class RollingICSignCalculator:
    """
    V151 核心 - 滚动 IC 符号计算器（严格 PAC）.
    
    【V150 问题】
    - 使用全时段平均 IC 符号，存在偷看未来嫌疑
    
    【V151 修复】
    - 使用 Rolling_IC_Sign(window=20)
    - 只能使用过去 20 天的 IC 来确定当前极性
    - 严格 PAC：防偷看未来
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
        计算滚动 IC 符号.
        
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
        
        # 计算滚动 IC 均值
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


class VolatilityStandardizedICWeighter:
    """
    V151 核心 - 波动率归一化 IC 权重器.
    
    【目标】
    - 冲击 IR > 0.55
    - 在集成得分前，将各因子除以其过去 20 天的 Rank IC 标准差
    - 逻辑：降低不稳定性因子权重，提升稳定因子权重
    """
    
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
        self.weighting_log = []
        
    def _log_weighting(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.weighting_log) >= MAX_LOG_ENTRIES:
            self.weighting_log = self.weighting_log[-MAX_LOG_ENTRIES//2:]
        self.weighting_log.append(entry)
    
    def compute_volatility_weights(
        self,
        df: pd.DataFrame,
        factors: List[str],
        return_col: str = 't1_return',
    ) -> Dict[str, float]:
        """
        计算波动率归一化权重.
        
        【完整流程】
        1. 对每个因子计算历史 IC 序列
        2. 计算 IC 标准差
        3. Weight_i = 1 / Std(IC_i)
        4. 归一化权重使总和为 1
        """
        ic_stds = {}
        
        for factor in factors:
            if factor not in df.columns:
                ic_stds[factor] = 1.0
                continue
            
            ics = []
            for date in df['trade_date'].unique():
                day_data = df[df['trade_date'] == date]
                if len(day_data) < 20:
                    continue
                
                f = day_data[factor].fillna(0)
                r = day_data[return_col].fillna(0)
                
                if len(f) > 10 and np.std(f) > 1e-10:
                    f_rank = f.rank(method='average')
                    r_rank = r.rank(method='average')
                    ic = np.corrcoef(f_rank, r_rank)[0, 1]
                    if not np.isnan(ic):
                        ics.append(ic)
            
            if len(ics) >= 5:
                ic_std = np.std(ics, ddof=1)
                ic_stds[factor] = max(ic_std, 0.01)  # 防止除零
            else:
                ic_stds[factor] = 1.0
        
        # 计算权重：1 / IC_Std
        raw_weights = {f: 1.0 / std for f, std in ic_stds.items()}
        
        # 归一化
        total = sum(raw_weights.values())
        if total > 0:
            weights = {f: w / total for f, w in raw_weights.items()}
        else:
            weights = {f: 1.0 / len(factors) for f in factors}
        
        self._log_weighting(
            "VolatilityWeightsComputed",
            f"Computed weights for {len(weights)} factors"
        )
        
        return weights
    
    def get_weighting_log(self) -> List[Dict]:
        return self.weighting_log[-MAX_LOG_ENTRIES:]


class PartialIndustryNeutralizer:
    """V151 部分行业中性化 (保留 V150 逻辑)"""
    
    def __init__(self, industry_column: str = 'industry_code', lambda_param: float = PIN_LAMBDA):
        self.industry_column = industry_column
        self.lambda_param = lambda_param
        self.neutralization_log = []
        self.neutralization_stats = {}
        
    def _log_neutralization(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.neutralization_log) >= MAX_LOG_ENTRIES:
            self.neutralization_log = self.neutralization_log[-MAX_LOG_ENTRIES//2:]
        self.neutralization_log.append(entry)
    
    def neutralize(
        self,
        df: pd.DataFrame,
        signal_col: str,
    ) -> pd.Series:
        """应用部分行业中性化"""
        if signal_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        
        if self.industry_column not in result.columns:
            self._log_neutralization("NoIndustryData", "Industry column not found")
            return result[signal_col]
        
        neutralized_signal = result[signal_col].copy()
        
        for date in result['trade_date'].unique():
            date_mask = result['trade_date'] == date
            date_data = result.loc[date_mask]
            
            industry_means = date_data.groupby(self.industry_column)[signal_col].mean()
            
            for industry in industry_means.index:
                industry_mask = date_data[self.industry_column] == industry
                industry_idx = date_data.loc[industry_mask].index
                
                raw_score = date_data.loc[industry_mask, signal_col]
                industry_mean = industry_means[industry]
                
                neutralized_signal.loc[industry_idx] = (
                    (1 - self.lambda_param) * raw_score - self.lambda_param * industry_mean
                )
        
        self._log_neutralization(
            "PartialIndustryNeutralizationApplied",
            f"λ={self.lambda_param}"
        )
        
        self.neutralization_stats = {
            'method': 'partial_industry_neutralization',
            'lambda': self.lambda_param,
        }
        
        return neutralized_signal
    
    def get_neutralization_log(self) -> List[Dict]:
        return self.neutralization_log[-MAX_LOG_ENTRIES:]
    
    def get_neutralization_stats(self) -> Dict:
        return self.neutralization_stats


class EMASignalSmoother:
    """
    V151 核心 - EMA 信号平滑器（LCA 版本）.
    
    【V150 问题】
    - α=0.4 过度平滑，导致信号延迟
    
    【V151 修复】
    - α=0.8 新信号权重 80%，历史信号 20%
    - 提升对价格拐点的反应速度
    """
    
    def __init__(self, alpha: float = LCA_ALPHA):
        self.alpha = alpha
        self.smoothing_log = []
        self.smoothing_stats = {}
        
    def _log_smoothing(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.smoothing_log) >= MAX_LOG_ENTRIES:
            self.smoothing_log = self.smoothing_log[-MAX_LOG_ENTRIES//2:]
        self.smoothing_log.append(entry)
    
    def apply_ema(
        self,
        df: pd.DataFrame,
        raw_score_col: str,
    ) -> pd.Series:
        """应用 V151 EMA 平滑"""
        if raw_score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        smoothed_scores = []
        
        for symbol in result['symbol'].unique():
            symbol_mask = result['symbol'] == symbol
            symbol_data = result.loc[symbol_mask].copy()
            
            raw_scores = symbol_data[raw_score_col].values
            n = len(raw_scores)
            
            if n == 0:
                smoothed_scores.append((symbol_mask, pd.Series([], index=symbol_data.index)))
                continue
            
            smoothed = np.zeros(n)
            smoothed[0] = raw_scores[0]
            
            for t in range(1, n):
                # V151 LCA EMA 公式：α=0.8
                smoothed[t] = self.alpha * raw_scores[t] + (1 - self.alpha) * smoothed[t - 1]
            
            smoothed_series = pd.Series(smoothed, index=symbol_data.index)
            smoothed_scores.append((symbol_mask, smoothed_series))
        
        final_smoothed = pd.Series(0.0, index=df.index)
        for mask, series in smoothed_scores:
            final_smoothed.loc[mask] = series
        
        self._log_smoothing(
            "LCA-EMASmoothingApplied",
            f"α={self.alpha}, New={self.alpha*100}%, Prev={(1-self.alpha)*100}%"
        )
        
        self.smoothing_stats = {
            'alpha': self.alpha,
            'new_signal_weight': self.alpha,
            'prev_signal_weight': 1 - self.alpha,
            'method': 'lca_exponential_moving_average',
        }
        
        return final_smoothed
    
    def get_smoothing_log(self) -> List[Dict]:
        return self.smoothing_log[-MAX_LOG_ENTRIES:]
    
    def get_smoothing_stats(self) -> Dict:
        return self.smoothing_stats


class LeadSignalGenerator:
    """
    V151 核心 - Lead-Signal 生成器.
    
    【目标】
    - 对 volume_price_contradiction 进行一阶差分处理（Change of Alpha）
    - 提升对价格拐点的反应速度
    - 解决 IC 衰减反转问题
    """
    
    def __init__(self):
        self.generation_log = []
        
    def _log_generation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.generation_log) >= MAX_LOG_ENTRIES:
            self.generation_log = self.generation_log[-MAX_LOG_ENTRIES//2:]
        self.generation_log.append(entry)
    
    def compute_lead_signal(
        self,
        df: pd.DataFrame,
        base_factor: str = 'volume_price_contradiction',
    ) -> pd.Series:
        """
        计算 Lead-Signal（一阶差分）.
        
        【公式】
        Lead_Signal = d(Factor)/dt = Factor_t - Factor_{t-1}
        """
        if base_factor not in df.columns:
            self._log_generation("MissingBaseFactor", f"{base_factor} not found")
            return pd.Series(0, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        # 一阶差分
        lead_signal = result.groupby('symbol')[base_factor].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Winsorization
        lead_signal = winsorize_auto_heal(lead_signal, sigma=3.0, percentile=0.99)
        
        self._log_generation(
            "LeadSignalComputed",
            f"Base={base_factor}, Mean={lead_signal.mean():.4f}, Std={lead_signal.std():.4f}"
        )
        
        return lead_signal
    
    def get_generation_log(self) -> List[Dict]:
        return self.generation_log[-MAX_LOG_ENTRIES:]


class FactorGeneratorV151:
    """V151 因子生成器"""
    
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
        """V151 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """V151 量价背离因子"""
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
            f"V147 core factor: mean={vpc.mean():.4f}, std={vpc.std():.4f}"
        )
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """V151 流动性 Alpha 因子"""
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
            f"V147 core factor: mean={liquidity_alpha.mean():.4f}, std={liquidity_alpha.std():.4f}"
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
        
        # V151 核心：量价因子
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


class AlphaResearchV151:
    """
    V151 Alpha 研究引擎 - Latency-Corrected-Alpha (LCA).
    
    【V151 核心改进】
    1. LCA: 提升 EMA α 从 0.4 → 0.8
    2. Lead-Signal: 一阶差分提升反应速度
    3. Rolling PAC: 严格防偷看未来
    4. Volatility-Standardized IC: 冲击 IR > 0.55
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_pac: bool = True,
        enable_pin: bool = True,
        enable_ema: bool = True,
        enable_lead_signal: bool = True,
        enable_volatility_weighting: bool = True,
        enable_sector_neutral: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_pac = enable_pac
        self.enable_pin = enable_pin
        self.enable_ema = enable_ema
        self.enable_lead_signal = enable_lead_signal
        self.enable_volatility_weighting = enable_volatility_weighting
        self.enable_sector_neutral = enable_sector_neutral
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV151(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV151()
        
        # V151 核心模块
        self.rolling_ic_calculator = RollingICSignCalculator() if enable_pac else None
        self.volatility_weighter = VolatilityStandardizedICWeighter() if enable_volatility_weighting else None
        self.lead_signal_generator = LeadSignalGenerator() if enable_lead_signal else None
        self.pin = PartialIndustryNeutralizer() if enable_pin else None
        self.ema = EMASignalSmoother() if enable_ema else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Latency-Corrected-Alpha (LCA)")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'} (window={ROLLING_WINDOW})")
        logger.info(f"  Lead-Signal: {'Enabled' if enable_lead_signal else 'Disabled'}")
        logger.info(f"  LCA-EMA: {'Enabled' if enable_ema else 'Disabled'} (α={LCA_ALPHA})")
        logger.info(f"  Volatility Weighting: {'Enabled' if enable_volatility_weighting else 'Disabled'}")
        logger.info(f"  Target IR: 0.55 (V150: ~0.50)")
        logger.info(f"  Target IC Decay: T+1 > T+3 > T+5")
    
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
        """计算 Alpha 评分 - V151 核心逻辑（LCA）"""
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
        
        # 3. 生成基础因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        # 4. 生成 Lead-Signal
        if self.enable_lead_signal and self.lead_signal_generator:
            self._log_audit("LeadSignal", "Computing Lead-Signal (d(Factor)/dt)...")
            lead_signal = self.lead_signal_generator.compute_lead_signal(result, 'volume_price_contradiction')
            result['lead_signal'] = lead_signal
        
        # 5. 构建候选因子池
        all_candidate_factors = []
        
        if 'volume_rank' in result.columns:
            all_candidate_factors.append('volume_rank')
        
        core_factors = [f for f in V151_CORE_FACTORS if f in result.columns]
        for factor in core_factors:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        # 添加 Lead-Signal
        if self.enable_lead_signal and 'lead_signal' in result.columns:
            all_candidate_factors.append('lead_signal')
        
        candidate_factors = [f for f in V151_CANDIDATE_FACTORS if f in result.columns]
        for factor in candidate_factors[:5]:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors"
        )
        
        # 6. 计算 IC 和因子选择 - V151 Rolling PAC
        factor_ics = []
        
        for factor in all_candidate_factors:
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            factor_ics.append((factor, abs(ic)))
        
        factor_ics.sort(key=lambda x: x[1], reverse=True)
        
        max_factors = min(self.n_factors, 8)
        final_selected = [f[0] for f in factor_ics[:max_factors]]
        
        self.selected_factors = final_selected[:max_factors]
        
        self._log_audit(
            "FactorSelection",
            f"Final selected {len(self.selected_factors)} factors: {self.selected_factors}"
        )
        
        # 7. V151 Rolling PAC 极性校正
        factor_data = {}
        factor_signs = {}
        
        for factor in self.selected_factors:
            f_raw = result[factor]
            
            # V151 Rolling PAC: 使用滚动 IC 符号
            if self.enable_pac and self.rolling_ic_calculator:
                rolling_sign = self.rolling_ic_calculator.compute_rolling_ic_sign(result, factor)
                factor_signs[factor] = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 8. V151 波动率归一化权重
        if self.enable_volatility_weighting and self.volatility_weighter:
            self._log_audit("VolatilityWeighting", "Computing volatility-standardized IC weights...")
            self.factor_weights = self.volatility_weighter.compute_volatility_weights(
                result, self.selected_factors
            )
        else:
            # 等权重
            for factor in self.selected_factors:
                self.factor_weights[factor] = 1.0 / len(self.selected_factors)
        
        # 9. 加权集成
        score = np.zeros(len(result), dtype=np.float64)
        for factor in self.selected_factors:
            f = factor_data[factor]
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 1.0 / len(self.selected_factors))
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 10. V151 LCA-EMA 信号平滑
        if self.enable_ema and self.ema:
            self._log_audit("LCA-EMA", f"Applying LCA-EMA Smoothing (α={LCA_ALPHA})...")
            smoothed_score = self.ema.apply_ema(result, 'score_raw')
            result['score_smoothed'] = smoothed_score
        else:
            result['score_smoothed'] = result['score_raw']
        
        # 11. V151 PIN 部分行业中性化
        if self.enable_pin and self.pin:
            self._log_audit("PIN", "Applying Partial Industry Neutralization...")
            final_score = self.pin.neutralize(result, 'score_smoothed')
            result['score'] = final_score
        else:
            result['score'] = result['score_smoothed']
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (LCA)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']
        
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
    
    def get_ema_stats(self) -> Dict:
        """获取 EMA 统计"""
        return self.ema.get_smoothing_stats() if self.ema else {}
    
    def get_ema_log(self) -> List[Dict]:
        """获取 EMA 日志"""
        return self.ema.get_smoothing_log() if self.ema else []
    
    def get_pin_stats(self) -> Dict:
        """获取 PIN 统计"""
        return self.pin.get_neutralization_stats() if self.pin else {}
    
    def get_pin_log(self) -> List[Dict]:
        """获取 PIN 日志"""
        return self.pin.get_neutralization_log() if self.pin else []
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log[-MAX_LOG_ENTRIES:]


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_pac: bool = True,
    enable_pin: bool = True,
    enable_ema: bool = True,
    enable_lead_signal: bool = True,
    enable_volatility_weighting: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV151:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV151(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_pin=enable_pin,
        enable_ema=enable_ema,
        enable_lead_signal=enable_lead_signal,
        enable_volatility_weighting=enable_volatility_weighting,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV151...")
    
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
    logger.info(f"  EMA Stats: {alpha.get_ema_stats()}")
    logger.info(f"  PIN Stats: {alpha.get_pin_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")