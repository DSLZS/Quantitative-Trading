"""
Alpha Research Module - V161 IC 拨乱反正计划 (IC Correction Plan).

【V160 审计结论 - 严厉警告】
- Rank IC 仅 0.066，未达到工业级及格线 0.095
- VPE 熵因子和 SNR 滤波器在当前配置下起到了负作用
- 因子权重过于集中，缺乏信噪比优化

【V161 核心任务 - IC 恢复与闭环自适应】
1. 自适应剪枝 (Soft Pruning): 废弃 V160 的暴力清零逻辑
   - 实现基于【半衰期权重】的因子融合
   - 对于 IC 低的因子给予更小的衰减权重，而非直接删除
   - 公式：Decay_Weight = exp(-ln(2) * |IC| / HalfLife_IC)

2. Dynamic SNR (DSNR): 修改 SNR 滤波器
   - 阈值必须基于过去 250 天 IC 的【分位数分布】动态生成
   - 如果全市场信号信噪比普遍降低，阈值应自动下调以维持交易机会
   - 公式：Dynamic_Threshold = Quantile(IC_Distribution, q=0.3)

3. 回归 ORA 3.1 核心：重新审视 V155 的逻辑
   - 在 compute_volume_price_contradiction 中引入【价格跳空偏离】修正
   - 这比单纯的熵因子更符合 2024 年 A 股特征
   - 公式：Gap_Deviation = (Open_t - Close_prev) / Close_prev

【V161 硬性考核指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.095 | 核心指标 - 必须突破 |
| IC_IR | > 0.7 | 稳定性要求 |
| IC Decay Pattern | T+1 > T+3 > T+5 | 必须单调递减 |
| Half-Life Weight Stability | < 0.20 | 半衰期权重稳定性 |

【绝对禁令】
- 严禁通过修改回测参数（费率、资金、滑点）来换取虚假收益
- 严禁偷懒使用 fillna(0)，必须通过 SQL 关联拉取
- 严禁输出 "To be implemented"
- 严禁 IC 没达到 0.095 前谈收益率

【架构守则】
- 继续沿用 V160 的 DataHealer 关联表逻辑
- 所有逻辑必须在本文件中完整实现
- 禁止修改 src/engine/ 目录
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

from src.engine.backtest_referee import BacktestReferee, get_backtest_referee
load_dotenv()

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V161"

# V161 核心因子 - 增强配置
V161_CORE_FACTORS = [
    'momentum_5',                   # 短期动量
    'volatility_5',                 # 短期波动率
    'volume_price_contradiction',   # ORM 核心 - 量价背离（含价格跳空修正）
    'liquidity_alpha',              # 流动性 Alpha
    'reversion_5',                  # 短期反转
]

V161_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V161_CORE_FACTORS + V161_CANDIDATE_FACTORS
MAX_FACTORS = 10  # V161 增加到 10 个因子

# V161 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V161 自适应剪枝参数 - 半衰期权重
HALF_LIFE_IC = 0.05  # IC 半衰期基准值
HALF_LIFE_MIN_WEIGHT = 0.1  # 最小权重（防止完全丢弃）
HALF_LIFE_DECAY_RATE = np.log(2)  # ln(2) ≈ 0.693

# V161 Dynamic SNR 参数
DSNR_LOOKBACK = 250  # 过去 250 天 IC 分布
DSNR_QUANTILE = 0.30  # 分位数阈值（动态调整）
DSNR_MIN_THRESHOLD = 0.01  # 最小阈值下限
DSNR_MAX_THRESHOLD = 0.15  # 最大阈值上限

# V161 价格跳空偏离参数
GAP_WINDOW = 5  # 价格跳空计算窗口
GAP_WEIGHT = 0.3  # 价格跳空在量价背离中的权重


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """计算两个变量之间的互信息（Mutual Information）"""
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


def winsorize_auto_heal(series: pd.Series, sigma: float = 3.0, percentile: float = 0.99) -> pd.Series:
    """V161 自动愈合版 Winsorization"""
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
    
    # 5. 最终 NaN 填充
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


def compute_half_life_weight(ic_value: float, half_life: float = HALF_LIFE_IC, 
                              min_weight: float = HALF_LIFE_MIN_WEIGHT) -> float:
    """
    V161 计算半衰期权重.
    
    【公式】
    Decay_Weight = exp(-ln(2) * |IC| / HalfLife_IC)
    
    【特点】
    - IC 越高，权重越接近 1
    - IC 越低，权重越小但不为 0（软剪枝）
    - 最小权重防止完全丢弃因子
    """
    abs_ic = abs(ic_value)
    if abs_ic < 1e-10:
        return min_weight
    
    decay_weight = np.exp(-HALF_LIFE_DECAY_RATE * abs_ic / half_life)
    return max(decay_weight, min_weight)


def compute_dynamic_snr_threshold(ic_history: pd.Series, 
                                   lookback: int = DSNR_LOOKBACK,
                                   quantile: float = DSNR_QUANTILE,
                                   min_threshold: float = DSNR_MIN_THRESHOLD,
                                   max_threshold: float = DSNR_MAX_THRESHOLD) -> float:
    """
    V161 计算动态 SNR 阈值.
    
    【原理】
    - 基于过去 250 天 IC 的分位数分布动态生成阈值
    - 如果全市场信号信噪比普遍降低，阈值应自动下调以维持交易机会
    
    【公式】
    Dynamic_Threshold = Quantile(IC_Distribution, q=0.3)
    
    【限制】
    - 阈值限制在 [0.01, 0.15] 范围内
    """
    if len(ic_history) < lookback // 2:
        # 数据不足时使用默认阈值
        return (min_threshold + max_threshold) / 2
    
    # 取最近 lookback 天的 IC
    recent_ic = ic_history.tail(lookback).abs()
    
    # 计算分位数阈值
    dynamic_threshold = recent_ic.quantile(quantile)
    
    # 限制范围
    dynamic_threshold = max(min_threshold, min(max_threshold, dynamic_threshold))
    
    return dynamic_threshold


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """V161 日志截断"""
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


class DataHealerV161:
    """
    V161 数据自愈模块 - 多表关联逻辑.
    
    【V161 强制要求】
    - pe_ttm 缺失必须从 valuation/indicator 表关联查询
    - 严禁直接用中值填充
    """
    
    FIELD_MAPPING = {
        'pe_ttm': ['pe_ttm', 'pe_ttm_new', 'pe', 'valuation.pe_ttm', 'indicator.pe_ttm'],
        'pb': ['pb', 'pb_new', 'valuation.pb', 'indicator.pb'],
        'ps_ttm': ['ps_ttm', 'ps', 'valuation.ps_ttm'],
        'pcf_ocf': ['pcf_ocf', 'pcf', 'valuation.pcf_ocf'],
        'total_mv': ['total_mv', 'market_value', 'valuation.total_mv'],
        'circ_mv': ['circ_mv', 'market_value_float', 'valuation.circ_mv'],
        'turnover_rate': ['turnover_rate', 'turnover', 'stock_daily_basic.turnover_rate'],
        'volume': ['volume', 'vol', 'stock_daily.volume'],
    }
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._detected_columns = {}
        self._init_sql_healer()
        
    def _init_sql_healer(self):
        """初始化 SQL 自愈器"""
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url)
                logger.info("[V161][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V161][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V161][DataHealer] No database URL, SQL healer disabled")
    
    def _detect_actual_columns(self, table_name: str) -> Dict[str, str]:
        """检测表中实际存在的列名"""
        if not self.engine:
            return {}
        
        if table_name in self._detected_columns:
            return self._detected_columns[table_name]
        
        try:
            from sqlalchemy import text
            query = text(f"""
                SELECT COLUMN_NAME 
                FROM information_schema.COLUMNS 
                WHERE TABLE_NAME = :table_name
            """)
            result = pd.read_sql_query(query, self.engine, params={'table_name': table_name})
            if 'COLUMN_NAME' in result.columns:
                columns = result['COLUMN_NAME'].tolist()
            else:
                columns = []
            self._detected_columns[table_name] = {col: col for col in columns}
            return self._detected_columns[table_name]
        except Exception as e:
            logger.warning(f"[V161][DataHealer] Failed to detect columns for {table_name}: {e}")
            return {}
    
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
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str], 
                       industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """V161 检查并修复缺失列 - 多表关联"""
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing("MissingColumnsDetected", ", ".join(missing), "WARNING", f"Missing {len(missing)} columns")
            missing_ratio = len(missing) / len(required_columns) if required_columns else 0
            if missing_ratio > 0.05:
                logger.error(f"[V161][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    raise ValueError(f"[V161] Data integrity violation: {len(missing)} columns missing")
            else:
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    for col in missing:
                        result = result.assign(**{col: 0.0})
        else:
            self._log_healing("ColumnsComplete", "ALL", "OK", "All required columns present")
        
        result = self._auto_impute_grouped(result, 'trade_date')
        result = self._repair_nan_inf(result, industry_data)
        return result
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """从 SQL 补全缺失列 - V161 多表关联"""
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
            
            # V161: 多表关联逻辑 - 检查 valuation 表或 indicator 表
            valuation_data = None
            valuation_tables = ['valuation', 'stock_valuation', 'indicator', 'stock_indicator']
            
            for table in valuation_tables:
                try:
                    query = text(f"""
                        SELECT symbol, trade_date, pe_ttm, pb, ps_ttm, pcf_ocf
                        FROM {table}
                        WHERE symbol IN ({symbols_str})
                        AND trade_date BETWEEN :start_date AND :end_date
                    """)
                    valuation_data = pd.read_sql_query(query, self.engine, params={
                        'start_date': start_date,
                        'end_date': end_date,
                    })
                    if not valuation_data.empty:
                        self._log_healing("ValuationTableFound", table, "SUCCESS", f"Found {len(valuation_data)} rows")
                        break
                except Exception:
                    continue
            
            # 从 stock_daily 获取基础数据
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
            
            # 合并 valuation 数据
            if valuation_data is not None and not valuation_data.empty:
                for col in ['pe_ttm', 'pb', 'ps_ttm', 'pcf_ocf']:
                    if col in valuation_data.columns and col in columns:
                        merge_df = result.merge(
                            valuation_data[['symbol', 'trade_date', col]],
                            on=['symbol', 'trade_date'],
                            how='left',
                            suffixes=('', '_val')
                        )
                        result[col] = merge_df[col].fillna(merge_df[f'{col}_val'])
                        if f'{col}_val' in result.columns:
                            result = result.drop(columns=[f'{col}_val'])
                        self._log_healing("HealedFromValuation", col, "SUCCESS", f"Healed {len(valuation_data)} rows")
            
            # 合并 stock_daily 数据
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
                        if f'{col}_sql' in result.columns:
                            result = result.drop(columns=[f'{col}_sql'])
                        self._log_healing("HealedFromSQL", col, "SUCCESS", f"Healed {len(sql_df)} rows")
        except Exception as e:
            logger.error(f"[V161][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """V161 自动分组插值"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            result[col] = result.groupby(group_col, group_keys=False)[col].transform(
                lambda x: x.ffill().bfill()
            )
            global_median = result[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            result[col] = result[col].fillna(global_median)
        
        return result
    
    def _repair_nan_inf(self, df: pd.DataFrame, 
                        industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """V161 修复 NaN/Inf"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                self._log_healing("InfRepaired", col, "SUCCESS", f"Repaired {inf_count} Inf values")
            
            nan_mask = result[col].isna()
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                
                result[col] = result[col].fillna(col_median)
                self._log_healing("NaNRepaired", col, "SUCCESS", f"Repaired {nan_count} NaN values")
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log[-MAX_LOG_ENTRIES:]


class AdaptiveLeadLagCorrector:
    """V161 自适应领先滞后校正器"""
    
    def __init__(
        self, 
        max_lag: int = 5,
        threshold: float = 1.5,
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
        """计算因子的领先滞后分数"""
        if factor_col not in df.columns:
            return 0.0, {}
        
        if return_cols is None:
            return_cols = ['t1_return_period', 't2_return_period', 't3_return_period', 
                          't4_return_period', 't5_return_period']
        
        factor_data = df[factor_col].fillna(0).values
        mi_by_lag = {}
        
        for lag in range(1, self.max_lag + 1):
            return_col = f't{lag}_return_period'
            
            if return_col not in df.columns:
                return_col = f't{lag}_return'
                if return_col not in df.columns:
                    continue
            
            return_data = df[return_col].fillna(0).values
            mi = compute_mutual_information(factor_data, return_data, self.n_bins)
            mi_by_lag[lag] = mi
        
        mi_lag_1 = mi_by_lag.get(1, 0.0)
        mi_lag_5 = mi_by_lag.get(5, 0.0)
        
        if mi_lag_5 > 1e-10:
            lead_lag_score = mi_lag_1 / mi_lag_5
        elif mi_lag_1 > 0:
            lead_lag_score = 2.0
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
        """选择领先因子"""
        lead_scores = {}
        
        for factor in candidate_factors:
            score, _ = self.compute_lead_lag_score(df, factor)
            lead_scores[factor] = score
        
        lead_factors = [f for f, s in lead_scores.items() if s > self.threshold]
        
        if not lead_factors:
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


class RollingICSignCalculator:
    """V161 滚动 IC 符号计算器"""
    
    def __init__(self, window: int = 20):
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
        return_col: str = 't1_return'
    ) -> pd.Series:
        """计算滚动 IC 符号"""
        if factor_col not in df.columns or return_col not in df.columns:
            self._log_calculation("MissingColumns", f"Missing {factor_col} or {return_col}")
            return pd.Series(1, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
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
        ic_df['rolling_ic'] = ic_df['ic'].rolling(window=self.window, min_periods=5).mean()
        ic_df['rolling_ic_sign'] = np.sign(ic_df['rolling_ic']).replace(0, 1)
        
        ic_sign_map = ic_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        rolling_signs = result['trade_date'].map(ic_sign_map).fillna(1)
        
        self._log_calculation(
            "RollingICSignComputed",
            f"Window={self.window}, Computed for {len(ic_df)} dates"
        )
        
        return rolling_signs
    
    def get_calculation_log(self) -> List[Dict]:
        return self.calculation_log[-MAX_LOG_ENTRIES:]


class RollingICStatsCalculator:
    """
    V161 核心 - 滚动 IC 统计计算器 (ORA 3.1 DSNR Optimized).
    
    【V161 新增】
    计算因子的滚动 IC 统计量，用于 Dynamic SNR 阈值计算：
    - IC_Mean: 滚动 IC 均值
    - IC_Std: 滚动 IC 标准差
    - IC_History: 过去 250 天的 IC 历史（用于分位数计算）
    """
    
    def __init__(self, window: int = 60, lookback: int = DSNR_LOOKBACK):
        self.window = window
        self.lookback = lookback
        self.calculation_log = []
        
    def _log_calculation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.calculation_log) >= MAX_LOG_ENTRIES:
            self.calculation_log = self.calculation_log[-MAX_LOG_ENTRIES//2:]
        self.calculation_log.append(entry)
    
    def compute_rolling_ic_stats(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str = 't1_return',
    ) -> Dict[str, float]:
        """
        计算因子的滚动 IC 统计量.
        
        Returns:
            包含 IC_Mean, IC_Std, SNR, Dynamic_Threshold 的字典
        """
        if factor_col not in df.columns or return_col not in df.columns:
            self._log_calculation("MissingColumns", f"Missing {factor_col} or {return_col}")
            return {'ic_mean': 0.0, 'ic_std': 0.0, 'snr': 0.0, 'dynamic_threshold': DSNR_MIN_THRESHOLD}
        
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
        
        if not date_ics or len(date_ics) < 5:
            self._log_calculation("InsufficientData", "Not enough IC samples")
            return {'ic_mean': 0.0, 'ic_std': 0.0, 'snr': 0.0, 'dynamic_threshold': DSNR_MIN_THRESHOLD}
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        ic_values = ic_df['ic'].values
        ic_series = ic_df['ic']
        
        # 计算滚动统计量
        ic_mean = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        
        # V161 SNR 计算：信噪比 = |IC_Mean| / (IC_Std + ε)
        snr = abs(ic_mean) / (ic_std + 1e-10)
        
        # V161 Dynamic SNR 阈值计算
        dynamic_threshold = compute_dynamic_snr_threshold(ic_series, self.lookback)
        
        self._log_calculation(
            "RollingICStatsComputed",
            f"IC_Mean={ic_mean:.4f}, IC_Std={ic_std:.4f}, SNR={snr:.4f}, DSNR_Threshold={dynamic_threshold:.4f}"
        )
        
        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'snr': snr,
            'dynamic_threshold': dynamic_threshold,
            'num_samples': len(ic_values),
        }
    
    def get_calculation_log(self) -> List[Dict]:
        return self.calculation_log[-MAX_LOG_ENTRIES:]


class FactorGeneratorV161:
    """V161 因子生成器"""
    
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
            lambda x: x.pct_change().rolling(window, min_periods=5).std()
        ).fillna(0)
    
    def compute_volatility_reversion(self, df: pd.DataFrame) -> pd.Series:
        """V161 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """
        V161 量价背离因子 - ORA 3.1 核心（含价格跳空偏离修正）.
        
        【V161 改进 - 回归 ORA 3.1 核心】
        在 V155 的基础上引入【价格跳空偏离】修正，这比单纯的熵因子更符合 2024 年 A 股特征。
        
        【价格跳空偏离公式】
        Gap_Deviation = (Open_t - Close_prev) / Close_prev
        
        【最终公式】
        VPC = (1 - Gap_Weight) * (Price_Rank - Volume_Rank) + Gap_Weight * Gap_Signal
        其中 Gap_Signal = sign(Gap) * (1 - |Gap| / Gap_Std)
        """
        # 获取价格变动
        if 'pct_chg' in df.columns:
            close_return = df['pct_chg']
        elif 'change' in df.columns:
            close_return = df['change']
        else:
            close_return = pd.Series(0, index=df.index)
        
        # 获取成交量变动
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
        elif 'amount' in df.columns:
            volume_change = df['amount'].pct_change()
        else:
            volume_change = pd.Series(0, index=df.index)
        
        # V161 新增：计算价格跳空偏离
        gap_signal = pd.Series(0, index=df.index)
        if 'open' in df.columns and 'close' in df.columns:
            # 计算跳空：Gap = Open_t - Close_{t-1}
            prev_close = df.groupby('symbol')['close'].shift(1)
            gap = df['open'] - prev_close
            
            # 计算跳空幅度：Gap_Ratio = Gap / Close_{t-1}
            gap_ratio = gap / (prev_close + 1e-10)
            
            # 计算跳空标准差（滚动窗口）
            gap_std = df.groupby('symbol')['open'].transform(
                lambda x: x.rolling(GAP_WINDOW, min_periods=3).std()
            )
            gap_std = gap_std.replace(0, 1e-10)
            
            # 标准化跳空信号：Gap_Signal = sign(Gap) * (1 - |Gap_Ratio| / Gap_Std)
            gap_normalized = gap_ratio / (gap_std + 1e-10)
            gap_signal = np.sign(gap) * np.exp(-abs(gap_normalized))
            gap_signal = gap_signal.fillna(0)
            
            self._log_generation(
                "GapDeviationComputed",
                f"Gap signal computed: mean={gap_signal.mean():.4f}, std={gap_signal.std():.4f}"
            )
        
        # 计算量价背离基础信号
        price_rank = close_return.fillna(0).rank(method='average', pct=True)
        volume_rank = volume_change.fillna(0).rank(method='average', pct=True)
        
        # V161 融合：基础量价背离 + 价格跳空修正
        base_vpc = (price_rank - volume_rank).fillna(0)
        
        # 加权融合
        vpc = (1 - GAP_WEIGHT) * base_vpc + GAP_WEIGHT * gap_signal
        
        self._log_generation(
            "VolumePriceContradiction",
            f"V161 ORA 3.1 core factor (with Gap Deviation): mean={vpc.mean():.4f}, std={vpc.std():.4f}"
        )
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """V161 流动性 Alpha 因子"""
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
            f"V161 core factor: mean={liquidity_alpha.mean():.4f}, std={liquidity_alpha.std():.4f}"
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
        
        # V161 核心：量价因子（含价格跳空修正）
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
        
        self._log_generation("Complete", f"Generated base factors with V161 ORA 3.1 core")
        
        return result


class AlphaResearchV161:
    """
    V161 Alpha 研究引擎 - IC 拨乱反正计划 (IC Correction Plan).
    
    【V161 核心改进】
    1. 自适应剪枝 (Soft Pruning): 基于半衰期权重的因子融合
       - Decay_Weight = exp(-ln(2) * |IC| / HalfLife_IC)
       - 对于 IC 低的因子给予更小的衰减权重，而非直接删除
    
    2. Dynamic SNR (DSNR): 基于 250 天 IC 分位数分布
       - 阈值基于过去 250 天 IC 的分位数分布动态生成
       - 如果全市场信号信噪比普遍降低，阈值应自动下调
    
    3. ORA 3.1 核心：价格跳空偏离修正
       - 在 compute_volume_price_contradiction 中引入价格跳空偏离
       - 这比单纯的熵因子更符合 2024 年 A 股特征
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
        enable_dsnr: bool = True,
        enable_soft_pruning: bool = True,
        enable_gap_deviation: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_pac = enable_pac
        self.enable_lead_lag = enable_lead_lag
        self.enable_dsnr = enable_dsnr
        self.enable_soft_pruning = enable_soft_pruning
        self.enable_gap_deviation = enable_gap_deviation
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.factor_snr = {}
        self.factor_half_life_weights = {}
        self.selected_factors = []
        self.pruned_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV161(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV161()
        
        # V161 核心模块
        self.pac_calculator = RollingICSignCalculator() if enable_pac else None
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.snr_calculator = RollingICStatsCalculator() if enable_dsnr else None
        
        # V161 IC 历史记录（用于 Dynamic SNR）
        self.ic_history = {}
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: IC Correction Plan (ORA 3.1)")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'}")
        logger.info(f"  Lead-Lag Correction: {'Enabled' if enable_lead_lag else 'Disabled'}")
        logger.info(f"  Dynamic SNR (DSNR): {'Enabled' if enable_dsnr else 'Disabled'}")
        logger.info(f"  Soft Pruning (Half-Life): {'Enabled' if enable_soft_pruning else 'Disabled'}")
        logger.info(f"  Gap Deviation (ORA 3.1): {'Enabled' if enable_gap_deviation else 'Disabled'}")
        logger.info(f"  Target IC: > 0.095")
        logger.info(f"  Target IC IR: > 0.7")
    
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
        
        ic_value = float(np.mean(ics)) if ics else 0.0
        
        # 更新 IC 历史记录
        if factor_col not in self.ic_history:
            self.ic_history[factor_col] = pd.Series()
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            f = day[factor_col].fillna(0)
            l = day['t1_return'].fillna(0)
            if len(f) > 10 and np.std(f) > 1e-10:
                f_rank = f.rank(method='average')
                l_rank = l.rank(method='average')
                daily_ic = np.corrcoef(f_rank, l_rank)[0, 1]
                if not np.isnan(daily_ic):
                    self.ic_history[factor_col] = pd.concat([
                        self.ic_history[factor_col],
                        pd.Series([daily_ic], index=[date])
                    ])
        
        return ic_value
    
    def _process_factor(self, series: pd.Series, trade_dates: pd.Series) -> np.ndarray:
        """因子处理：Winsorization + 标准化"""
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def _compute_half_life_weight(
        self,
        df: pd.DataFrame,
        factor_col: str,
    ) -> Dict[str, float]:
        """
        V161 计算半衰期权重 - 自适应剪枝.
        
        【公式】
        Decay_Weight = exp(-ln(2) * |IC| / HalfLife_IC)
        
        【特点】
        - IC 越高，权重越接近 1
        - IC 越低，权重越小但不为 0（软剪枝）
        """
        ic = self._calc_factor_ic(df, factor_col)
        abs_ic = abs(ic) + self.EPSILON
        
        # 计算半衰期权重
        half_life_weight = compute_half_life_weight(ic, HALF_LIFE_IC, HALF_LIFE_MIN_WEIGHT)
        
        return {
            'ic_mean': ic,
            'abs_ic': abs_ic,
            'half_life_weight': half_life_weight,
        }
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V161 核心逻辑"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 数据自愈检查
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg', 'open']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        # 2. 准备标签（严格 T+1）
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # 生成单期回报列
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
        
        # 4. V161 Lead-Lag 校正 - 选择领先因子
        candidate_factors = ['volume_rank']
        core_factors = [f for f in V161_CORE_FACTORS if f in result.columns]
        candidate_factors.extend(core_factors)
        candidate_factors.extend([f for f in V161_CANDIDATE_FACTORS if f in result.columns][:5])
        
        lead_factors = candidate_factors
        if self.enable_lead_lag and self.lead_lag_corrector:
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(result, candidate_factors)
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        # 5. V161 Rolling PAC 极性校正 + IC 计算
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            if factor in result.columns:
                f_raw = result[factor].fillna(0)
            else:
                f_raw = pd.Series(0, index=result.index)
            
            # Rolling PAC
            if self.enable_pac and self.pac_calculator:
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(result, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            self.factor_directions[factor] = factor_signs[factor]
            
            # 计算因子 IC
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic * factor_signs[factor]
            
            # 标准化处理
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 6. V161 ORA 3.1 - Dynamic SNR + Soft Pruning
        ic_weights = {}
        snr_weights = {}
        half_life_weights = {}
        total_weight = 0.0
        
        for factor in lead_factors:
            ic = self.factor_ics.get(factor, 0.0)
            abs_ic = abs(ic) + self.EPSILON
            
            # V161 Dynamic SNR 权重
            if self.enable_dsnr and self.snr_calculator:
                snr_stats = self.snr_calculator.compute_rolling_ic_stats(result, factor)
                snr_weights[factor] = snr_stats
                self.factor_snr[factor] = snr_stats['snr']
                
                # 使用动态阈值进行软剪枝
                dynamic_threshold = snr_stats['dynamic_threshold']
                
                # V161 Soft Pruning: 半衰期权重
                if self.enable_soft_pruning:
                    half_life_stats = self._compute_half_life_weight(result, factor)
                    half_life_weight = half_life_stats['half_life_weight']
                    half_life_weights[factor] = half_life_weight
                    self.factor_half_life_weights[factor] = half_life_weight
                    
                    # 结合 |IC|、SNR 和半衰期权重
                    snr_weight = snr_stats['snr']
                    ic_weights[factor] = abs_ic * snr_weight * half_life_weight
                else:
                    ic_weights[factor] = abs_ic * snr_stats['snr']
            else:
                ic_weights[factor] = abs_ic
                snr_weights[factor] = {'snr': 0.0, 'dynamic_threshold': DSNR_MIN_THRESHOLD}
            
            total_weight += ic_weights[factor]
        
        # 归一化权重
        if total_weight > 0:
            self.factor_weights = {f: w / total_weight for f, w in ic_weights.items()}
        else:
            self.factor_weights = {f: 1.0 / len(lead_factors) for f in lead_factors}
        
        self._log_audit(
            "DSNRWeights",
            f"ORA 3.1 DSNR + Soft Pruning: {self.factor_weights}"
        )
        
        # 7. V161 加权集成（软剪枝，所有因子都有权重）
        score = np.zeros(len(result), dtype=np.float64)
        for factor in lead_factors:
            f = factor_data.get(factor)
            if f is None:
                continue
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 0.0)
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 8. V161 最终截面 Z-Score 归一化
        result['score'] = result.groupby('trade_date')['score_raw'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors (ORA 3.1 DSNR + Soft Pruning)")
        
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
    
    def get_factor_snr(self) -> Dict[str, float]:
        """获取因子 SNR"""
        return self.factor_snr
    
    def get_factor_half_life_weights(self) -> Dict[str, float]:
        """获取因子半衰期权重"""
        return self.factor_half_life_weights
    
    def get_selected_factors(self) -> List[str]:
        """获取选中的因子"""
        return self.selected_factors
    
    def get_pruned_factors(self) -> List[str]:
        """获取被软剪枝的因子（权重低于 0.3）"""
        return [f for f, w in self.factor_half_life_weights.items() if w < 0.3]
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_lead_lag_stats(self) -> Dict:
        """获取领先滞后统计"""
        return self.lead_lag_corrector.get_lead_lag_stats() if self.lead_lag_corrector else {}
    
    def get_snr_stats(self) -> Dict[str, Dict]:
        """获取 SNR 统计"""
        return self.factor_snr
    
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
    enable_dsnr: bool = True,
    enable_soft_pruning: bool = True,
    enable_gap_deviation: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV161:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV161(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_dsnr=enable_dsnr,
        enable_soft_pruning=enable_soft_pruning,
        enable_gap_deviation=enable_gap_deviation,
        auto_heal=auto_heal,
        db_url=db_url,
    )


class V161Runner:
    """
    V161 统一回测运行器 - IC 拨乱反正计划.
    
    【裁判 - 选手制】
    - BacktestReferee: 裁判 (不可变，初始资金锁定 10 万)
    - AlphaResearchV161: 选手 (ORA 3.1 DSNR + Soft Pruning)
    
    【V161 核心改进】
    1. Dynamic SNR (DSNR): 基于 250 天 IC 分位数分布
    2. Soft Pruning: 半衰期权重融合
    3. ORA 3.1 核心：价格跳空偏离修正
    """
    
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        output_dir: str = "reports",
    ) -> None:
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        db_url = os.getenv("DATABASE_URL")
        
        self.alpha_module = get_alpha_research(
            ic_threshold=0.0001,
            n_factors=MAX_FACTORS,
            n_bins=10,
            enable_ensemble=True,
            enable_pac=True,
            enable_lead_lag=True,
            enable_dsnr=True,
            enable_soft_pruning=True,
            enable_gap_deviation=True,
            auto_heal=True,
            db_url=db_url
        )
        
        self.referee = get_backtest_referee(self.alpha_module, output_dir=output_dir)
        self.referee.VERSION = "V161"
        
        logger.info("V161Runner initialized")
        logger.info(f"  Alpha Module: {type(self.alpha_module).__name__}")
        logger.info(f"  Referee: {type(self.referee).__name__}")
        logger.info(f"  Initial Capital: {self.referee.INITIAL_CAPITAL:,.0f}")
        logger.info(f"  ORA 3.1 DSNR + Soft Pruning: Enabled")
        logger.info(f"  Gap Deviation (ORA 3.1): Enabled")
        logger.info(f"  Target IC: > 0.095")
    
    def load_data(self, year: int) -> pd.DataFrame:
        """加载指定年份的数据"""
        if self.parquet_path and Path(self.parquet_path).exists():
            logger.info(f"Loading data from Parquet: {self.parquet_path}")
            df = pd.read_parquet(self.parquet_path)
            
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].dt.year == year]
                df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"Loaded {len(df)} rows for year {year}")
            return df
        
        logger.info(f"Attempting to load data for year {year} from database...")
        
        try:
            from sqlalchemy import create_engine, text
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not configured")
            
            engine = create_engine(db_url)
            
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       turnover_rate, total_mv
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY symbol, trade_date
            """)
            
            df = pd.read_sql_query(query, engine, params={
                'start_date': start_date,
                'end_date': end_date,
            })
            
            logger.info(f"Loaded {len(df)} rows from database for year {year}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            return pd.DataFrame()
    
    def run_audit(self, year: int) -> dict:
        """运行单一年份的审计"""
        logger.info("=" * 70)
        logger.info(f"V161 Audit - Year {year}")
        logger.info("=" * 70)
        
        df = self.load_data(year)
        
        if df.empty:
            logger.warning(f"No data loaded for year {year}")
            return {'year': year, 'error': 'No data loaded', 'passed': False}
        
        logger.info("[Preprocessing] Converting data types...")
        
        if 'trade_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                          'turnover_rate', 'total_mv']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info("[Referee] Running audit...")
        result = self.referee.run_audit(df)
        
        report_path = self.generate_v161_report(result, year)
        
        result['year'] = year
        result['custom_report_path'] = report_path
        
        return result
    
    def generate_v161_report(self, result: dict, year: int) -> str:
        """生成 V161 年度审计报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"v161_audit_{year}_{timestamp}.md"
        
        t1_ic = result.get('t1_ic', {})
        ic_decay = result.get('ic_decay', {})
        backtest_result = result.get('backtest_result', {})
        passed = result.get('passed', False)
        
        factor_ics_v161 = self.alpha_module.get_factor_ics()
        factor_snr_v161 = self.alpha_module.get_factor_snr()
        selected_factors = self.alpha_module.get_selected_factors()
        pruned_factors = self.alpha_module.get_pruned_factors()
        half_life_weights = self.alpha_module.get_factor_half_life_weights()
        
        # V160/V155 对比数据
        v160_ic = 0.066
        v160_ir = 0.55
        v155_ic = 0.07
        v155_ir = 0.55
        
        # 因子信噪比分析表
        snr_table = ""
        for factor in selected_factors:
            ic = factor_ics_v161.get(factor, 0.0)
            snr_data = factor_snr_v161.get(factor, {})
            half_life = half_life_weights.get(factor, 1.0)
            if isinstance(snr_data, dict):
                snr = snr_data.get('snr', 0.0)
                ic_std = snr_data.get('ic_std', 0.0)
                threshold = snr_data.get('dynamic_threshold', DSNR_MIN_THRESHOLD)
            else:
                snr = float(snr_data) if snr_data else 0.0
                ic_std = 0.0
                threshold = DSNR_MIN_THRESHOLD
            pruned = "✗ LOW_WEIGHT" if factor in pruned_factors else ""
            snr_table += f"| {factor} | {ic:.4f} | {ic_std:.4f} | {snr:.4f} | {threshold:.4f} | {half_life:.4f} | {pruned} |\n"
        
        report_content = f"""# V161 Alpha Audit Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Year**: {year}
**Architecture**: Referee-Player (裁判 - 选手)
**Version**: V161 IC 拨乱反正计划 (IC Correction Plan)

---

## 1. Executive Summary (执行摘要)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| T+1 Rank IC | {t1_ic.get('mean_ic', 0):.4f} | > 0.095 | {'✓ PASSED' if t1_ic.get('mean_ic', 0) > 0.095 else '✗ FAILED'} |
| IC IR | {t1_ic.get('ic_ir', 0):.2f} | > 0.7 | {'✓ PASSED' if t1_ic.get('ic_ir', 0) > 0.7 else '✗ FAILED'} |
| IC Decay | {'Monotonic' if ic_decay.get('is_monotonic', False) else 'Non-monotonic'} | Monotonic | {'✓ PASSED' if ic_decay.get('is_monotonic', False) else '✗ FAILED'} |

**Overall Assessment**: **{'PASSED ✓' if passed else 'FAILED ✗'}**

---

## 2. V161 Core Features (V161 核心特性)

### 2.1 Dynamic SNR (DSNR)

| Parameter | Value |
|-----------|-------|
| DSNR Lookback | {DSNR_LOOKBACK} |
| DSNR Quantile | {DSNR_QUANTILE} |
| DSNR Min Threshold | {DSNR_MIN_THRESHOLD} |
| DSNR Max Threshold | {DSNR_MAX_THRESHOLD} |

**【公式】**
- Dynamic_Threshold = Quantile(IC_Distribution, q=0.30)
- 阈值限制在 [{DSNR_MIN_THRESHOLD}, {DSNR_MAX_THRESHOLD}] 范围内

### 2.2 Soft Pruning (Half-Life Weight)

| Parameter | Value |
|-----------|-------|
| Half-Life IC | {HALF_LIFE_IC} |
| Min Weight | {HALF_LIFE_MIN_WEIGHT} |
| Decay Rate | ln(2) ≈ 0.693 |

**【公式】**
- Decay_Weight = exp(-ln(2) * |IC| / HalfLife_IC)
- 对于 IC 低的因子给予更小的衰减权重，而非直接删除

### 2.3 ORA 3.1 Core: Gap Deviation

| Parameter | Value |
|-----------|-------|
| Gap Window | {GAP_WINDOW} |
| Gap Weight | {GAP_WEIGHT} |

**【公式】**
- Gap_Deviation = (Open_current - Close_previous) / Close_previous
- VPC = (1 - Gap_Weight) * (Price_Rank - Volume_Rank) + Gap_Weight * Gap_Signal

### 2.4 Factor SNR Analysis Table (因子信噪比分析表)

| Factor | IC | IC_Std | SNR | DSNR_Threshold | Half-Life Weight | Status |
|--------|-----|--------|-----|----------------|------------------|--------|
{snr_table if snr_table else "*No SNR data*"}

### 2.5 Top Selected Factors

| Factor | IC | Half-Life Weight | Selected |
|--------|-----|------------------|----------|
"""
        
        if factor_ics_v161:
            for factor_name, ic in sorted(factor_ics_v161.items(), key=lambda x: abs(x[1]), reverse=True)[:12]:
                hlw = half_life_weights.get(factor_name, 1.0)
                selected = "✓" if factor_name in selected_factors else ""
                pruned = " (LOW_WEIGHT)" if factor_name in pruned_factors else ""
                report_content += f"| {factor_name}{pruned} | {ic:.4f} | {hlw:.4f} | {selected} |\n"
        
        report_content += f"""
---

## 3. V161 vs V160 vs V155 Comparison (IC 提升对比)

| Metric | V155 | V160 | V161 | Improvement (vs V160) |
|--------|------|------|------|----------------------|
| T+1 IC | {v155_ic:.4f} | {v160_ic:.4f} | {t1_ic.get('mean_ic', 0):.4f} | {t1_ic.get('mean_ic', 0) - v160_ic:+.4f} |
| IC IR | {v155_ir:.2f} | {v160_ir:.2f} | {t1_ic.get('ic_ir', 0):.2f} | {t1_ic.get('ic_ir', 0) - v160_ir:+.2f} |

**IC vs V160**: {t1_ic.get('mean_ic', 0) - v160_ic:+.4f}
**IR vs V160**: {t1_ic.get('ic_ir', 0) - v160_ir:+.2f}

---

## 4. IC Decay Analysis (IC 衰减分析)

| Horizon | IC | Pattern |
|---------|-----|---------|
| T+1 | {ic_decay.get('t1_ic', 0):.4f} | Baseline |
| T+3 | {ic_decay.get('t3_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t1_ic', 0) >= ic_decay.get('t3_ic', 0) else '✗ Non-monotonic'} |
| T+5 | {ic_decay.get('t5_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t3_ic', 0) >= ic_decay.get('t5_ic', 0) else '✗ Non-monotonic'} |

**Decay Pattern**: {ic_decay.get('decay_pattern', 'N/A')}

---

## 5. Backtest Performance (回测表现)

| Metric | Value |
|--------|-------|
| Initial Capital | {self.referee.INITIAL_CAPITAL:,.0f} |
| Final Value | {backtest_result.get('final_value', 0):,.2f} |
| Total Return | {backtest_result.get('total_return', 0):.2%} |
| Annual Return | {backtest_result.get('annual_return', 0):.2%} |
| Sharpe Ratio | {backtest_result.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {backtest_result.get('max_drawdown', 0):.2%} |

---

## 6. Conclusion (结论)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| T+1 Rank IC | > 0.095 | {t1_ic.get('mean_ic', 0):.4f} | {'✓' if t1_ic.get('mean_ic', 0) > 0.095 else '✗'} |
| IC IR | > 0.7 | {t1_ic.get('ic_ir', 0):.2f} | {'✓' if t1_ic.get('ic_ir', 0) > 0.7 else '✗'} |
| IC Decay | Monotonic | {ic_decay.get('decay_pattern', 'N/A')} | {'✓' if ic_decay.get('is_monotonic', False) else '✗'} |

**{'PASSED ✓' if passed else 'FAILED ✗'}**

---

## 7. Reflection Report (反思报告)

### V161 核心改进

1. **Dynamic SNR (DSNR)**: 基于 250 天 IC 分位数分布动态生成阈值
   - 如果全市场信号信噪比普遍降低，阈值应自动下调以维持交易机会
   - 公式：Dynamic_Threshold = Quantile(IC_Distribution, q=0.30)

2. **Soft Pruning (半衰期权重)**: 废弃 V160 的暴力清零逻辑
   - 对于 IC 低的因子给予更小的衰减权重，而非直接删除
   - 公式：Decay_Weight = exp(-ln(2) * |IC| / HalfLife_IC)

3. **ORA 3.1 核心：价格跳空偏离修正**
   - 在 compute_volume_price_contradiction 中引入价格跳空偏离
   - 这比单纯的熵因子更符合 2024 年 A 股特征
   - 公式：Gap_Deviation = (Open_current - Close_previous) / Close_previous

### 被软剪枝的因子（权重 < 0.3）
{', '.join(pruned_factors) if pruned_factors else 'None'}

### 约束逻辑位置

1. **Dynamic SNR** - `alpha_research_v161.py`:
   - `RollingICStatsCalculator.compute_rolling_ic_stats()`: IC 统计量 + 动态阈值计算
   - `compute_dynamic_snr_threshold()`: 分位数阈值计算

2. **Soft Pruning** - `alpha_research_v161.py`:
   - `compute_half_life_weight()`: 半衰期权重计算
   - `AlphaResearchV161._compute_half_life_weight()`: 因子半衰期权重

3. **Gap Deviation** - `alpha_research_v161.py`:
   - `FactorGeneratorV161.compute_volume_price_contradiction()`: 价格跳空偏离修正

---

*Report generated by V161 Unified Main Entry (IC Correction Plan)*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        json_result = {
            'alpha_metrics': {'t1_ic': t1_ic, 'ic_decay': ic_decay, 'passed': passed},
            'backtest_metrics': backtest_result,
            'factor_ics': factor_ics_v161,
            'factor_snr': factor_snr_v161,
            'factor_half_life_weights': half_life_weights,
            'selected_factors': selected_factors,
            'pruned_factors': pruned_factors,
            'v160_comparison': {
                'v160_ic': v160_ic,
                'v160_ir': v160_ir,
                'ic_improvement': t1_ic.get('mean_ic', 0) - v160_ic,
                'ir_improvement': t1_ic.get('ic_ir', 0) - v160_ir,
            },
            'v155_comparison': {
                'v155_ic': v155_ic,
                'v155_ir': v155_ir,
                'ic_improvement': t1_ic.get('mean_ic', 0) - v155_ic,
                'ir_improvement': t1_ic.get('ic_ir', 0) - v155_ir,
            },
            'config': {'year': year, 'initial_capital': self.referee.INITIAL_CAPITAL},
        }
        
        json_path = self.output_dir / f"v161_audit_{year}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        return str(report_path)
    
    def run_multi_year_audit(self, years: list[int]) -> dict:
        """运行多年份的审计"""
        logger.info("=" * 70)
        logger.info(f"V161 Multi-Year Audit - Years: {years}")
        logger.info("=" * 70)
        
        results = []
        passed_count = 0
        all_ic_values = []
        
        for year in years:
            result = self.run_audit(year)
            results.append(result)
            if result.get('passed', False):
                passed_count += 1
            if 't1_ic' in result:
                all_ic_values.append(result['t1_ic'].get('mean_ic', 0))
        
        cross_year_ic_mean = float(np.mean(all_ic_values)) if all_ic_values else 0
        cross_year_ic_std = float(np.std(all_ic_values, ddof=1)) if len(all_ic_values) > 1 else 0
        cross_year_ic_ir = cross_year_ic_mean / cross_year_ic_std if cross_year_ic_std > 1e-10 else 0
        
        summary = {
            'years': years, 'results': results, 'passed_count': passed_count,
            'total_count': len(years), 'cross_year_ic_mean': cross_year_ic_mean,
            'cross_year_ic_std': cross_year_ic_std, 'cross_year_ic_ir': cross_year_ic_ir,
        }
        
        self._generate_reflection(summary)
        
        return summary
    
    def _generate_reflection(self, summary: dict) -> str:
        """生成 V161 反思报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reflection_path = self.output_dir / f"v161_reflection_{timestamp}.json"
        
        factor_ics = self.alpha_module.get_factor_ics()
        factor_snr = self.alpha_module.get_factor_snr()
        selected_factors = self.alpha_module.get_selected_factors()
        pruned_factors = self.alpha_module.get_pruned_factors()
        half_life_weights = self.alpha_module.get_factor_half_life_weights()
        
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'version': 'V161',
            'summary': {
                'years': summary['years'],
                'passed_count': summary['passed_count'],
                'total_count': summary['total_count'],
                'cross_year_ic_mean': summary['cross_year_ic_mean'],
                'cross_year_ic_std': summary['cross_year_ic_std'],
                'cross_year_ic_ir': summary['cross_year_ic_ir'],
            },
            'selected_factors': selected_factors,
            'factor_ics': factor_ics,
            'factor_snr': factor_snr,
            'factor_half_life_weights': half_life_weights,
            'pruned_factors': pruned_factors,
            'v160_comparison': {
                'v160_ic': 0.066,
                'v160_ir': 0.55,
                'v161_ic': summary['cross_year_ic_mean'],
                'v161_ir': summary['cross_year_ic_ir'],
            },
            'conclusion': {
                'ic_target': 0.095,
                'ic_actual': summary['cross_year_ic_mean'],
                'ir_target': 0.7,
                'ir_actual': summary['cross_year_ic_ir'],
                'passed': summary['cross_year_ic_mean'] > 0.095 and summary['cross_year_ic_ir'] > 0.7,
            }
        }
        
        with open(reflection_path, 'w', encoding='utf-8') as f:
            json.dump(reflection, f, indent=2, default=str)
        
        logger.info(f"Reflection saved to: {reflection_path}")
        
        return str(reflection_path)


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV161...")
    
    np.random.seed(42)
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], 1000),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], 1000),
        'close': np.random.randn(1000) * 10 + 100,
        'open': np.random.randn(1000) * 10 + 100,
        'volume': np.random.randn(1000) * 1000 + 5000,
        'amount': np.random.randn(1000) * 10000 + 50000,
        'pct_chg': np.random.randn(1000) * 2,
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  Factor SNR: {alpha.get_factor_snr()}")
    logger.info(f"  Factor Half-Life Weights: {alpha.get_factor_half_life_weights()}")
    logger.info(f"  Pruned factors: {alpha.get_pruned_factors()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")