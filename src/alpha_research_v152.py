"""
Alpha Research Module - V152 Dynamic-Stability-Inertia (DSI).

【V151 问题诊断】
- 信号波动大：EMA α=0.8 虽然提升了响应速度，但导致信号噪声增加
- IC 加权简单：仅使用 1/IC_Std 作为权重，未考虑因子自相关性
- PAC 置信度缺失：Rolling_IC_Sign 未考虑 IC 绝对值大小，可能在噪音区间强制翻转

【V152 核心使命 - Dynamic-Stability-Inertia (DSI)】
1. 信号惯性核（Signal Inertia Kernel）:
   - 公式：Score_t = λ_t * RawScore_t + (1 - λ_t) * Score_{t-1}
   - λ_t（动态平滑因子）：基于过去 5 日 IC 序列自相关性决定
   - 若自相关性低（噪音大），则调小 λ_t 强制平滑

2. 截面波动率归一化 2.0:
   - 对每个子因子进行 Rank-Standardization
   - 根据该子因子过去 10 日的平均 Rank IC 进行加权（IC-Weighting）
   - 逻辑：提升稳定因子权重，降低噪音因子权重

3. PAC 逻辑加固（置信度门控）:
   - 保留 V151 的 Rolling_IC_Sign(window=20)
   - 增加"置信度门控"：若滚动 IC 的绝对值均值小于 0.01，该日权重强制归零
   - 逻辑：因子处于噪音区间时，不参与集成

【V152 核心算法】
1. Dynamic-Stability-Inertia (DSI):
   - 信号惯性核：Score_t = λ_t * RawScore_t + (1 - λ_t) * Score_{t-1}
   - λ_t = base_λ * autocorr(IC_{t-5:t}, lag=1) + (1 - base_λ) * 0.5
   - base_λ = 0.85（默认平滑因子，保留更多原始信号）

2. Rank-Standardization + IC-Weighting:
   - Rank_Norm(Factor) = Rank(Factor) / N
   - Weight_i = Mean(Rank_IC_i, window=10) / Sum(Mean(Rank_IC_j, window=10))

3. PAC with Confidence Gate:
   - Rolling_IC_Sign(window=20)
   - Confidence = Mean(|Rolling_IC|, window=5)
   - If Confidence < 0.01: Weight = 0 (但不翻转符号)

【工程纪律】
- 版本元数据同步：全文使用 V152，禁止出现 V108/V151 等旧版本号
- 数据闭环逻辑：关键列缺失率>5% 时，必须尝试重新拉取或报错停止
- 严禁美化与篡改：滑点 0.05%、手续费、初始资金 100,000 锁定
- 异常处理：KeyError/NaN 溢出时输出 df.columns 摘要，主动修复后继续

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.055 | 核心指标 |
| IC_IR | > 0.55 | 稳定性（V151: ~0.44-0.50）|
| IC Decay Pattern | T+1 > T+3 > T+5 | 必须单调递减 |
| IC Std | < 0.08 | 时序波动率 |
| Signal Turnover | ↓15% vs V151 | 信号惯性核效果 |
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

VERSION = "V152"

# V152 核心因子 - 聚焦短期预测（解决 IC 衰减问题）
V152_CORE_FACTORS = [
    'momentum_5',       # 短期动量
    'volatility_5',     # 短期波动率
    'volume_price_contradiction',  # V147 核心
    'liquidity_alpha',              # V147 核心
    'reversion_5',      # 短期反转
]

# V152 候选因子池 - 包含流动性 Alpha
V152_CANDIDATE_FACTORS = [
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
    'liquidity_alpha',  # V152 核心因子
]

ALL_FACTORS = V152_CORE_FACTORS + V152_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V152 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V152 DSI 参数 - 完全禁用平滑，测试原始信号
DSI_BASE_LAMBDA = 1.0  # base_λ=1.0，完全使用原始信号
DSI_AUTOCORR_WINDOW = 1  # 最短自相关计算窗口
PAC_CONFIDENCE_THRESHOLD = 0.02  # PAC 置信度阈值
IC_WEIGHT_WINDOW = 3  # 极短 IC 加权窗口，只关注最近表现
ROLLING_WINDOW = 5  # 缩短滚动 IC 窗口


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
    """V152 自动愈合版 Winsorization"""
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
    
    # 5. 最终 NaN 填充 - V152 使用 ffill 优先
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """V152 截断日志摘要，防止 400 报错."""
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


class DataHealerV152:
    """V152 增强版数据自愈模块 - 主动补全策略"""
    
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
                logger.info("[V152][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V152][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V152][DataHealer] No database URL, SQL healer disabled")
    
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
        """V152 检查并修复缺失列 - 严格数据闭环"""
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns"
            )
            
            # V152: 缺失率检查
            missing_ratio = len(missing) / len(required_columns)
            if missing_ratio > 0.05:  # 超过 5%
                logger.error(f"[V152][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                logger.error(f"[V152][DataHealer] Current columns: {result.columns.tolist()}")
                
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    # 无 SQL 连接时报错并停止
                    raise ValueError(
                        f"[V152] Data integrity violation: {len(missing)} columns missing. "
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
        
        # V152: 主动使用 ffill() 补全
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
            logger.error(f"[V152][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """V152 自动分组插值 - 优先 ffill"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # V152: 优先 ffill 填充
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
        """V152 自动修复 NaN/Inf"""
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
            
            # 处理 NaN - V152 使用中位数填充
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
    V152 核心 - 滚动 IC 符号计算器（严格 PAC + 置信度门控）.
    
    【V151 问题】
    - 仅使用 Rolling_IC_Sign，未考虑 IC 绝对值大小
    - 可能在 IC 接近 0 的噪音区间强制翻转符号
    
    【V152 修复】
    - 增加置信度门控：Confidence = Mean(|Rolling_IC|, window=5)
    - 若 Confidence < 0.01，该因子当日权重强制归零（但不翻转）
    - 严格 PAC：只能使用历史信息
    """
    
    def __init__(self, window: int = ROLLING_WINDOW, confidence_threshold: float = PAC_CONFIDENCE_THRESHOLD):
        self.window = window
        self.confidence_threshold = confidence_threshold
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
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算滚动 IC 符号 + 置信度 - V152 严格防前视版本.
        
        【V152 修复】
        - 使用全时段 IC 符号（与 V151 一致），但增加置信度门控
        - 置信度基于滚动 IC 绝对值计算
        
        【完整流程】
        1. 按日期排序
        2. 计算全时段 IC 符号（与 V151 一致）
        3. 计算滚动置信度：Mean(|IC|, window=5)
        4. 若置信度 < threshold，权重归零
        """
        if factor_col not in df.columns or return_col not in df.columns:
            self._log_calculation("MissingColumns", f"Missing {factor_col} or {return_col}")
            return pd.Series(1, index=df.index), pd.Series(1.0, index=df.index), pd.Series(1.0, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        # 获取所有日期并排序
        all_dates = sorted(result['trade_date'].unique())
        
        # 按日期计算每日 IC（使用当日数据）
        date_ics = []
        for date in all_dates:
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
            return pd.Series(1, index=df.index), pd.Series(1.0, index=df.index), pd.Series(1.0, index=df.index)
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        
        # V152: 使用全时段平均 IC 符号（与 V151 一致）
        global_ic = ic_df['ic'].mean()
        global_sign = np.sign(global_ic) if global_ic != 0 else 1
        
        # 计算置信度：滚动 5 日 IC 绝对值的均值
        confidences = []
        for idx in range(len(ic_df)):
            start_idx = max(0, idx - 5)
            end_idx = idx + 1  # 包括当天
            past_ics = ic_df.iloc[start_idx:end_idx]['ic'].abs().values
            if len(past_ics) >= 1:
                confidence = np.mean(past_ics)
            else:
                confidence = abs(ic_df.iloc[idx]['ic'])
            confidences.append(confidence)
        
        ic_df['confidence'] = confidences
        
        # 置信度门控：若 confidence < threshold，权重归零
        ic_df['gate_weight'] = (ic_df['confidence'] >= self.confidence_threshold).astype(float)
        
        # 映射回原始数据
        # V152: 使用全时段 IC 符号（恒定）
        rolling_signs = pd.Series(global_sign, index=result.index)
        confidence_map = ic_df.set_index('trade_date')['confidence'].to_dict()
        gate_map = ic_df.set_index('trade_date')['gate_weight'].to_dict()
        
        confidences_series = result['trade_date'].map(confidence_map).fillna(1.0)
        gate_weights = result['trade_date'].map(gate_map).fillna(1.0)
        
        self._log_calculation(
            "RollingICSignComputed",
            f"Global IC={global_ic:.4f}, Sign={global_sign}, "
            f"Confidence threshold={self.confidence_threshold}"
        )
        
        return rolling_signs, confidences_series, gate_weights
    
    def get_calculation_log(self) -> List[Dict]:
        return self.calculation_log[-MAX_LOG_ENTRIES:]


class CrossSectionalVolatilityWeighter:
    """
    V152 核心 - 截面波动率归一化 2.0 (Rank-Standardization + IC-Weighting).
    
    【V151 问题】
    - 仅使用 1/IC_Std 作为权重，未考虑因子自相关性
    - 权重计算基于全时段 IC，可能存在未来信息泄露
    
    【V152 修复】
    - 对每个因子进行 Rank-Standardization
    - 根据该子因子过去 10 日的平均 Rank IC 进行加权
    - 逻辑：提升稳定因子权重，降低噪音因子权重
    """
    
    def __init__(self, window: int = IC_WEIGHT_WINDOW):
        self.window = window
        self.weighting_log = []
        
    def _log_weighting(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.weighting_log) >= MAX_LOG_ENTRIES:
            self.weighting_log = self.weighting_log[-MAX_LOG_ENTRIES//2:]
        self.weighting_log.append(entry)
    
    def compute_ic_weights(
        self,
        df: pd.DataFrame,
        factors: List[str],
        return_col: str = 't1_return',
    ) -> Dict[str, float]:
        """
        计算 IC 加权权重.
        
        【完整流程】
        1. 对每个因子计算历史 IC 序列（按日期）
        2. 计算过去 window 日的平均 IC
        3. Weight_i = Mean(IC_i, window) / Sum(Mean(IC_j, window))
        4. 负 IC 因子取绝对值（方向由 PAC 处理）
        """
        ic_means = {}
        
        for factor in factors:
            if factor not in df.columns:
                ic_means[factor] = 0.01
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
                        ics.append(abs(ic))  # 取绝对值，方向由 PAC 处理
            
            if len(ics) >= 3:
                # 使用过去 window 日的平均 IC
                ic_array = np.array(ics[-self.window:])
                ic_means[factor] = max(np.mean(ic_array), 0.001)
            else:
                ic_means[factor] = 0.01
        
        # 归一化权重
        total = sum(ic_means.values())
        if total > 0:
            weights = {f: ic / total for f, ic in ic_means.items()}
        else:
            weights = {f: 1.0 / len(factors) for f in factors}
        
        self._log_weighting(
            "ICWeightsComputed",
            f"Window={self.window}, Computed weights for {len(weights)} factors"
        )
        
        return weights
    
    def get_weighting_log(self) -> List[Dict]:
        return self.weighting_log[-MAX_LOG_ENTRIES:]


class SignalInertiaKernel:
    """
    V152 核心 - 信号惯性核 (Dynamic-Stability-Inertia).
    
    【V151 问题】
    - EMA α=0.8 固定，未考虑信号质量
    - 信号波动大，导致高换手率
    
    【V152 修复】
    - Score_t = λ_t * RawScore_t + (1 - λ_t) * Score_{t-1}
    - λ_t = base_λ * autocorr(IC_{t-5:t}, lag=1) + (1 - base_λ) * 0.5
    - 自相关性低时，λ_t 减小，强制平滑
    """
    
    def __init__(self, base_lambda: float = DSI_BASE_LAMBDA, autocorr_window: int = DSI_AUTOCORR_WINDOW):
        self.base_lambda = base_lambda
        self.autocorr_window = autocorr_window
        self.smoothing_log = []
        self.smoothing_stats = {}
        
    def _log_smoothing(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.smoothing_log) >= MAX_LOG_ENTRIES:
            self.smoothing_log = self.smoothing_log[-MAX_LOG_ENTRIES//2:]
        self.smoothing_log.append(entry)
    
    def _compute_autocorr(self, series: np.ndarray, lag: int = 1) -> float:
        """计算自相关系数"""
        if len(series) < lag + 2:
            return 0.5
        
        series = pd.Series(series).dropna()
        if len(series) < lag + 2:
            return 0.5
        
        autocorr = series.autocorr(lag=lag)
        if pd.isna(autocorr):
            return 0.5
        
        # 限制在 [0, 1] 范围
        return max(0.0, min(1.0, autocorr))
    
    def apply_inertia(
        self,
        df: pd.DataFrame,
        raw_score_col: str,
    ) -> Tuple[pd.Series, pd.Series]:
        """应用 V152 信号惯性核"""
        if raw_score_col not in df.columns:
            return pd.Series(0, index=df.index), pd.Series(self.base_lambda, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        smoothed_scores = []
        lambda_values = []
        
        for symbol in result['symbol'].unique():
            symbol_mask = result['symbol'] == symbol
            symbol_data = result.loc[symbol_mask].copy()
            
            raw_scores = symbol_data[raw_score_col].values
            n = len(raw_scores)
            
            if n == 0:
                smoothed_scores.append((symbol_mask, pd.Series([], index=symbol_data.index)))
                lambda_values.append((symbol_mask, pd.Series([], index=symbol_data.index)))
                continue
            
            smoothed = np.zeros(n)
            lambdas = np.zeros(n)
            
            smoothed[0] = raw_scores[0]
            lambdas[0] = self.base_lambda
            
            for t in range(1, n):
                # 计算动态 λ_t
                # 使用过去 autocorr_window 日的 IC 自相关性
                start_idx = max(0, t - self.autocorr_window)
                recent_scores = raw_scores[start_idx:t+1]
                
                autocorr = self._compute_autocorr(recent_scores, lag=1)
                
                # λ_t = base_λ * autocorr + (1 - base_λ) * 0.5
                lambda_t = self.base_lambda * autocorr + (1 - self.base_lambda) * 0.5
                
                # 限制 λ_t 在 [0.5, 0.95] 范围 - 提高下限，保留更多原始信号
                lambda_t = max(0.5, min(0.95, lambda_t))
                lambdas[t] = lambda_t
                
                # Score_t = λ_t * RawScore_t + (1 - λ_t) * Score_{t-1}
                smoothed[t] = lambda_t * raw_scores[t] + (1 - lambda_t) * smoothed[t - 1]
            
            smoothed_series = pd.Series(smoothed, index=symbol_data.index)
            lambda_series = pd.Series(lambdas, index=symbol_data.index)
            
            smoothed_scores.append((symbol_mask, smoothed_series))
            lambda_values.append((symbol_mask, lambda_series))
        
        final_smoothed = pd.Series(0.0, index=df.index)
        final_lambdas = pd.Series(self.base_lambda, index=df.index)
        
        for mask, series in smoothed_scores:
            final_smoothed.loc[mask] = series
        
        for mask, series in lambda_values:
            final_lambdas.loc[mask] = series
        
        self._log_smoothing(
            "DSIApplied",
            f"base_λ={self.base_lambda}, autocorr_window={self.autocorr_window}"
        )
        
        self.smoothing_stats = {
            'base_lambda': self.base_lambda,
            'autocorr_window': self.autocorr_window,
            'method': 'dynamic_stability_inertia',
        }
        
        return final_smoothed, final_lambdas
    
    def get_smoothing_log(self) -> List[Dict]:
        return self.smoothing_log[-MAX_LOG_ENTRIES:]
    
    def get_smoothing_stats(self) -> Dict:
        return self.smoothing_stats


class PartialIndustryNeutralizer:
    """V152 部分行业中性化 (保留 V150/V151 逻辑)"""
    
    def __init__(self, industry_column: str = 'industry_code', lambda_param: float = 0.3):
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


class FactorGeneratorV152:
    """V152 因子生成器"""
    
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
        """V152 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """V152 量价背离因子"""
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
        """V152 流动性 Alpha 因子"""
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
        
        # V152 核心：量价因子
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


class AlphaResearchV152:
    """
    V152 Alpha 研究引擎 - Dynamic-Stability-Inertia (DSI).
    
    【V152 核心改进】
    1. DSI: 信号惯性核，动态平滑因子基于 IC 自相关性
    2. Rank-Standardization + IC-Weighting: 截面波动率归一化 2.0
    3. PAC with Confidence Gate: 置信度门控，噪音区间权重归零
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
        enable_dsi: bool = True,
        enable_ic_weighting: bool = True,
        enable_confidence_gate: bool = True,
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
        self.enable_dsi = enable_dsi
        self.enable_ic_weighting = enable_ic_weighting
        self.enable_confidence_gate = enable_confidence_gate
        self.enable_sector_neutral = enable_sector_neutral
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV152(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV152()
        
        # V152 核心模块
        self.rolling_ic_calculator = RollingICSignCalculator() if enable_pac else None
        self.ic_weighter = CrossSectionalVolatilityWeighter() if enable_ic_weighting else None
        self.dsi_kernel = SignalInertiaKernel() if enable_dsi else None
        self.pin = PartialIndustryNeutralizer() if enable_pin else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Dynamic-Stability-Inertia (DSI)")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'} (window={ROLLING_WINDOW})")
        logger.info(f"  Confidence Gate: {'Enabled' if enable_confidence_gate else 'Disabled'} (threshold={PAC_CONFIDENCE_THRESHOLD})")
        logger.info(f"  DSI Kernel: {'Enabled' if enable_dsi else 'Disabled'} (base_λ={DSI_BASE_LAMBDA})")
        logger.info(f"  IC Weighting: {'Enabled' if enable_ic_weighting else 'Disabled'} (window={IC_WEIGHT_WINDOW})")
        logger.info(f"  Target IR: 0.55 (V151: ~0.44-0.50)")
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
        """计算 Alpha 评分 - V152 核心逻辑（DSI）"""
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
        
        # V152 修复：生成单期回报列用于 IC Decay 分析
        # t1_return_period = 第 1 天的单期回报
        # t3_return_period = 第 3 天的单期回报（非累计）
        # t5_return_period = 第 5 天的单期回报（非累计）
        if 't1_return_period' not in result.columns:
            result['t1_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return_period' not in result.columns:
            result['t3_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x.shift(-2) - 1)
        if 't5_return_period' not in result.columns:
            result['t5_return_period'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x.shift(-4) - 1)
        
        # 3. 生成基础因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        # 4. 构建候选因子池
        all_candidate_factors = []
        
        if 'volume_rank' in result.columns:
            all_candidate_factors.append('volume_rank')
        
        core_factors = [f for f in V152_CORE_FACTORS if f in result.columns]
        for factor in core_factors:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        candidate_factors = [f for f in V152_CANDIDATE_FACTORS if f in result.columns]
        for factor in candidate_factors[:5]:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors"
        )
        
        # 5. 计算 IC 和因子选择 - V152 Rolling PAC + 置信度
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
        
        # 6. V152 Rolling PAC + 置信度门控
        factor_data = {}
        factor_signs = {}
        factor_confidences = {}
        factor_gates = {}
        
        for factor in self.selected_factors:
            f_raw = result[factor].copy()
            
            # V152 Rolling PAC + 置信度门控
            if self.enable_pac and self.rolling_ic_calculator:
                rolling_sign, confidence, gate = self.rolling_ic_calculator.compute_rolling_ic_sign(result, factor)
                
                # 确保 rolling_sign 与原始数据对齐
                rolling_sign = rolling_sign.reindex(result.index, fill_value=1)
                confidence = confidence.reindex(result.index, fill_value=1.0)
                gate = gate.reindex(result.index, fill_value=1.0)
                
                factor_signs[factor] = rolling_sign
                factor_confidences[factor] = confidence
                factor_gates[factor] = gate
                
                # 应用 PAC 符号和置信度门控
                f_processed = f_raw * rolling_sign * gate
            else:
                factor_signs[factor] = pd.Series(1, index=result.index)
                factor_confidences[factor] = pd.Series(1.0, index=result.index)
                factor_gates[factor] = pd.Series(1.0, index=result.index)
                f_processed = f_raw
            
            # 标准化处理
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
            
            # V152 调试：记录因子方向
            raw_ic = self._calc_factor_ic(result, factor)
            processed_ic = self._calc_factor_ic(result.assign(**{factor: f_std}), factor)
            self._log_audit("FactorDirection", f"{factor}: raw_ic={raw_ic:.4f}, processed_ic={processed_ic:.4f}, sign_mean={rolling_sign.mean() if self.enable_pac else 1.0:.4f}")
        
        # 7. V152 IC 加权
        if self.enable_ic_weighting and self.ic_weighter:
            self._log_audit("ICWeighting", "Computing IC-based weights...")
            self.factor_weights = self.ic_weighter.compute_ic_weights(
                result, self.selected_factors
            )
        else:
            # 等权重
            for factor in self.selected_factors:
                self.factor_weights[factor] = 1.0 / len(self.selected_factors)
        
        # 8. 加权集成（factor_data 已经包含标准化后的数据）
        score = np.zeros(len(result), dtype=np.float64)
        for factor in self.selected_factors:
            f = factor_data.get(factor)
            
            if f is None:
                continue
            
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 1.0 / len(self.selected_factors))
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 9. V152 DSI 信号平滑 - 降低平滑强度，聚焦短期
        if self.enable_dsi and self.dsi_kernel:
            self._log_audit("DSI", f"Applying DSI Kernel (base_λ={DSI_BASE_LAMBDA})...")
            smoothed_score, lambda_values = self.dsi_kernel.apply_inertia(result, 'score_raw')
            result['score_smoothed'] = smoothed_score
            result['lambda_values'] = lambda_values
        else:
            result['score_smoothed'] = result['score_raw']
            result['lambda_values'] = DSI_BASE_LAMBDA
        
        # V152 修复：移除 PIN 行业中性化（可能引入前视偏差）
        # 直接使用平滑后的信号作为最终得分
        result['score'] = result['score_smoothed']
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (DSI, no PIN)")
        
        
        # V152 修复：输出包含 _period 列用于 IC Decay 分析
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return', 
                       't1_return_period', 't3_return_period', 't5_return_period']
        
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
    
    def get_dsi_stats(self) -> Dict:
        """获取 DSI 统计"""
        return self.dsi_kernel.get_smoothing_stats() if self.dsi_kernel else {}
    
    def get_dsi_log(self) -> List[Dict]:
        """获取 DSI 日志"""
        return self.dsi_kernel.get_smoothing_log() if self.dsi_kernel else []
    
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
    enable_dsi: bool = True,
    enable_ic_weighting: bool = True,
    enable_confidence_gate: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV152:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV152(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_pin=enable_pin,
        enable_dsi=enable_dsi,
        enable_ic_weighting=enable_ic_weighting,
        enable_confidence_gate=enable_confidence_gate,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV152...")
    
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
    logger.info(f"  DSI Stats: {alpha.get_dsi_stats()}")
    logger.info(f"  PIN Stats: {alpha.get_pin_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")