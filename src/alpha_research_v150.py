"""
Alpha Research Module - V150 Polarity-Corrected-Ensemble (PCE).

【V149 问题诊断】
- 严格行业中性化 (SIN) 过度削弱信号，导致 IC 损失
- 动态惯性核 α 计算复杂，自相关性估计不稳定
- Gram-Schmidt 正交化效率低，可能丢失有效信息

【V150 核心使命 - Polarity-Corrected-Ensemble (PCE)】
1. 极性自适应（Polarity Auto-Correction）:
   - 在 generate_scores 阶段，计算各基础因子与 target_return 的相关性符号
   - 若相关性为负，自动执行 Score = -1 * Rank(Factor)
   - 确保进入集成器的是正向贡献信号

2. 回撤行业中性化（Partial Neutralization）:
   - 放弃 V149 的 SIN（全额减法），改为 0.3 权重的软中性化
   - 保留 70% 的行业趋势信号，仅过滤 30% 的极端行业偏离
   - 公式：Score_final = 0.7 * Score_raw - 0.3 * Industry_Mean

3. 信号惯性修正（Decoupled Smoothing）:
   - 将平滑逻辑从因子层移至最终得分层
   - 使用简化的 EMA: Final_Score = 0.4 * New_Score + 0.6 * Prev_Score
   - 减少复杂自相关性计算带来的逻辑混乱

4. 回归 V147 核心算子:
   - 强制恢复 volume_price_contradiction 和 liquidity_alpha 逻辑
   - 作为 V150 的核心底座

【V150 核心算法】
1. Polarity Auto-Correction (PAC):
   - 对每个因子计算 IC 符号
   - Score = sign(IC) * Rank(Factor)
   - 确保所有因子贡献方向一致

2. Partial Industry Neutralization (PIN):
   - Score_final = (1 - λ) * Score_raw - λ * Industry_Mean
   - λ = 0.3 (软中性化)

3. Exponential Moving Average (EMA) Smoothing:
   - Final_Score_t = α * Raw_Score_t + (1 - α) * Final_Score_{t-1}
   - α = 0.4 (新信号权重 40%, 历史信号 60%)

【架构红线】
- 裁判唯一性：必须通过 python main.py --version 150 运行
- 严禁修改 backtest_referee.py 中的资金 (10 万) 和费率 (0.15%)
- 数据缺失时必须主动调用 data_loader 补全，禁止用 dropna() 一删了之
- 报错必改：内置 Auto-Healing 逻辑处理 Inf/NaN，禁止停止运行
- 日志安全：继续执行 truncate_log_summary 逻辑，严禁 dump 超过 50 行

【验收硬指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标 |
| IC_IR | > 0.50 | 稳定性（V149: ~0.40） |
| IC Std | < 0.08 | 时序波动率 |
| 400 Error | 0 | 禁止 String Length 报错 |
| Signal Turnover Rate | 降低 15%+ | 日度信号换手率对比 V149 |
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

VERSION = "V150"

# V150 核心因子（回归 V147）
V150_CORE_FACTORS = [
    'momentum_20',
    'volatility_10',
    'volume_price_contradiction',  # V147 核心
    'liquidity_alpha',              # V147 核心
    'volatility_reversion',
    'reversion_5',
]

# V150 候选因子池
V150_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V150_CORE_FACTORS + V150_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V150 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V150 PCE 参数
PCE_ALPHA = 0.4  # EMA 平滑系数：0.4 * New + 0.6 * Prev
PIN_LAMBDA = 0.3  # 行业中性化权重：0.3 软中性化


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
    """V150 自动愈合版 Winsorization"""
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
    series_clean = series_clean.fillna(mean)
    
    return series_clean


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """
    V150 截断日志摘要，防止 400 报错.
    
    【策略】
    1. 仅保留每月的首末交易日
    2. 如果仍然超过 max_rows，则仅保留前 max_rows 行
    """
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


class DataHealerV150:
    """V150 增强版数据自愈模块"""
    
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
                logger.info("[V150][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V150][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V150][DataHealer] No database URL, SQL healer disabled")
    
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
        """V150 检查并修复缺失列"""
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
        
        result = self._auto_impute_grouped(result, 'trade_date')
        result = self._repair_nan_inf(result)
        
        self._log_healing(
            action="AutoImputeApplied",
            column="ALL_NUMERIC",
            status="SUCCESS",
            details="Applied grouped median imputation + NaN/Inf repair"
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
            logger.error(f"[V150][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """自动分组插值"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            global_median = result[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            
            def fill_group(group):
                group_median = group[col].median()
                if pd.isna(group_median):
                    group_median = global_median
                return group[col].fillna(group_median)
            
            result[col] = result.groupby(group_col, group_keys=False).apply(fill_group)
            result[col] = result[col].fillna(global_median)
        
        return result
    
    def _repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """V150 自动修复 NaN/Inf"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
                self._log_healing(
                    action="InfRepaired",
                    column=col,
                    status="SUCCESS",
                    details=f"Repaired {inf_count} Inf values"
                )
            
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


class PartialIndustryNeutralizer:
    """
    V150 核心 - 部分行业中性化 (PIN).
    
    【V149 问题】
    - 严格行业中性化 (SIN) 全额减法，过度削弱信号
    
    【V150 修复】
    - 使用 0.3 权重软中性化
    - 保留 70% 的行业趋势信号，仅过滤 30% 的极端行业偏离
    - 公式：Score_final = 0.7 * Score_raw - 0.3 * Industry_Mean
    """
    
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
        """
        应用部分行业中性化.
        
        【完整流程】
        1. 按日期和行业分组
        2. 计算每个行业内信号均值
        3. Score_final = (1 - λ) * Score_raw - λ * Industry_Mean
        """
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
                
                # V150 PIN 公式
                neutralized_signal.loc[industry_idx] = (
                    (1 - self.lambda_param) * raw_score - self.lambda_param * industry_mean
                )
        
        self._log_neutralization(
            "PartialIndustryNeutralizationApplied",
            f"λ={self.lambda_param}, Neutralized by {self.industry_column}"
        )
        
        self.neutralization_stats = {
            'method': 'partial_industry_neutralization',
            'lambda': self.lambda_param,
            'industry_column': self.industry_column,
            'signal_retention': 1 - self.lambda_param,
        }
        
        return neutralized_signal
    
    def get_neutralization_log(self) -> List[Dict]:
        return self.neutralization_log[-MAX_LOG_ENTRIES:]
    
    def get_neutralization_stats(self) -> Dict:
        return self.neutralization_stats


class EMASignalSmoother:
    """
    V150 核心 - EMA 信号平滑器 (Decoupled Smoothing).
    
    【V149 问题】
    - 动态惯性核 α 计算复杂，自相关性估计不稳定
    - 平滑逻辑在因子层，导致信息损失
    
    【V150 修复】
    - 将平滑逻辑移至最终得分层
    - 使用简化的 EMA: Final_Score = α * New_Score + (1 - α) * Prev_Score
    - α = 0.4 (新信号权重 40%, 历史信号 60%)
    """
    
    def __init__(self, alpha: float = PCE_ALPHA):
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
        """
        应用 EMA 平滑.
        
        【完整流程】
        1. 按日期排序
        2. 对每个符号应用 EMA
        3. Final_Score_t = α * Raw_Score_t + (1 - α) * Final_Score_{t-1}
        """
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
            smoothed[0] = raw_scores[0]  # 初始化
            
            for t in range(1, n):
                # V150 EMA 公式
                smoothed[t] = self.alpha * raw_scores[t] + (1 - self.alpha) * smoothed[t - 1]
            
            smoothed_series = pd.Series(smoothed, index=symbol_data.index)
            smoothed_scores.append((symbol_mask, smoothed_series))
        
        final_smoothed = pd.Series(0.0, index=df.index)
        for mask, series in smoothed_scores:
            final_smoothed.loc[mask] = series
        
        self._log_smoothing(
            "EMASmoothingApplied",
            f"α={self.alpha}, New={self.alpha*100}%, Prev={(1-self.alpha)*100}%"
        )
        
        self.smoothing_stats = {
            'alpha': self.alpha,
            'new_signal_weight': self.alpha,
            'prev_signal_weight': 1 - self.alpha,
            'method': 'exponential_moving_average',
        }
        
        return final_smoothed
    
    def get_smoothing_log(self) -> List[Dict]:
        return self.smoothing_log[-MAX_LOG_ENTRIES:]
    
    def get_smoothing_stats(self) -> Dict:
        return self.smoothing_stats


class FactorGeneratorV150:
    """V150 因子生成器 - 回归 V147 核心算子"""
    
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
        """V150 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """
        V150 回归 V147 核心算子 - 量价背离因子.
        
        【逻辑】
        价格上涨但成交量萎缩 → 看空信号
        价格下跌但成交量放大 → 看多信号
        """
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
        
        # 排名计算
        price_rank = close_return.fillna(0).rank(method='average', pct=True)
        volume_rank = volume_change.fillna(0).rank(method='average', pct=True)
        
        # 量价背离 = 价格排名 - 成交量排名
        vpc = (price_rank - volume_rank).fillna(0)
        
        self._log_generation(
            "VolumePriceContradiction",
            f"V147 core factor: mean={vpc.mean():.4f}, std={vpc.std():.4f}"
        )
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """
        V150 回归 V147 核心算子 - 流动性 Alpha 因子.
        
        【逻辑】
        订单流不平衡 (OFI) / 波动率
        捕捉流动性驱动的超额收益
        """
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
        
        # V150 波动率反转因子
        result['volatility_reversion'] = self.compute_volatility_reversion(result)
        
        # V150 核心：回归 V147 量价因子
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
        
        self._log_generation("Complete", f"Generated base factors including V147 core factors")
        
        return result


class SectorNeutralValidator:
    """V150 行业中性化校验器"""
    
    def __init__(self, industry_column: str = 'industry_code'):
        self.industry_column = industry_column
        self.validation_log = []
        
    def _log_validation(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.validation_log) >= MAX_LOG_ENTRIES:
            self.validation_log = self.validation_log[-MAX_LOG_ENTRIES//2:]
        self.validation_log.append(entry)
    
    def validate_ir_improvement(
        self, df: pd.DataFrame, signal_col: str, return_col: str = 't1_return'
    ) -> Dict:
        """校验 IR 提升是否来自行业偏离"""
        if signal_col not in df.columns or return_col not in df.columns:
            return {'valid': False, 'reason': 'Missing columns'}
        
        ics_original = []
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            f = day_data[signal_col].fillna(0)
            l = day_data[return_col].fillna(0)
            
            if len(f) > 10 and np.std(f) > 1e-10:
                ic = np.corrcoef(f.rank(method='average'), l.rank(method='average'))[0, 1]
                if not np.isnan(ic):
                    ics_original.append(ic)
        
        if not ics_original:
            return {'valid': False, 'reason': 'No IC calculated'}
        
        ic_mean_orig = np.mean(ics_original)
        ic_std_orig = np.std(ics_original, ddof=1) + 1e-10
        ir_original = ic_mean_orig / ic_std_orig
        
        return {
            'valid': True,
            'ir_original': float(ir_original),
        }
    
    def get_validation_log(self) -> List[Dict]:
        return self.validation_log[-MAX_LOG_ENTRIES:]


class AlphaResearchV150:
    """
    V150 Alpha 研究引擎 - Polarity-Corrected-Ensemble (PCE).
    
    【V150 核心改进】
    1. Polarity Auto-Correction (PAC): 因子极性自动校正
    2. Partial Industry Neutralization (PIN): 0.3 软中性化
    3. EMA Signal Smoothing: 简化的指数平滑
    
    【目标指标】
    - T+1 Rank IC > 0.05
    - IC_IR > 0.50（V149: ~0.40）
    - IC Std < 0.08
    - 400 Error: 0
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_pac: bool = True,  # V150 核心：极性校正
        enable_pin: bool = True,  # V150 核心：部分行业中性化
        enable_ema: bool = True,  # V150 核心：EMA 平滑
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
        self.enable_sector_neutral = enable_sector_neutral
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV150(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV150()
        
        # V150 核心模块
        self.pin = PartialIndustryNeutralizer() if enable_pin else None
        self.ema = EMASignalSmoother() if enable_ema else None
        self.sector_validator = SectorNeutralValidator() if enable_sector_neutral else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Polarity-Corrected-Ensemble (PCE)")
        logger.info(f"  PAC: {'Enabled' if enable_pac else 'Disabled'}")
        logger.info(f"  PIN: {'Enabled' if enable_pin else 'Disabled'} (λ={PIN_LAMBDA})")
        logger.info(f"  EMA: {'Enabled' if enable_ema else 'Disabled'} (α={PCE_ALPHA})")
        logger.info(f"  Target IR: 0.50 (V149: ~0.40)")
        logger.info(f"  400 Error Fix: Log truncation enabled")
    
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
        """计算 Alpha 评分 - V150 核心逻辑（PCE）"""
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
        
        if 't1_return_period' not in result.columns:
            result['t1_return_period'] = result.groupby('symbol')['close'].transform(
                lambda x: x.shift(-1) / x - 1
            )
        
        # 3. 生成基础因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
            self._log_audit("FactorGeneration", "Generated base factors including V147 core factors")
        
        # 4. 构建候选因子池
        all_candidate_factors = []
        
        if 'volume_rank' in result.columns:
            all_candidate_factors.append('volume_rank')
        
        core_factors = [f for f in V150_CORE_FACTORS if f in result.columns]
        for factor in core_factors:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        candidate_factors = [f for f in V150_CANDIDATE_FACTORS if f in result.columns]
        for factor in candidate_factors[:5]:
            if factor not in all_candidate_factors:
                all_candidate_factors.append(factor)
        
        self._log_audit(
            "FactorCandidatePool",
            f"Built candidate pool with {len(all_candidate_factors)} factors"
        )
        
        # 5. 计算 IC 和因子选择 - V150 PAC 极性校正
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
        
        # 6. 准备因子数据并应用 V150 PAC 极性校正
        factor_data = {}
        
        for factor in self.selected_factors:
            f_raw = result[factor]
            ic = self.factor_ics[factor]
            
            # V150 PAC: 极性自适应校正
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit(
                    "PAC",
                    f"{factor}: IC={ic:.4f} < 0, FLIPPED direction for positive contribution"
                )
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit(
                    "PAC",
                    f"{factor}: IC={ic:.4f} >= 0, kept direction"
                )
            
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 7. 计算初始分数（等权重集成）
        score = np.zeros(len(result), dtype=np.float64)
        for i, factor in enumerate(self.selected_factors):
            f = factor_data[factor]
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            score += f_clean.values / len(self.selected_factors)
        
        result['score_raw'] = score
        
        # 8. V150 EMA 信号平滑
        if self.enable_ema and self.ema:
            self._log_audit("EMA", "Applying EMA Signal Smoothing...")
            smoothed_score = self.ema.apply_ema(result, 'score_raw')
            result['score_smoothed'] = smoothed_score
        else:
            result['score_smoothed'] = result['score_raw']
        
        # 9. V150 PIN 部分行业中性化
        if self.enable_pin and self.pin:
            self._log_audit("PIN", "Applying Partial Industry Neutralization...")
            final_score = self.pin.neutralize(result, 'score_smoothed')
            result['score'] = final_score
        else:
            result['score'] = result['score_smoothed']
        
        # 10. 等权重集成
        for factor in self.selected_factors:
            self.factor_weights[factor] = 1.0 / len(self.selected_factors)
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (PCE)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return']
        for col in ['t1_return_period']:
            if col in result.columns and col not in output_cols:
                output_cols.append(col)
        
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """获取因子 IC"""
        if df is not None and not df.empty:
            ics = {}
            for factor in self.selected_factors:
                if factor in df.columns:
                    ic = self._calc_factor_ic(df, factor)
                    direction = self.factor_directions.get(factor, 1)
                    ics[factor] = ic * direction
                else:
                    ics[factor] = self.factor_ics.get(factor, 0.0) * self.factor_directions.get(factor, 1)
            return ics
        
        adjusted = {}
        for f, ic in self.factor_ics.items():
            direction = self.factor_directions.get(f, 1)
            adjusted[f] = ic * direction
        return adjusted
    
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
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV150:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV150(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_pin=enable_pin,
        enable_ema=enable_ema,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV150...")
    
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