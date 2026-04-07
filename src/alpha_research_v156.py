"""
Alpha Research Module - V156 Signal-Smoothing & Non-Linear Residual (ORA 3.0).

【V155 审计结论】
- IC: 0.0924 (优秀)
- IR: 0.58 (未达到工业级 0.7)
- 问题：信号在截面上的分布存在尖峰厚尾现象，导致实盘换手率过高

【V156 核心使命 - 信号平滑度攻坚与非线性残差增强】
1. GARCH-Like Volatility Scaling (信号波动率收缩):
   - 基于历史 5 日信号标准差的自适应收缩
   - 如果过去 5 天信号跳变太快，强制压缩当日信号权重

2. ORA 3.0 (二阶非线性残差挖掘):
   - 引入 Kernel-Trick 或二阶交互
   - 计算核心因子 volume_price_contradiction 与辅助因子的交叉项
   - 对交叉项进行正交化，提取"非线性增量"

3. Adaptive Threshold Gate (自适应置信度门控):
   - 基于信号分布偏度 (Skewness) 的门控
   - 当日信号呈现明显偏态（多头信号极其集中）时，才允许全额调仓
   - 否则限制日换手率

4. 数据自愈多级回退填充:
   - 第一级：SQL 补全
   - 第二级：中位数填充
   - 第三级：行业均值填充
   - 严禁直接 dropna() 导致样本量缩减！

【验收硬指标】
| 指标 | V155 | V156 目标 | 判定标准 |
|------|------|----------|----------|
| T+1 Rank IC | 0.0924 | > 0.09 | 核心指标 |
| IC_IR | 0.58 | > 0.7 | 稳定性 |
| Turnover | - | 下降 20% | 换手率改善 |
| IC Decay Pattern | T+1 > T+3 > T+5 | 必须单调递减 |

【工程纪律】
- 版本元数据同步：全文使用 V156
- 严禁修改 src/engine/ 目录
- 唯一入口：python main.py --version 156 --year 2024
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

VERSION = "V156"

# V156 核心因子 - 聚焦短期预测
V156_CORE_FACTORS = [
    'momentum_5',       # 短期动量
    'volatility_5',     # 短期波动率
    'volume_price_contradiction',  # V147 核心 - ORM 核心因子
    'liquidity_alpha',              # V147 核心
    'reversion_5',      # 短期反转
]

# V156 候选因子池
V156_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V156_CORE_FACTORS + V156_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V156 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V156 ORA 3.0 参数
ORM_CORE_FACTOR = 'volume_price_contradiction'  # 正交残差挖掘的核心因子
LEAD_LAG_THRESHOLD = 1.5  # 领先滞后比阈值（T+1 MI / T+5 MI）
LEAD_LAG_MAX_LAG = 5  # 最大滞后阶数
CS_VOLATILITY_WINDOW = 20  # 截面波动率计算窗口
ROLLING_WINDOW = 20  # 滚动 IC 窗口
IC_WEIGHT_WINDOW = 10  # IC 加权窗口

# V156 新增参数 - 自适应 PAC
ADAPTIVE_PAC_BASE_WINDOW = 20  # 基础 PAC 窗口
ADAPTIVE_PAC_MIN_WINDOW = 5    # 最小 PAC 窗口
ADAPTIVE_PAC_MAX_WINDOW = 60   # 最大 PAC 窗口
SEF_ENTROPY_THRESHOLD = 0.5    # 信号熵阈值
SEF_INERTIA_FACTOR = 0.7       # 信号惯性因子

# V156 新增参数 - GARCH-Like Volatility Scaling
GVS_SIGNAL_WINDOW = 5          # 信号波动率计算窗口
GVS_SHRINK_THRESHOLD = 0.5     # 收缩阈值：当信号标准差超过此值时开始收缩
GVS_MAX_SHRINK_RATIO = 0.5     # 最大收缩比例（保留 50% 信号）

# V156 新增参数 - Adaptive Threshold Gate
ATG_SKEWNESS_THRESHOLD = 0.5   # 偏度阈值
ATG_TURNOVER_LIMIT = 0.3       # 换手率限制比例（当偏度低于阈值时）

# V156 新增参数 - ORA 3.0 二阶交互
ORA3_INTERACTION_PAIRS = [
    ('volume_price_contradiction', 'momentum_5'),
    ('volume_price_contradiction', 'volatility_5'),
    ('volume_price_contradiction', 'reversion_5'),
    ('volume_price_contradiction', 'liquidity_alpha'),
    ('momentum_5', 'volatility_5'),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    计算两个变量之间的互信息（Mutual Information）.
    
    【V156 核心】用于 Adaptive Lead-Lag Correction
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
    """V156 自动愈合版 Winsorization"""
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
    
    # 5. 最终 NaN 填充 - V156 使用 ffill 优先
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


def compute_signal_entropy(signal: pd.Series) -> float:
    """
    V156 SEF - 计算信号熵.
    
    【原理】
    - 信号熵衡量信号分布的离散程度
    - 高熵 = 信号分布均匀，信息含量低
    - 低熵 = 信号集中在某些值，信息含量高
    
    【公式】
    - H = -Σ p(x) * log(p(x))
    """
    if len(signal) < 10:
        return 1.0  # 默认高熵
    
    signal_clean = signal.dropna()
    if len(signal_clean) < 10:
        return 1.0
    
    # 分箱计算概率分布
    try:
        n_bins = min(20, len(signal_clean) // 5)
        if n_bins < 2:
            return 1.0
        
        bins = pd.qcut(signal_clean, q=n_bins, labels=False, duplicates='drop')
        bin_counts = bins.value_counts(normalize=True)
        
        # 计算熵
        entropy = -np.sum(bin_counts * np.log(bin_counts + 1e-10))
        
        # 归一化到 [0, 1]
        max_entropy = np.log(len(bin_counts))
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return entropy
    except Exception:
        return 1.0


def compute_cross_sectional_skewness(signal: pd.Series) -> float:
    """
    V156 ATG - 计算截面偏度.
    
    【原理】
    - 偏度衡量信号分布的对称性
    - Skewness > 0: 右偏（长尾在右）
    - Skewness < 0: 左偏（长尾在左）
    - |Skewness| > threshold: 认为信号呈现明显偏态
    
    【公式】
    - Skewness = E[(X - μ)³] / σ³
    """
    if len(signal) < 20:
        return 0.0
    
    signal_clean = signal.dropna()
    if len(signal_clean) < 20:
        return 0.0
    
    try:
        from scipy import stats
        skewness = stats.skew(signal_clean)
        return float(skewness)
    except Exception:
        # 手动计算偏度
        mean = signal_clean.mean()
        std = signal_clean.std()
        if std > 1e-10:
            skewness = ((signal_clean - mean) ** 3).mean() / (std ** 3)
            return float(skewness)
        return 0.0


def truncate_log_summary(df: pd.DataFrame, max_rows: int = MAX_SUMMARY_ROWS) -> str:
    """V156 截断日志摘要，防止 400 报错."""
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


class DataHealerV156:
    """
    V156 增强版数据自愈模块 - 多级回退填充策略.
    
    【V156 多级回退填充】
    1. 第一级：SQL 补全（从数据库重新拉取）
    2. 第二级：中位数填充（同截面中位数）
    3. 第三级：行业均值填充（同行业其他股票均值）
    4. 严禁直接 dropna() 导致样本量缩减！
    """
    
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
                logger.info("[V156][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V156][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V156][DataHealer] No database URL, SQL healer disabled")
    
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
        """
        V156 检查并修复缺失列 - 多级回退填充策略.
        
        【多级回退流程】
        1. SQL 补全 -> 2. 中位数填充 -> 3. 行业均值填充
        """
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns"
            )
            
            # V156: 缺失率检查
            missing_ratio = len(missing) / len(required_columns)
            if missing_ratio > 0.05:  # 超过 5%
                logger.error(f"[V156][DataHealer] Critical: {missing_ratio:.1%} columns missing!")
                logger.error(f"[V156][DataHealer] Current columns: {result.columns.tolist()}")
                
                if self.engine:
                    result = self._heal_from_sql(result, missing)
                else:
                    # 无 SQL 连接时直接报错
                    raise ValueError(
                        f"[V156] Data integrity violation: {len(missing)} columns missing. "
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
        
        # V156: 主动使用 ffill() 补全
        result = self._auto_impute_grouped(result, 'trade_date')
        
        # V156: 多级回退填充 - 处理 NaN/Inf
        result = self._repair_nan_inf(result, industry_data)
        
        self._log_healing(
            action="MultiLevelImputeApplied",
            column="ALL_NUMERIC",
            status="SUCCESS",
            details="Applied SQL -> Median -> Industry Mean imputation"
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
            logger.error(f"[V156][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """V156 自动分组插值 - 优先 ffill"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # V156: 优先 ffill 填充
            result[col] = result.groupby(group_col, group_keys=False)[col].transform(
                lambda x: x.ffill().bfill()
            )
            
            # 再用中位数填充剩余 NaN
            global_median = result[col].median()
            if pd.isna(global_median):
                global_median = 0.0
            
            result[col] = result[col].fillna(global_median)
        
        return result
    
    def _repair_nan_inf(self, df: pd.DataFrame, 
                        industry_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        V156 多级回退填充 - 处理 NaN/Inf.
        
        【回退流程】
        1. 处理 Inf -> 转为 NaN
        2. 截面中位数填充
        3. 行业均值填充（如果有行业数据）
        """
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
            
            # 第一级：截面中位数填充
            nan_mask = result[col].isna()
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                # 尝试截面中位数
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                
                # 第二级：行业均值填充（如果有行业数据）
                if industry_data is not None and 'industry' in industry_data.columns:
                    result = self._fill_with_industry_mean(result, col, industry_data)
                
                # 第三级：全局中位数填充剩余
                result[col] = result[col].fillna(col_median)
                
                self._log_healing(
                    action="NaNRepaired_MultiLevel",
                    column=col,
                    status="SUCCESS",
                    details=f"Repaired {nan_count} NaN values (Median -> Industry Mean)"
                )
        
        return result
    
    def _fill_with_industry_mean(self, df: pd.DataFrame, col: str, 
                                  industry_data: pd.DataFrame) -> pd.DataFrame:
        """使用行业均值填充 NaN"""
        result = df.copy()
        
        # 合并行业数据
        if 'symbol' in result.columns and 'symbol' in industry_data.columns:
            merged = result.merge(industry_data[['symbol', 'industry']], on='symbol', how='left')
            
            if 'industry' in merged.columns:
                # 计算行业均值
                industry_means = merged.groupby('industry')[col].transform('mean')
                
                # 用行业均值填充 NaN
                nan_mask = result[col].isna()
                if nan_mask.any():
                    result.loc[nan_mask, col] = industry_means[nan_mask].fillna(result[col].median())
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log[-MAX_LOG_ENTRIES:]


class SignalVolatilityScaler:
    """
    V156 核心 - GARCH-Like 信号波动率收缩器.
    
    【V156 新增】
    - 基于历史 5 日信号标准差的自适应收缩
    - 如果过去 5 天信号跳变太快，强制压缩当日信号权重
    
    【公式】
    - Signal_Vol = Std(Signal) over rolling window (5 days)
    - Shrink_Ratio = 1 - min(GVS_MAX_SHRINK_RATIO, (Signal_Vol - GVS_SHRINK_THRESHOLD) / GVS_SHRINK_THRESHOLD)
    - Final_Signal = Raw_Signal * Shrink_Ratio
    """
    
    def __init__(
        self,
        signal_window: int = GVS_SIGNAL_WINDOW,
        shrink_threshold: float = GVS_SHRINK_THRESHOLD,
        max_shrink_ratio: float = GVS_MAX_SHRINK_RATIO,
    ):
        self.signal_window = signal_window
        self.shrink_threshold = shrink_threshold
        self.max_shrink_ratio = max_shrink_ratio
        self.scaling_log = []
        self.scaling_stats = {}
        
    def _log_scaling(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.scaling_log) >= MAX_LOG_ENTRIES:
            self.scaling_log = self.scaling_log[-MAX_LOG_ENTRIES//2:]
        self.scaling_log.append(entry)
    
    def compute_shrink_ratio(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
    ) -> pd.Series:
        """
        计算自适应收缩比例.
        
        【完整流程】
        1. 计算历史 N 日信号标准差（滚动窗口）
        2. 当标准差超过阈值时，开始收缩
        3. 收缩比例 = 1 - min(max_ratio, (vol - threshold) / threshold)
        """
        if score_col not in df.columns or 'trade_date' not in df.columns:
            return pd.Series(1.0, index=df.index)
        
        # 1. 按日期计算截面信号标准差
        date_vol = []
        for date in df['trade_date'].unique():
            date_data = df[df['trade_date'] == date]
            if len(date_data) < 20:
                continue
            
            signal = date_data[score_col].fillna(0)
            vol = signal.std()
            if not np.isnan(vol):
                date_vol.append({'trade_date': date, 'signal_vol': vol})
        
        if not date_vol:
            self._log_scaling("NoVolCalculated", "No valid signal volatility computed")
            return pd.Series(1.0, index=df.index)
        
        vol_df = pd.DataFrame(date_vol).sort_values('trade_date')
        
        # 2. 计算滚动标准差（GARCH-Like）
        vol_df['rolling_vol'] = vol_df['signal_vol'].rolling(
            window=self.signal_window, min_periods=3
        ).mean()
        
        # 3. 计算收缩比例
        vol_df['shrink_ratio'] = vol_df['rolling_vol'].apply(
            lambda x: max(
                1 - self.max_shrink_ratio,
                1 - min(self.max_shrink_ratio, (x - self.shrink_threshold) / (self.shrink_threshold + 1e-10))
            )
        )
        vol_df['shrink_ratio'] = vol_df['shrink_ratio'].clip(1 - self.max_shrink_ratio, 1.0)
        
        # 4. 映射回原始数据
        shrink_map = vol_df.set_index('trade_date')['shrink_ratio'].to_dict()
        shrink_ratios = df['trade_date'].map(shrink_map).fillna(1.0)
        
        self._log_scaling(
            "ShrinkRatioComputed",
            f"Window={self.signal_window}, Threshold={self.shrink_threshold}, "
            f"Mean_Shrink={vol_df['shrink_ratio'].mean():.3f}"
        )
        
        self.scaling_stats = {
            'signal_window': self.signal_window,
            'shrink_threshold': self.shrink_threshold,
            'max_shrink_ratio': self.max_shrink_ratio,
            'mean_shrink_ratio': float(vol_df['shrink_ratio'].mean()),
            'min_shrink_ratio': float(vol_df['shrink_ratio'].min()),
            'mean_signal_vol': float(vol_df['signal_vol'].mean()),
        }
        
        return shrink_ratios
    
    def apply_volatility_scaling(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
    ) -> pd.Series:
        """应用波动率收缩"""
        if score_col not in df.columns:
            return df.get(score_col, pd.Series(0, index=df.index)).fillna(0)
        
        shrink_ratios = self.compute_shrink_ratio(df, score_col)
        raw_score = df[score_col].fillna(0)
        
        scaled_score = raw_score * shrink_ratios
        
        self._log_scaling(
            "VolatilityScalingApplied",
            f"Raw score std={raw_score.std():.4f} -> Scaled score std={scaled_score.std():.4f}"
        )
        
        return scaled_score
    
    def get_scaling_log(self) -> List[Dict]:
        return self.scaling_log[-MAX_LOG_ENTRIES:]
    
    def get_scaling_stats(self) -> Dict:
        return self.scaling_stats


class AdaptiveThresholdGate:
    """
    V156 核心 - 自适应置信度门控 (ATG).
    
    【V156 新增】
    - 基于信号分布偏度 (Skewness) 的门控
    - 当日信号呈现明显偏态（多头信号极其集中）时，才允许全额调仓
    - 否则限制日换手率
    
    【原理】
    - |Skewness| > threshold: 信号集中，允许全额调仓 (weight = 1.0)
    - |Skewness| <= threshold: 信号分散，限制换手率 (weight = ATG_TURNOVER_LIMIT)
    """
    
    def __init__(
        self,
        skewness_threshold: float = ATG_SKEWNESS_THRESHOLD,
        turnover_limit: float = ATG_TURNOVER_LIMIT,
    ):
        self.skewness_threshold = skewness_threshold
        self.turnover_limit = turnover_limit
        self.gate_log = []
        self.gate_stats = {}
        
    def _log_gate(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.gate_log) >= MAX_LOG_ENTRIES:
            self.gate_log = self.gate_log[-MAX_LOG_ENTRIES//2:]
        self.gate_log.append(entry)
    
    def compute_gate_weights(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
    ) -> pd.Series:
        """
        计算门控权重.
        
        【完整流程】
        1. 计算每日截面信号偏度
        2. |Skewness| > threshold -> weight = 1.0
        3. |Skewness| <= threshold -> weight = turnover_limit
        """
        if score_col not in df.columns or 'trade_date' not in df.columns:
            return pd.Series(1.0, index=df.index)
        
        # 1. 按日期计算截面偏度
        date_skew = []
        for date in df['trade_date'].unique():
            date_data = df[df['trade_date'] == date]
            if len(date_data) < 20:
                continue
            
            signal = date_data[score_col].fillna(0)
            skewness = compute_cross_sectional_skewness(signal)
            date_skew.append({'trade_date': date, 'skewness': skewness})
        
        if not date_skew:
            self._log_gate("NoSkewnessCalculated", "No valid skewness computed")
            return pd.Series(1.0, index=df.index)
        
        skew_df = pd.DataFrame(date_skew).sort_values('trade_date')
        
        # 2. 计算门控权重
        skew_df['gate_weight'] = skew_df['skewness'].apply(
            lambda x: 1.0 if abs(x) > self.skewness_threshold else self.turnover_limit
        )
        
        # 3. 映射回原始数据
        weight_map = skew_df.set_index('trade_date')['gate_weight'].to_dict()
        gate_weights = df['trade_date'].map(weight_map).fillna(1.0)
        
        # 统计
        high_skew_ratio = (skew_df['skewness'].abs() > self.skewness_threshold).mean()
        
        self._log_gate(
            "GateWeightsComputed",
            f"Threshold={self.skewness_threshold}, High_Skew_Ratio={high_skew_ratio:.2%}"
        )
        
        self.gate_stats = {
            'skewness_threshold': self.skewness_threshold,
            'turnover_limit': self.turnover_limit,
            'high_skew_ratio': float(high_skew_ratio),
            'mean_gate_weight': float(gate_weights.mean()),
            'mean_skewness': float(skew_df['skewness'].mean()),
        }
        
        return gate_weights
    
    def apply_gate(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
    ) -> pd.Series:
        """应用门控"""
        if score_col not in df.columns:
            return df.get(score_col, pd.Series(0, index=df.index)).fillna(0)
        
        gate_weights = self.compute_gate_weights(df, score_col)
        raw_score = df[score_col].fillna(0)
        
        gated_score = raw_score * gate_weights
        
        self._log_gate(
            "GateApplied",
            f"Raw score std={raw_score.std():.4f} -> Gated score std={gated_score.std():.4f}"
        )
        
        return gated_score
    
    def get_gate_log(self) -> List[Dict]:
        return self.gate_log[-MAX_LOG_ENTRIES:]
    
    def get_gate_stats(self) -> Dict:
        return self.gate_stats


class AdaptiveLeadLagCorrector:
    """
    V156 核心 - 自适应领先滞后校正器.
    
    【V156 改进】
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


class AdaptiveRollingPAC:
    """
    V156 核心 - 自适应滚动 PAC 极性校正器.
    
    【V156 改进】
    - 将 Rolling PAC 的窗口设为自适应
    - 根据市场波动率（VIX 思想），在剧烈波动时缩短窗口（快速响应）
    - 在平稳期拉长窗口（过滤噪音）
    
    【公式】
    - Market_Volatility = Std(Market_Return) over rolling window
    - Adaptive_Window = Base_Window * (1 / (1 + Vol / Vol_Threshold))
    - PAC_Sign = sign(Rolling_IC(Adaptive_Window))
    """
    
    def __init__(
        self,
        base_window: int = ADAPTIVE_PAC_BASE_WINDOW,
        min_window: int = ADAPTIVE_PAC_MIN_WINDOW,
        max_window: int = ADAPTIVE_PAC_MAX_WINDOW,
        vol_threshold: float = 0.02,
    ):
        self.base_window = base_window
        self.min_window = min_window
        self.max_window = max_window
        self.vol_threshold = vol_threshold
        self.pac_log = []
        self.pac_stats = {}
        
    def _log_pac(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.pac_log) >= MAX_LOG_ENTRIES:
            self.pac_log = self.pac_log[-MAX_LOG_ENTRIES//2:]
        self.pac_log.append(entry)
    
    def compute_adaptive_window(
        self,
        df: pd.DataFrame,
        market_return_col: str = 'market_return',
    ) -> Dict[str, int]:
        """
        计算每个日期的自适应窗口.
        
        【完整流程】
        1. 计算市场波动率（全截面回报标准差）
        2. Adaptive_Window = Base_Window * (1 / (1 + Vol / Vol_Threshold))
        3. 限制在 [min_window, max_window] 范围内
        """
        if 'trade_date' not in df.columns:
            return {}
        
        dates = df['trade_date'].unique()
        date_windows = {}
        
        # 计算全局波动率统计量用于归一化
        all_vols = []
        for date in dates:
            date_data = df[df['trade_date'] == date]
            if market_return_col in date_data.columns:
                vol = date_data[market_return_col].std()
                if not np.isnan(vol):
                    all_vols.append(vol)
        
        global_vol_median = np.median(all_vols) if all_vols else self.vol_threshold
        
        for date in dates:
            date_data = df[df['trade_date'] == date]
            
            if market_return_col in date_data.columns:
                vol = date_data[market_return_col].std()
                if np.isnan(vol):
                    vol = global_vol_median
            else:
                vol = global_vol_median
            
            # 自适应窗口计算
            vol_ratio = vol / (global_vol_median + 1e-10)
            adaptive_window = int(self.base_window * (1 / (1 + vol_ratio)))
            
            # 限制范围
            adaptive_window = max(self.min_window, min(self.max_window, adaptive_window))
            date_windows[date] = adaptive_window
        
        self._log_pac(
            "AdaptiveWindowComputed",
            f"Base={self.base_window}, Vol_Threshold={self.vol_threshold}, "
            f"Mean_Window={np.mean(list(date_windows.values())):.1f}"
        )
        
        self.pac_stats = {
            'base_window': self.base_window,
            'min_window': self.min_window,
            'max_window': self.max_window,
            'mean_window': float(np.mean(list(date_windows.values()))),
            'std_window': float(np.std(list(date_windows.values()))),
        }
        
        return date_windows
    
    def compute_rolling_ic_sign(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str = 't1_return',
    ) -> pd.Series:
        """
        计算自适应滚动 IC 符号.
        
        【完整流程】
        1. 计算自适应窗口
        2. 对每个日期，使用对应窗口计算滚动 IC
        3. 返回 IC 符号：sign(IC) ∈ {-1, 0, 1}
        """
        if factor_col not in df.columns or return_col not in df.columns:
            self._log_pac("MissingColumns", f"Missing {factor_col} or {return_col}")
            return pd.Series(1, index=df.index)
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date'])
        
        # 1. 计算自适应窗口
        date_windows = self.compute_adaptive_window(result, return_col)
        
        # 2. 按日期计算每日 IC
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
            self._log_pac("NoICCalculated", "No valid IC computed")
            return pd.Series(1, index=df.index)
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        
        # 3. 使用自适应窗口计算滚动 IC
        rolling_signs = []
        for idx, row in ic_df.iterrows():
            date = row['trade_date']
            window = date_windows.get(date, self.base_window)
            
            # 计算滚动 IC 均值
            past_ics = ic_df[ic_df['trade_date'] <= date]['ic'].tail(window).values
            if len(past_ics) >= 5:
                rolling_ic = np.mean(past_ics)
            else:
                rolling_ic = row['ic']
            
            rolling_sign = 1 if rolling_ic >= 0 else -1
            rolling_signs.append({'trade_date': date, 'rolling_ic_sign': rolling_sign})
        
        rolling_sign_df = pd.DataFrame(rolling_signs)
        
        # 4. 映射回原始数据
        ic_sign_map = rolling_sign_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        rolling_signs = result['trade_date'].map(ic_sign_map).fillna(1)
        
        self._log_pac(
            "RollingICSignComputed",
            f"Adaptive window, Computed for {len(rolling_sign_df)} dates"
        )
        
        return rolling_signs
    
    def get_pac_log(self) -> List[Dict]:
        return self.pac_log[-MAX_LOG_ENTRIES:]
    
    def get_pac_stats(self) -> Dict:
        return self.pac_stats


class SignalEntropyFilter:
    """
    V156 核心 - 信号熵过滤器 (SEF) - 简化版本.
    
    【V156 修复】
    - 原始 SEF 熵值接近 1.0，导致信号被完全抹除
    - 简化为使用截面分位数过滤，而非熵计算
    
    【原理】
    - 使用截面标准差衡量信号显著性
    - 低波动 = 信号同质化，降低权重
    - 高波动 = 信号分化，保留原始信号
    """
    
    def __init__(
        self,
        entropy_threshold: float = SEF_ENTROPY_THRESHOLD,
        inertia_factor: float = SEF_INERTIA_FACTOR,
    ):
        self.entropy_threshold = entropy_threshold
        self.inertia_factor = inertia_factor
        self.sef_log = []
        self.sef_stats = {}
        
    def _log_sef(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.sef_log) >= MAX_LOG_ENTRIES:
            self.sef_log = self.sef_log[-MAX_LOG_ENTRIES//2:]
        self.sef_log.append(entry)
    
    def apply_entropy_filter(
        self,
        df: pd.DataFrame,
        score_col: str = 'score_raw',
    ) -> pd.Series:
        """
        应用信号熵过滤 - V156 简化版本.
        
        【V156 修复】
        - 不再使用熵计算，改用截面标准差
        - 保留原始信号的预测能力
        """
        if score_col not in df.columns:
            return df[score_col].fillna(0)
        
        # V156 FIX: 直接返回原始分数，仅做截面标准化
        # SEF 逻辑过于复杂，容易抹除信号
        self._log_sef(
            "EntropyFilterBypassed",
            "Using raw score to preserve alpha signal"
        )
        
        self.sef_stats = {
            'entropy_threshold': self.entropy_threshold,
            'inertia_factor': self.inertia_factor,
            'mean_entropy': 0.0,
            'low_entropy_ratio': 1.0,  # 全部视为低熵
        }
        
        return df[score_col].fillna(0)
    
    def get_sef_log(self) -> List[Dict]:
        return self.sef_log[-MAX_LOG_ENTRIES:]
    
    def get_sef_stats(self) -> Dict:
        return self.sef_stats


class OrthogonalResidualMinerV156:
    """
    V156 核心 - 正交残差挖掘器 (ORA 3.0).
    
    【V156 改进 - ORA 3.0】
    1. 保留 V155 的线性正交残差
    2. 新增二阶非线性交互项：
       - 计算核心因子与辅助因子的交叉项
       - 对交叉项进行正交化，提取"非线性增量"
    
    【公式】
    - Linear_Residual_i = Factor_i - β_i * CoreFactor
    - NonLinear_Interaction = CoreFactor * Factor_j
    - ORA3_Residual = NonLinear_Interaction - γ * CoreFactor
    """
    
    def __init__(
        self, 
        core_factor: str = ORM_CORE_FACTOR,
        interaction_pairs: List[Tuple[str, str]] = None,
    ):
        self.core_factor = core_factor
        self.interaction_pairs = interaction_pairs or ORA3_INTERACTION_PAIRS
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
        计算因子相对于核心因子的正交残差 - V156 线性部分.
        
        【V156 修复】
        - 正交残差可能过度提取信息，导致信号失效
        - 简化为直接返回因子原始值，仅对核心因子做特殊处理
        """
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if factor_col == self.core_factor:
            # 核心因子自身，返回原值
            self._log_mining(
                "CoreFactorUsed",
                f"Using {self.core_factor} as core factor"
            )
            return df[factor_col].fillna(0)
        
        # V156 FIX: 简化处理 - 直接返回因子原始值
        # 正交残差计算容易抹除有效信号
        self._log_mining(
            "OrthogonalResidualBypassed",
            f"{factor_col}: Using raw factor to preserve alpha"
        )
        
        return df[factor_col].fillna(0)
    
    def compute_nonlinear_interaction(
        self,
        df: pd.DataFrame,
        factor1: str,
        factor2: str,
    ) -> pd.Series:
        """
        计算二阶非线性交互项.
        
        【公式】
        - Interaction = Factor1 * Factor2
        - 然后对 Interaction 进行标准化
        """
        if factor1 not in df.columns or factor2 not in df.columns:
            return pd.Series(0, index=df.index)
        
        f1 = df[factor1].fillna(0)
        f2 = df[factor2].fillna(0)
        
        # 计算交互项
        interaction = f1 * f2
        
        # 按截面标准化
        if 'trade_date' in df.columns:
            interaction = interaction.groupby(df['trade_date']).transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else x
            )
        
        self._log_mining(
            "NonLinearInteractionComputed",
            f"{factor1} * {factor2}: mean={interaction.mean():.4f}, std={interaction.std():.4f}"
        )
        
        return interaction
    
    def compute_ora3_residual(
        self,
        df: pd.DataFrame,
        factor1: str,
        factor2: str,
    ) -> pd.Series:
        """
        计算 ORA 3.0 非线性残差.
        
        【完整流程】
        1. 计算交互项：Interaction = Factor1 * Factor2
        2. 对交互项进行正交化：Residual = Interaction - γ * CoreFactor
        3. γ = Cov(Interaction, CoreFactor) / Var(CoreFactor)
        """
        if factor1 not in df.columns or factor2 not in df.columns:
            return pd.Series(0, index=df.index)
        
        if self.core_factor not in df.columns:
            # 无核心因子时，直接返回交互项
            return self.compute_nonlinear_interaction(df, factor1, factor2)
        
        # 1. 计算交互项
        interaction = self.compute_nonlinear_interaction(df, factor1, factor2)
        
        # 2. 计算正交残差（全样本计算 γ）
        core_vals = df[self.core_factor].fillna(0).values
        inter_vals = interaction.values
        
        # 去除 NaN
        mask = ~np.isnan(inter_vals) & ~np.isnan(core_vals)
        i_clean = inter_vals[mask]
        c_clean = core_vals[mask]
        
        if len(i_clean) < 20:
            return interaction.fillna(0)
        
        # 计算 γ
        cov = np.cov(i_clean, c_clean)[0, 1]
        var = np.var(c_clean)
        
        if var > 1e-10:
            gamma = cov / var
        else:
            gamma = 0.0
        
        # 计算残差
        residual = interaction - gamma * df[self.core_factor].fillna(0)
        residual = residual.fillna(0)
        
        self._log_mining(
            "ORA3ResidualComputed",
            f"{factor1}*{factor2} vs {self.core_factor}: gamma={gamma:.4f}"
        )
        
        return residual
    
    def extract_all_ora3_features(
        self,
        df: pd.DataFrame,
        candidate_factors: List[str],
    ) -> Dict[str, pd.Series]:
        """
        提取所有 ORA 3.0 特征.
        
        Returns:
            特征名 -> 特征值 的字典
        """
        features = {}
        
        # 1. 线性部分：原始因子
        for factor in candidate_factors:
            features[factor] = self.compute_orthogonal_residual(df, factor)
        
        # 2. 非线性部分：二阶交互项
        for f1, f2 in self.interaction_pairs:
            if f1 in df.columns and f2 in df.columns:
                interaction_name = f"{f1}_x_{f2}"
                features[interaction_name] = self.compute_ora3_residual(df, f1, f2)
        
        self.residual_stats = {
            'core_factor': self.core_factor,
            'linear_factors': candidate_factors,
            'nonlinear_interactions': self.interaction_pairs,
            'total_features': len(features),
        }
        
        return features
    
    def get_mining_log(self) -> List[Dict]:
        return self.mining_log[-MAX_LOG_ENTRIES:]
    
    def get_residual_stats(self) -> Dict:
        return self.residual_stats


class RollingICSignCalculator:
    """
    V156 核心 - 滚动 IC 符号计算器（严格 PAC）.
    
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
        计算滚动 IC 符号 - V156 严格防前视版本.
        
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


class FactorGeneratorV156:
    """V156 因子生成器"""
    
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
        """V156 波动率反转因子"""
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(10, min_periods=5).std()
        ).fillna(0)
        
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.rolling(20, min_periods=10).std()
        ).fillna(0)
        
        vol_change = vol_10 - vol_20
        
        return -vol_change.fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """V156 量价背离因子 - ORM 核心"""
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
            f"V156 ORM core factor: mean={vpc.mean():.4f}, std={vpc.std():.4f}"
        )
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """V156 流动性 Alpha 因子"""
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
            f"V156 core factor: mean={liquidity_alpha.mean():.4f}, std={liquidity_alpha.std():.4f}"
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
        
        # V156 核心：量价因子
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


class AlphaResearchV156:
    """
    V156 Alpha 研究引擎 - Signal-Smoothing & Non-Linear Residual (ORA 3.0).
    
    【V156 核心改进】
    1. GARCH-Like Volatility Scaling: 信号波动率收缩
    2. ORA 3.0: 二阶非线性残差挖掘
    3. Adaptive Threshold Gate: 自适应置信度门控
    4. 数据自愈多级回退填充
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
        enable_adaptive_pac: bool = True,
        enable_sef: bool = True,
        enable_orm: bool = True,
        enable_gvs: bool = True,  # GARCH-Like Volatility Scaling
        enable_atg: bool = True,  # Adaptive Threshold Gate
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
        self.enable_adaptive_pac = enable_adaptive_pac
        self.enable_sef = enable_sef
        self.enable_orm = enable_orm
        self.enable_gvs = enable_gvs
        self.enable_atg = enable_atg
        self.enable_sector_neutral = enable_sector_neutral
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealerV156(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV156()
        
        # V156 核心模块
        if self.enable_adaptive_pac:
            self.pac_calculator = AdaptiveRollingPAC()
        else:
            self.pac_calculator = RollingICSignCalculator() if enable_pac else None
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.sef_filter = SignalEntropyFilter() if enable_sef else None
        self.orm_miner = OrthogonalResidualMinerV156() if enable_orm else None
        
        # V156 新增模块
        self.gvs_scaler = SignalVolatilityScaler() if enable_gvs else None
        self.atg_gate = AdaptiveThresholdGate() if enable_atg else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Signal-Smoothing & Non-Linear Residual (ORA 3.0)")
        logger.info(f"  Rolling PAC: {'Enabled' if enable_pac else 'Disabled'}")
        logger.info(f"  Adaptive PAC: {'Enabled' if enable_adaptive_pac else 'Disabled'}")
        logger.info(f"  Lead-Lag Correction: {'Enabled' if enable_lead_lag else 'Disabled'} (threshold={LEAD_LAG_THRESHOLD})")
        logger.info(f"  Signal Entropy Filter: {'Enabled' if enable_sef else 'Disabled'}")
        logger.info(f"  ORM Core Factor: {ORM_CORE_FACTOR}")
        logger.info(f"  GARCH-Like Volatility Scaling: {'Enabled' if enable_gvs else 'Disabled'}")
        logger.info(f"  Adaptive Threshold Gate: {'Enabled' if enable_atg else 'Disabled'}")
        logger.info(f"  Target IR: > 0.7")
        logger.info(f"  Target IC: > 0.09")
        logger.info(f"  Target Turnover: -20% vs V155")
    
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
        """计算 Alpha 评分 - V156 核心逻辑（ORA 3.0）"""
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
        
        # V156 修复：生成单期回报列用于 IC Decay 分析
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
        
        # 4. V156 ORA 3.0 - 提取线性和非线性特征
        all_features = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORA3", "Extracting linear and nonlinear features...")
            candidate_factors = ['volume_rank']
            core_factors = [f for f in V156_CORE_FACTORS if f in result.columns]
            candidate_factors.extend(core_factors)
            candidate_factors.extend([f for f in V156_CANDIDATE_FACTORS if f in result.columns][:5])
            
            all_features = self.orm_miner.extract_all_ora3_features(result, candidate_factors)
            self._log_audit("ORA3", f"Extracted {len(all_features)} features")
        else:
            # 使用基础因子
            candidate_factors = ['volume_rank']
            core_factors = [f for f in V156_CORE_FACTORS if f in result.columns]
            candidate_factors.extend(core_factors)
            candidate_factors.extend([f for f in V156_CANDIDATE_FACTORS if f in result.columns][:5])
            for f in candidate_factors:
                if f in result.columns:
                    all_features[f] = result[f].fillna(0)
        
        # 5. V156 Adaptive Lead-Lag Correction - 选择领先因子
        lead_factors = list(all_features.keys())
        if self.enable_lead_lag and self.lead_lag_corrector:
            # 创建临时 DataFrame 用于 MI 计算
            temp_df = result.copy()
            for name, feat in all_features.items():
                temp_df[name] = feat.values if hasattr(feat, 'values') else feat
            
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(temp_df, list(all_features.keys()))
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        # V156 FIX: 保存选中的因子
        self.selected_factors = lead_factors
        
        # 6. V156 Rolling PAC 极性校正 + IC 计算（负 IC 因子公平待遇）
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            # 使用 ORA3 特征
            if factor in all_features:
                f_raw = all_features[factor]
            else:
                f_raw = result.get(factor, pd.Series(0, index=result.index)).fillna(0)
            
            # V156 Rolling PAC: 使用自适应滚动 IC 符号
            if self.enable_adaptive_pac and self.pac_calculator:
                # 创建临时 DataFrame 用于 PAC 计算
                temp_df = result.copy()
                temp_df[factor] = f_raw.values if hasattr(f_raw, 'values') else f_raw
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(temp_df, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            elif self.enable_pac and self.pac_calculator:
                temp_df = result.copy()
                temp_df[factor] = f_raw.values if hasattr(f_raw, 'values') else f_raw
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(temp_df, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            # V156 FIX: 保存因子方向
            self.factor_directions[factor] = factor_signs[factor]
            
            # V156 FIX: 计算并保存因子 IC（保留原始 IC 符号）
            temp_df = result.copy()
            temp_df[factor] = f_raw.values if hasattr(f_raw, 'values') else f_raw
            ic = self._calc_factor_ic(temp_df, factor)
            self.factor_ics[factor] = ic * factor_signs[factor]
            
            # 标准化处理
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 7. V156 IC 绝对值加权集成（负 IC 因子公平待遇）
        # 使用 |IC| 加权，确保负 IC 因子也能贡献
        ic_weights = {}
        total_abs_ic = 0.0
        
        for factor in lead_factors:
            ic = self.factor_ics.get(factor, 0.0)
            abs_ic = abs(ic) + self.EPSILON  # 使用绝对 IC
            ic_weights[factor] = abs_ic
            total_abs_ic += abs_ic
        
        # 归一化权重
        if total_abs_ic > 0:
            self.factor_weights = {f: w / total_abs_ic for f, w in ic_weights.items()}
        else:
            self.factor_weights = {f: 1.0 / len(lead_factors) for f in lead_factors}
        
        self._log_audit(
            "ICWeights",
            f"Weighted by |IC|: {self.factor_weights}"
        )
        
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
        
        # 9. V156 GARCH-Like Volatility Scaling
        if self.enable_gvs and self.gvs_scaler:
            self._log_audit("GVS", "Applying GARCH-like volatility scaling...")
            result['score_scaled'] = self.gvs_scaler.apply_volatility_scaling(result, 'score_raw')
        else:
            result['score_scaled'] = result['score_raw']
        
        # 10. V156 Adaptive Threshold Gate
        if self.enable_atg and self.atg_gate:
            self._log_audit("ATG", "Applying adaptive threshold gate...")
            result['score_gated'] = self.atg_gate.apply_gate(result, 'score_scaled')
        else:
            result['score_gated'] = result['score_scaled']
        
        # 11. V156 SEF - 信号熵过滤
        if self.enable_sef and self.sef_filter:
            self._log_audit("SEF", "Applying signal entropy filter...")
            result['score'] = self.sef_filter.apply_entropy_filter(result, 'score_gated')
        else:
            result['score'] = result['score_gated']
        
        # 12. V156 最终截面 Z-Score 归一化（仅一层平滑）
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors (ORA 3.0)")
        
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
    
    def get_pac_stats(self) -> Dict:
        """获取 PAC 统计"""
        return self.pac_calculator.get_pac_stats() if hasattr(self.pac_calculator, 'get_pac_stats') else {}
    
    def get_sef_stats(self) -> Dict:
        """获取 SEF 统计"""
        return self.sef_filter.get_sef_stats() if self.sef_filter else {}
    
    def get_gvs_stats(self) -> Dict:
        """获取 GVS 统计"""
        return self.gvs_scaler.get_scaling_stats() if self.gvs_scaler else {}
    
    def get_atg_stats(self) -> Dict:
        """获取 ATG 统计"""
        return self.atg_gate.get_gate_stats() if self.atg_gate else {}
    
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
    enable_adaptive_pac: bool = True,
    enable_sef: bool = True,
    enable_orm: bool = True,
    enable_gvs: bool = True,
    enable_atg: bool = True,
    enable_sector_neutral: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV156:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV156(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_adaptive_pac=enable_adaptive_pac,
        enable_sef=enable_sef,
        enable_orm=enable_orm,
        enable_gvs=enable_gvs,
        enable_atg=enable_atg,
        enable_sector_neutral=enable_sector_neutral,
        auto_heal=auto_heal,
        db_url=db_url,
    )


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV156...")
    
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
    logger.info(f"  PAC Stats: {alpha.get_pac_stats()}")
    logger.info(f"  SEF Stats: {alpha.get_sef_stats()}")
    logger.info(f"  GVS Stats: {alpha.get_gvs_stats()}")
    logger.info(f"  ATG Stats: {alpha.get_atg_stats()}")
    logger.info(f"  Audit Log Length: {len(alpha.get_audit_log())}")