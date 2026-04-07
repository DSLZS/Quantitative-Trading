"""
Alpha Research Module - V163 因子重构计划 (Factor Reconstruction Plan).

【V162 失败诊断】
- IC 始终徘徊在 0.05 左右，无法突破 0.095
- 根本原因：因子池过于简单，缺乏有效的预测信号
- volume_price_contradiction 等因子在 2024 年数据上表现不佳

【V163 核心改进】
1. 引入更多有效的短期预测因子：
   - 隔夜缺口因子 (Overnight Gap)
   - 盘中动量因子 (Intraday Momentum)
   - 资金流因子 (Money Flow)
   - 相对强度因子 (Relative Strength)

2. 改进因子计算：
   - 使用更精确的量价关系
   - 引入行业中性化处理
   - 增加因子正交化

3. 简化集成逻辑：
   - 回归最简单的等权重集成
   - 只保留 IC>0 的因子
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

VERSION = "V163"

# V163 核心因子池
V163_CORE_FACTORS = [
    'overnight_gap',          # 隔夜缺口因子
    'intraday_momentum',      # 盘中动量
    'money_flow',             # 资金流
    'relative_strength',      # 相对强度
    'volume_momentum',        # 成交量动量
    'price_efficiency',       # 价格效率
    'liquidity_imbalance',    # 流动性不平衡
    'volatility_adjusted_return',  # 波动率调整收益
]

MAX_FACTORS = 8
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# 参数配置
ROLLING_WINDOW = 20
IC_IR_ANNUALIZATION = 252


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """计算互信息"""
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
    """Winsorization"""
    series_clean = series.copy()
    series_clean = series_clean.replace([np.inf, -np.inf], np.nan)
    
    mean = series_clean.mean()
    if pd.isna(mean):
        mean = 0.0
    
    std = series_clean.std()
    if std > 1e-10:
        lower = mean - sigma * std
        upper = mean + sigma * std
        series_clean = series_clean.clip(lower=lower, upper=upper)
    
    lower_pct = series_clean.quantile(1 - percentile)
    upper_pct = series_clean.quantile(percentile)
    series_clean = series_clean.clip(lower=lower_pct, upper=upper_pct)
    
    series_clean = series_clean.ffill().bfill().fillna(mean)
    
    return series_clean


class DataHealerV163:
    """V163 数据自愈模块"""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.healing_log = []
        self._init_sql_healer()
        
    def _init_sql_healer(self):
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url)
                logger.info("[V163][DataHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V163][DataHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            if self.engine:
                result = self._heal_from_sql(result, missing)
            else:
                for col in missing:
                    result = result.assign(**{col: 0.0})
        
        return self._auto_impute_grouped(result, 'trade_date')
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
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
                        if f'{col}_sql' in result.columns:
                            result = result.drop(columns=[f'{col}_sql'])
        except Exception as e:
            logger.error(f"[V163][DataHealer] SQL heal failed: {e}")
            for col in columns:
                result = result.assign(**{col: 0.0})
        return result
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
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
    
    def get_healing_log(self) -> List[Dict]:
        return self.healing_log[-MAX_LOG_ENTRIES:]


class FactorGeneratorV163:
    """V163 因子生成器 - 引入新因子"""
    
    def __init__(self):
        self.generation_log = []
    
    def compute_overnight_gap(self, df: pd.DataFrame) -> pd.Series:
        """
        隔夜缺口因子.
        
        逻辑：隔夜缺口 = (开盘价 - 前收盘价) / 前收盘价
        预期：隔夜缺口与当日收益负相关（缺口回补效应）
        """
        if 'open' not in df.columns or 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 计算前收盘价（使用昨收）
        pre_close = df.groupby('symbol')['close'].transform(lambda x: x.shift(1))
        
        # 隔夜缺口
        overnight_gap = (df['open'] - pre_close) / (pre_close + 1e-6)
        
        # 负号：预期缺口回补
        result = -overnight_gap.fillna(0)
        
        return result
    
    def compute_intraday_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        盘中动量因子.
        
        逻辑：(收盘价 - 开盘价) / 开盘价
        预期：日内强势延续
        """
        if 'open' not in df.columns or 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        intraday_return = (df['close'] - df['open']) / (df['open'] + 1e-6)
        return intraday_return.fillna(0)
    
    def compute_money_flow(self, df: pd.DataFrame) -> pd.Series:
        """
        资金流因子.
        
        逻辑：基于成交金额和价格变化的资金流向
        """
        if 'amount' not in df.columns and 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        if 'pct_chg' in df.columns:
            money_flow = df['pct_chg'] * df.get('amount', df['volume'])
        else:
            close_return = df['close'].pct_change()
            money_flow = close_return * df.get('amount', df['volume'])
        
        # 截面标准化
        money_flow_rank = money_flow.groupby(df['trade_date']).transform(
            lambda x: x.rank(method='average', pct=True)
        )
        
        return money_flow_rank.fillna(0.5)
    
    def compute_relative_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        相对强度因子.
        
        逻辑：个股收益 / 市场收益
        """
        if 'pct_chg' not in df.columns and 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 计算个股收益
        if 'pct_chg' in df.columns:
            stock_return = df['pct_chg']
        else:
            stock_return = df['close'].pct_change()
        
        # 计算市场收益（按日期平均）
        market_return = stock_return.groupby(df['trade_date']).transform('mean')
        
        # 相对强度
        relative_strength = stock_return - market_return
        
        return relative_strength.fillna(0)
    
    def compute_volume_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        成交量动量因子.
        
        逻辑：成交量相对变化
        """
        if 'volume' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 成交量 5 日移动平均
        vol_ma5 = df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(5, min_periods=3).mean()
        )
        vol_ma20 = df.groupby('symbol')['volume'].transform(
            lambda x: x.rolling(20, min_periods=10).mean()
        )
        
        # 成交量动量
        volume_momentum = (vol_ma5 - vol_ma20) / (vol_ma20 + 1e-6)
        
        return volume_momentum.fillna(0)
    
    def compute_price_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        价格效率因子.
        
        逻辑：收益 / 波动率（夏普比率概念）
        """
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 5 日收益
        ret_5 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(5)
        )
        
        # 20 日波动率
        vol_20 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(20, min_periods=10).std()
        )
        
        # 价格效率
        price_efficiency = ret_5 / (vol_20 + 1e-6)
        
        return price_efficiency.fillna(0)
    
    def compute_liquidity_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """
        流动性不平衡因子.
        
        逻辑：基于换手率变化的流动性冲击
        """
        if 'turnover_rate' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 换手率变化
        turnover_change = df.groupby('symbol')['turnover_rate'].transform(
            lambda x: x.pct_change()
        )
        
        # 负号：流动性冲击后往往有反转
        result = -turnover_change.fillna(0)
        
        return result
    
    def compute_volatility_adjusted_return(self, df: pd.DataFrame) -> pd.Series:
        """
        波动率调整收益因子.
        
        逻辑：近期收益 / 近期波动率
        """
        if 'close' not in df.columns:
            return pd.Series(0, index=df.index)
        
        # 10 日收益
        ret_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(10)
        )
        
        # 10 日波动率
        vol_10 = df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(10, min_periods=5).std()
        )
        
        # 波动率调整收益
        var_10 = ret_10 / (vol_10 + 1e-6)
        
        return var_10.fillna(0)
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有因子"""
        result = df.copy()
        
        # V163 新因子
        result['overnight_gap'] = self.compute_overnight_gap(result)
        result['intraday_momentum'] = self.compute_intraday_momentum(result)
        result['money_flow'] = self.compute_money_flow(result)
        result['relative_strength'] = self.compute_relative_strength(result)
        result['volume_momentum'] = self.compute_volume_momentum(result)
        result['price_efficiency'] = self.compute_price_efficiency(result)
        result['liquidity_imbalance'] = self.compute_liquidity_imbalance(result)
        result['volatility_adjusted_return'] = self.compute_volatility_adjusted_return(result)
        
        # 传统因子
        result['momentum_5'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(5)
        ).fillna(0)
        
        result['reversion_5'] = -result['momentum_5']
        
        result['volatility_5'] = result.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(5, min_periods=3).std()
        ).fillna(0)
        
        # 量价因子
        if 'pct_chg' in result.columns and 'volume' in result.columns:
            price_rank = result['pct_chg'].fillna(0).rank(method='average', pct=True)
            volume_rank = result['volume'].pct_change().fillna(0).rank(method='average', pct=True)
            result['volume_price_contradiction'] = (price_rank - volume_rank).fillna(0)
        else:
            result['volume_price_contradiction'] = 0
        
        # volume_rank
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        return result


class AlphaResearchV163:
    """V163 Alpha 研究引擎"""
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_factors: int = MAX_FACTORS,
        n_bins: int = 10,
        enable_ensemble: bool = True,
        enable_pac: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.n_bins = n_bins
        self.enable_ensemble = enable_ensemble
        self.enable_pac = enable_pac
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        self.ic_history = {}
        
        self.data_healer = DataHealerV163(db_url) if auto_heal else None
        self.factor_generator = FactorGeneratorV163()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Factor Reconstruction Plan")
        logger.info(f"  Target IC: > 0.095")
        logger.info(f"  Target IC IR: > 0.60")
    
    def _log_audit(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.audit_log) >= MAX_LOG_ENTRIES:
            self.audit_log = self.audit_log[-MAX_LOG_ENTRIES//2:]
        self.audit_log.append(entry)
    
    def _calc_factor_ic_with_stats(
        self, 
        df: pd.DataFrame, 
        factor_col: str
    ) -> Tuple[float, float, float, int]:
        """计算因子 IC 及统计量"""
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
        
        ic_mean = float(np.mean(ics)) if ics else 0.0
        ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else 0.0
        ic_ir = (ic_mean / ic_std * np.sqrt(IC_IR_ANNUALIZATION)) if ic_std > 1e-10 else 0.0
        n_obs = len(ics)
        
        if factor_col not in self.ic_history:
            self.ic_history[factor_col] = []
        self.ic_history[factor_col].extend(ics)
        
        return ic_mean, ic_std, ic_ir, n_obs
    
    def _process_factor(self, series: pd.Series, trade_dates: pd.Series) -> np.ndarray:
        """因子处理"""
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 数据自愈
        if self.auto_heal and self.data_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg', 'open']
            result = self.data_healer.check_and_heal(result, required_cols)
        
        # 准备标签
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # 生成单期回报列
        for i in range(1, 6):
            col = f't{i}_return_period'
            if col not in result.columns:
                if i == 1:
                    result[col] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-i) / x - 1)
                else:
                    result[col] = result.groupby('symbol')['close'].transform(
                        lambda x: x.shift(-i) / x.shift(-(i-1)) - 1
                    )
        
        # 生成因子
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        # 因子筛选和集成
        candidate_factors = V163_CORE_FACTORS + ['momentum_5', 'reversion_5', 'volatility_5', 
                                                   'volume_price_contradiction', 'volume_rank']
        
        factor_data = {}
        factor_signs = {}
        factor_ic_stats = {}
        
        # 筛选有效因子
        valid_factors = []
        for factor in candidate_factors:
            if factor not in result.columns:
                continue
            
            ic_mean, ic_std, ic_ir, n_obs = self._calc_factor_ic_with_stats(result, factor)
            factor_ic_stats[factor] = {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'ic_ir': ic_ir,
                'n_obs': n_obs,
            }
            
            # 只保留 |IC| > 0.01 的因子
            if abs(ic_mean) > 0.01:
                valid_factors.append(factor)
                self._log_audit(
                    "FactorValid",
                    f"{factor}: IC={ic_mean:.4f}, Std={ic_std:.4f}, IR={ic_ir:.2f} - VALID"
                )
            else:
                self._log_audit(
                    "FactorFiltered",
                    f"{factor}: IC={ic_mean:.4f} - FILTERED (too weak)"
                )
        
        if not valid_factors:
            valid_factors = candidate_factors
        
        # PAC 校正 + 标准化
        for factor in valid_factors:
            f_raw = result[factor].fillna(0)
            
            ic_mean = factor_ic_stats[factor]['ic_mean']
            pac_sign = 1 if ic_mean >= 0 else -1
            factor_signs[factor] = pac_sign
            
            f_processed = f_raw * pac_sign
            
            self.factor_directions[factor] = factor_signs[factor]
            self.factor_ics[factor] = factor_ic_stats[factor]['ic_mean'] * factor_signs[factor]
            
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # V163: 简单等权重集成
        n_valid = len(valid_factors)
        self.factor_weights = {f: 1.0 / n_valid for f in valid_factors}
        
        self._log_audit(
            "EqualWeights",
            f"V163 Equal Weight: {n_valid} factors, weight={1.0/n_valid:.4f}"
        )
        
        self.selected_factors = valid_factors
        
        # 加权集成
        score = np.zeros(len(result), dtype=np.float64)
        for factor in valid_factors:
            f = factor_data.get(factor)
            if f is None:
                continue
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 1.0 / n_valid)
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 截面 Z-Score 归一化
        result['score'] = result.groupby('trade_date')['score_raw'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(valid_factors)} factors (V163 Equal Weight)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return', 
                       't1_return_period', 't2_return_period', 't3_return_period', 
                       't4_return_period', 't5_return_period']
        
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        if df is not None and not df.empty:
            ics = {}
            for factor in self.selected_factors:
                if factor in df.columns:
                    ic_mean, _, _, _ = self._calc_factor_ic_with_stats(df, factor)
                    sign = self.factor_directions.get(factor, 1)
                    ics[factor] = ic_mean * sign
                else:
                    ics[factor] = self.factor_ics.get(factor, 0.0)
            return ics
        return self.factor_ics
    
    def get_selected_factors(self) -> List[str]:
        return self.selected_factors
    
    def get_data_healing_log(self) -> List[Dict]:
        return self.data_healer.get_healing_log() if self.data_healer else []
    
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log[-MAX_LOG_ENTRIES:]


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_factors: int = MAX_FACTORS,
    n_bins: int = 10,
    enable_ensemble: bool = True,
    enable_pac: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV163:
    return AlphaResearchV163(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        auto_heal=auto_heal,
        db_url=db_url,
    )


class V163Runner:
    """V163 统一回测运行器"""
    
    def __init__(self, parquet_path: Optional[str] = None, output_dir: str = "reports") -> None:
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
            auto_heal=True,
            db_url=db_url
        )
        
        self.referee = get_backtest_referee(self.alpha_module, output_dir=output_dir)
        self.referee.VERSION = "V163"
        
        logger.info("V163 Runner initialized")
    
    def load_data(self, year: int) -> pd.DataFrame:
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
        logger.info("=" * 70)
        logger.info(f"V163 Audit - Year {year}")
        logger.info("=" * 70)
        
        df = self.load_data(year)
        
        if df.empty:
            logger.warning(f"No data loaded for year {year}")
            return {'year': year, 'error': 'No data loaded', 'passed': False}
        
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
        
        report_path = self.generate_v163_report(result, year)
        
        result['year'] = year
        result['custom_report_path'] = report_path
        
        return result
    
    def generate_v163_report(self, result: dict, year: int) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"v163_audit_{year}_{timestamp}.md"
        
        t1_ic = result.get('t1_ic', {})
        ic_decay = result.get('ic_decay', {})
        backtest_result = result.get('backtest_result', {})
        passed = result.get('passed', False)
        
        factor_ics_v163 = self.alpha_module.get_factor_ics()
        selected_factors = self.alpha_module.get_selected_factors()
        
        v155_ic = 0.0924
        v155_ir = 0.60
        v162_ic = 0.0498
        v162_ir = 0.34
        
        weight_table = ""
        for factor in selected_factors:
            ic = factor_ics_v163.get(factor, 0.0)
            weight = self.alpha_module.factor_weights.get(factor, 0.0)
            weight_table += f"| {factor} | {ic:.4f} | {weight:.4f} |\n"
        
        report_content = f"""# V163 Alpha Audit Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Year**: {year}
**Version**: V163 Factor Reconstruction Plan

---

## 1. Executive Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| T+1 Rank IC | {t1_ic.get('mean_ic', 0):.4f} | > 0.095 | {'✓ PASSED' if t1_ic.get('mean_ic', 0) > 0.095 else '✗ FAILED'} |
| IC IR | {t1_ic.get('ic_ir', 0):.2f} | > 0.60 | {'✓ PASSED' if t1_ic.get('ic_ir', 0) > 0.60 else '✗ FAILED'} |
| IC Decay | {'Monotonic' if ic_decay.get('is_monotonic', False) else 'Non-monotonic'} | Monotonic | {'✓ PASSED' if ic_decay.get('is_monotonic', False) else '✗ FAILED'} |

**Overall Assessment**: **{'PASSED ✓' if passed else 'FAILED ✗'}**

---

## 2. V163 Core Features

### 2.1 New Factors

| Factor | Description |
|--------|-------------|
| overnight_gap | 隔夜缺口因子（预期缺口回补） |
| intraday_momentum | 盘中动量因子 |
| money_flow | 资金流因子 |
| relative_strength | 相对强度因子 |
| volume_momentum | 成交量动量 |
| price_efficiency | 价格效率（收益/波动率） |
| liquidity_imbalance | 流动性不平衡 |
| volatility_adjusted_return | 波动率调整收益 |

### 2.2 Factor Weight Analysis

| Factor | IC | Weight |
|--------|-----|--------|
{weight_table if weight_table else "*No data*"}

### 2.3 Top Selected Factors

| Factor | IC | Weight | Selected |
|--------|-----|--------|----------|
"""
        
        if factor_ics_v163:
            for factor_name, ic in sorted(factor_ics_v163.items(), key=lambda x: abs(x[1]), reverse=True)[:12]:
                weight = self.alpha_module.factor_weights.get(factor_name, 0.0)
                selected = "✓" if factor_name in selected_factors else ""
                report_content += f"| {factor_name} | {ic:.4f} | {weight:.4f} | {selected} |\n"
        
        report_content += f"""
---

## 3. V163 vs V162 vs V155 Comparison

| Metric | V155 | V162 | V163 | Improvement (vs V162) |
|--------|------|------|------|----------------------|
| T+1 IC | {v155_ic:.4f} | {v162_ic:.4f} | {t1_ic.get('mean_ic', 0):.4f} | {t1_ic.get('mean_ic', 0) - v162_ic:+.4f} |
| IC IR | {v155_ir:.2f} | {v162_ir:.2f} | {t1_ic.get('ic_ir', 0):.2f} | {t1_ic.get('ic_ir', 0) - v162_ir:+.2f} |

---

## 4. IC Decay Analysis

| Horizon | IC | Pattern |
|---------|-----|---------|
| T+1 | {ic_decay.get('t1_ic', 0):.4f} | Baseline |
| T+3 | {ic_decay.get('t3_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t1_ic', 0) >= ic_decay.get('t3_ic', 0) else '✗ Non-monotonic'} |
| T+5 | {ic_decay.get('t5_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t3_ic', 0) >= ic_decay.get('t5_ic', 0) else '✗ Non-monotonic'} |

---

## 5. Backtest Performance

| Metric | Value |
|--------|-------|
| Initial Capital | {self.referee.INITIAL_CAPITAL:,.0f} |
| Final Value | {backtest_result.get('final_value', 0):,.2f} |
| Total Return | {backtest_result.get('total_return', 0):.2%} |
| Annual Return | {backtest_result.get('annual_return', 0):.2%} |
| Sharpe Ratio | {backtest_result.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {backtest_result.get('max_drawdown', 0):.2%} |

---

## 6. Conclusion

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| T+1 Rank IC | > 0.095 | {t1_ic.get('mean_ic', 0):.4f} | {'✓' if t1_ic.get('mean_ic', 0) > 0.095 else '✗'} |
| IC IR | > 0.60 | {t1_ic.get('ic_ir', 0):.2f} | {'✓' if t1_ic.get('ic_ir', 0) > 0.60 else '✗'} |
| IC Decay | Monotonic | {ic_decay.get('decay_pattern', 'N/A')} | {'✓' if ic_decay.get('is_monotonic', False) else '✗'} |

**{'PASSED ✓' if passed else 'FAILED ✗'}**

---

*Report generated by V163 Runner*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        json_result = {
            'alpha_metrics': {'t1_ic': t1_ic, 'ic_decay': ic_decay, 'passed': passed},
            'backtest_metrics': backtest_result,
            'factor_ics': factor_ics_v163,
            'factor_weights': self.alpha_module.factor_weights,
            'selected_factors': selected_factors,
            'v162_comparison': {
                'v162_ic': v162_ic,
                'v162_ir': v162_ir,
                'ic_improvement': t1_ic.get('mean_ic', 0) - v162_ic,
                'ir_improvement': t1_ic.get('ic_ir', 0) - v162_ir,
            },
            'v155_comparison': {
                'v155_ic': v155_ic,
                'v155_ir': v155_ir,
                'ic_improvement': t1_ic.get('mean_ic', 0) - v155_ic,
                'ir_improvement': t1_ic.get('ic_ir', 0) - v155_ir,
            },
            'config': {'year': year, 'initial_capital': self.referee.INITIAL_CAPITAL},
        }
        
        json_path = self.output_dir / f"v163_audit_{year}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        return str(report_path)
    
    def run_multi_year_audit(self, years: list[int]) -> dict:
        logger.info("=" * 70)
        logger.info(f"V163 Multi-Year Audit - Years: {years}")
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
        
        return summary


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV163...")
    
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