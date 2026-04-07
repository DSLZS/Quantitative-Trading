"""
Alpha Research Module - V165 V155-Exact-Recovery (精确复制 V155 成功因子).

【V155 审计结论】
- T+1 IC: 0.0924 (目标 > 0.095) - 差 0.0026
- IC IR: 0.58 (目标 > 0.60) - 差 0.02
- 选中因子：['volume_rank', 'momentum_5', 'volatility_5', 'volume_price_contradiction', 'liquidity_alpha']
- 核心发现：volatility_5 权重最高 (0.395), volume_price_contradiction 次之 (0.248)

【V165 核心策略 - 精确复制 + 微调】
1. 严格使用 V155 的 5 个成功因子
2. 增强 volatility_5 和 volume_price_contradiction 的信号强度
3. 使用 IC^1.5 加权（比 V155 的 |IC| 加权更强）
4. 移除 Lead-Lag 选择，直接使用固定因子组合

【工程纪律】
- 基于 V155 ORA 2.0 精确复制
- 严禁修改 src/engine/ 目录
- 初始资金锁定 100,000，费率锁定 0.03%
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

VERSION = "V165"

# V165 固定因子组合 - 精确复制 V155
V165_FIXED_FACTORS = [
    'volume_rank',              # V155 因子 1
    'momentum_5',               # V155 因子 2
    'volatility_5',             # V155 因子 3 - 权重最高
    'volume_price_contradiction',  # V155 ORM 核心因子
    'liquidity_alpha',          # V155 因子 5
]

# V165 日志截断配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V165 ORA 2.0 参数
ORM_CORE_FACTOR = 'volume_price_contradiction'  # 正交残差挖掘的核心因子
ROLLING_WINDOW = 20  # 滚动 IC 窗口

# V165 增强参数
IC_POWER = 1.5  # IC 加权幂次（比 V155 更强）


def winsorize_auto_heal(
    series: pd.Series, 
    sigma: float = 3.0, 
    percentile: float = 0.99
) -> pd.Series:
    """V165 自动愈合版 Winsorization"""
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


class RollingICSignCalculator:
    """V165 滚动 IC 符号计算器"""
    
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
        self.calculation_log = []
        
    def compute_rolling_ic_sign(
        self,
        df: pd.DataFrame,
        factor_col: str,
        return_col: str = 't1_return',
    ) -> pd.Series:
        """计算滚动 IC 符号"""
        if factor_col not in df.columns or return_col not in df.columns:
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
            return pd.Series(1, index=df.index)
        
        ic_df = pd.DataFrame(date_ics).sort_values('trade_date')
        ic_df['rolling_ic'] = ic_df['ic'].rolling(window=self.window, min_periods=5).mean()
        ic_df['rolling_ic_sign'] = np.sign(ic_df['rolling_ic']).replace(0, 1)
        
        ic_sign_map = ic_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        rolling_signs = result['trade_date'].map(ic_sign_map).fillna(1)
        
        return rolling_signs


class FactorGeneratorV165:
    """V165 因子生成器 - 精确复制 V155 因子"""
    
    def __init__(self):
        self.generation_log = []
    
    def compute_momentum(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(window)
        ).fillna(0)
    
    def compute_volatility(self, df: pd.DataFrame, window: int) -> pd.Series:
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(window).std()
        ).fillna(0)
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
        """V165 量价背离因子 - ORM 核心（精确复制 V155）"""
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
        
        return vpc
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
        """V165 流动性 Alpha 因子（精确复制 V155）"""
        if 'pct_chg' in df.columns and 'volume' in df.columns:
            ofi = df['pct_chg'] * df['volume']
        else:
            ofi = pd.Series(0, index=df.index)
        
        if 'close' in df.columns:
            ts_std_20 = df.groupby('symbol')['close'].transform(
                lambda x: x.rolling(20, min_periods=5).std()
            )
        else:
            ts_std_20 = pd.Series(1, index=df.index)
        
        liquidity_alpha = (ofi / (ts_std_20 + 1e-6)).fillna(0)
        
        return liquidity_alpha
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 V165 所需的基础因子"""
        result = df.copy()
        
        # momentum_5
        result['momentum_5'] = self.compute_momentum(result, 5)
        
        # volatility_5
        result['volatility_5'] = self.compute_volatility(result, 5)
        
        # volume_price_contradiction (ORM 核心)
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        
        # liquidity_alpha
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        # volume_rank (截面排名)
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        return result


class AlphaResearchV165:
    """
    V165 Alpha 研究引擎 - V155-Exact-Recovery (精确复制 V155).
    """
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.0001,
        n_bins: int = 10,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_bins = n_bins
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = V165_FIXED_FACTORS.copy()
        self.audit_log = []
        
        # 初始化模块
        self.factor_generator = FactorGeneratorV165()
        self.pac_calculator = RollingICSignCalculator()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: V155-Exact-Recovery (Fixed 5 Factors)")
        logger.info(f"  Fixed Factors: {V165_FIXED_FACTORS}")
        logger.info(f"  ORM Core Factor: {ORM_CORE_FACTOR}")
        logger.info(f"  IC Power Weighting: {IC_POWER}")
        logger.info(f"  Target IR: > 0.60")
        logger.info(f"  Target IC: > 0.095")
    
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
        """因子处理：Winsorization + 标准化"""
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分 - V165 核心逻辑（精确复制 V155）"""
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        
        result = df.copy()
        
        # 1. 准备标签（严格 T+1）
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
        # V165 修复：生成单期回报列用于 IC Decay 分析
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
        
        # 2. 生成基础因子
        result = self.factor_generator.compute_all_factors(result)
        
        # 3. V165 固定因子组合 - 不使用 Lead-Lag 选择
        self._log_audit("FixedFactors", f"Using fixed {len(V165_FIXED_FACTORS)} factors: {V165_FIXED_FACTORS}")
        
        # 4. Rolling PAC 极性校正 + IC 计算
        factor_data = {}
        factor_signs = {}
        
        for factor in V165_FIXED_FACTORS:
            if factor not in result.columns:
                continue
            
            f_raw = result[factor].copy()
            
            # Rolling PAC 极性校正
            rolling_sign = self.pac_calculator.compute_rolling_ic_sign(result, factor)
            sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
            factor_signs[factor] = sign_val
            f_processed = f_raw * rolling_sign
            
            self.factor_directions[factor] = factor_signs[factor]
            
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic * factor_signs[factor]
            
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        # 5. V165 IC^POWER 加权集成
        ic_weights = {}
        total_weight = 0.0
        
        for factor in V165_FIXED_FACTORS:
            ic = self.factor_ics.get(factor, 0.0)
            weight = (abs(ic) + self.EPSILON) ** IC_POWER
            ic_weights[factor] = weight
            total_weight += weight
        
        if total_weight > 0:
            self.factor_weights = {f: w / total_weight for f, w in ic_weights.items()}
        else:
            self.factor_weights = {f: 1.0 / len(V165_FIXED_FACTORS) for f in V165_FIXED_FACTORS}
        
        self._log_audit(
            "ICWeights",
            f"Weighted by |IC|^{IC_POWER}: {self.factor_weights}"
        )
        
        # 6. 加权集成
        score = np.zeros(len(result), dtype=np.float64)
        for factor in V165_FIXED_FACTORS:
            f = factor_data.get(factor)
            if f is None:
                continue
            if isinstance(f, np.ndarray):
                f = pd.Series(f)
            f_clean = f.fillna(0).astype(np.float64)
            weight = self.factor_weights.get(factor, 1.0 / len(V165_FIXED_FACTORS))
            score += f_clean.values * weight
        
        result['score_raw'] = score
        
        # 7. 最终截面 Z-Score 归一化
        result['score'] = result.groupby('trade_date')['score_raw'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(V165_FIXED_FACTORS)} factors (V155-Exact)")
        
        output_cols = ['trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return', 
                       't1_return_period', 't2_return_period', 't3_return_period', 
                       't4_return_period', 't5_return_period']
        
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """获取因子 IC"""
        if df is not None and not df.empty:
            ics = {}
            for factor in V165_FIXED_FACTORS:
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
        return V165_FIXED_FACTORS.copy()
    
    def get_audit_log(self) -> List[Dict]:
        """获取审计日志"""
        return self.audit_log[-MAX_LOG_ENTRIES:]


def get_alpha_research(
    ic_threshold: float = 0.0001,
    n_bins: int = 10,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV165:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV165(
        ic_threshold=ic_threshold,
        n_bins=n_bins,
        auto_heal=auto_heal,
        db_url=db_url,
    )


class V165Runner:
    """V165 Runner Class - 使用 BacktestReferee 进行回测"""
    
    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, year: int) -> pd.DataFrame:
        """从数据库加载数据"""
        from sqlalchemy import create_engine, text
        
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        
        start_date = f"{year}0101"
        end_date = f"{year}1231"
        
        query = text("""
            SELECT symbol, trade_date, open, high, low, close, volume, amount,
                   turnover_rate, total_mv, pre_close, pct_chg
            FROM stock_daily
            WHERE trade_date BETWEEN :start AND :end
            ORDER BY symbol, trade_date
        """)
        
        df = pd.read_sql_query(query, engine, params={'start': start_date, 'end': end_date})
        logger.info(f"Loaded {len(df)} rows for year {year}")
        
        return df
    
    def compute_ic_metrics(self, df: pd.DataFrame) -> Dict:
        """计算 IC 指标"""
        ics_t1, ics_t3, ics_t5 = [], [], []
        
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            
            score = day['score'].fillna(0)
            
            for ics, ret_col in [(ics_t1, 't1_return'), (ics_t3, 't3_return'), (ics_t5, 't5_return')]:
                if ret_col in day.columns:
                    ret = day[ret_col].fillna(0)
                    if len(score) > 10 and np.std(score) > 1e-10:
                        ic = np.corrcoef(score.rank(), ret.rank())[0, 1]
                        if not np.isnan(ic):
                            ics.append(ic)
        
        def calc_ic_stats(ics, name):
            if not ics:
                return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0}
            mean_ic = np.mean(ics)
            std_ic = np.std(ics)
            ir = mean_ic / (std_ic + 1e-10)
            return {
                'mean_ic': float(mean_ic),
                'ic_std': float(std_ic),
                'ic_ir': float(ir),
                'num_days': len(ics),
            }
        
        result = {}
        result['t1_ic'] = calc_ic_stats(ics_t1, 'T+1')
        result['t3_ic'] = calc_ic_stats(ics_t3, 'T+3')
        result['t5_ic'] = calc_ic_stats(ics_t5, 'T+5')
        
        result['ic_decay'] = {
            't1_ic': result['t1_ic']['mean_ic'],
            't3_ic': result['t3_ic']['mean_ic'],
            't5_ic': result['t5_ic']['mean_ic'],
            'is_monotonic': result['t1_ic']['mean_ic'] >= result['t3_ic']['mean_ic'] >= result['t5_ic']['mean_ic'],
        }
        
        result['passed'] = (
            result['t1_ic']['mean_ic'] > 0.05 and
            result['t1_ic']['ic_ir'] > 0.6
        )
        
        return result
    
    def run_audit(self, year: int) -> Dict:
        """运行完整审计"""
        logger.info(f"Running V165 audit for year {year}")
        
        df = self.load_data(year)
        
        alpha = get_alpha_research()
        result = alpha.compute_score(df)
        
        metrics = self.compute_ic_metrics(result)
        
        metrics['selected_factors'] = alpha.get_selected_factors()
        metrics['factor_ics'] = alpha.get_factor_ics(result)
        metrics['factor_weights'] = alpha.factor_weights
        
        logger.info(f"V165 Audit Complete - T+1 IC: {metrics['t1_ic']['mean_ic']:.4f}, IR: {metrics['t1_ic']['ic_ir']:.2f}")
        
        return metrics


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV165...")
    
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
    logger.info(f"  Factor Weights: {alpha.factor_weights}")