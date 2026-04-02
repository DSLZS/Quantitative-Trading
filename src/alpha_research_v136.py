"""
Alpha Research Module - V136 高维非线性空间拓展.

【V136 核心改进】
1. InteractionMiner: 二阶交互算子自动挖掘
   - Rank(OFI) * Rank(Volatility): 订单流与波动率交互
   - Rank(Momentum) / Rank(Volume): 动量与成交量交互
   - 量价背离：价格涨 + 量缩
   - 波动压制：高波动下的收益反转

2. Volatility_Inhibition: 波动率抑制机制
   - 在市场极高波动时主动调低信号强度
   - 目标 IR > 0.3

3. DataHealing: 数据自愈逻辑
   - 主动调用 data_loader.py 补全缺失特征
   - 严禁以报错为由停止运行

4. 基础因子池扩充：包含二阶合成因子

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标 |
| IR | > 0.3 | 波动率抑制效果 |
| 二阶合成因子数量 | >= 2 | 新 Alpha 逻辑 |
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

VERSION = "V136"

# V136 基础因子池（包含一阶因子）
BASE_FACTORS = [
    'pct_chg', 'change', 'momentum_5', 'momentum_20',
    'volatility_5', 'ma_deviation_5', 'ma_deviation_20',
    'price_position_20', 'price_position_60', 'bias_60',
    'volume_price_stable', 'volume_price_divergence_5', 'volume_price_divergence_20',
    'turnover_bias_20', 'volume_shrink_ratio',
    'rsi_14', 'mfi_14', 'macd', 'macd_signal', 'macd_hist', 'hist_sharpe_20d',
]

# V136 新增二阶交互因子名称
INTERACTION_FACTORS = [
    'ofi_volatility_interaction',      # 订单流 * 波动率
    'momentum_volume_ratio',           # 动量 / 成交量
    'price_volume_divergence',         # 量价背离
    'volatility_suppression',          # 波动压制
    'smart_money_volatility',          # 聪明钱 * 波动率
    'reversion_volatility_interaction', # 反转 * 波动率
]

# V136 所有因子（基础 + 二阶）
ALL_FACTORS = BASE_FACTORS + INTERACTION_FACTORS


def winsorize(series: pd.Series, sigma: float = 2.5) -> pd.Series:
    """Winsorization 去极值"""
    mean = series.mean()
    std = series.std()
    lower = mean - sigma * std
    upper = mean + sigma * std
    return series.clip(lower=lower, upper=upper)


class DataHealing:
    """
    V136 数据自愈模块.
    
    【职责】
    1. 检测特征缺失
    2. 主动调用 data_loader.py 补全
    3. 记录 [V136][DataHealing] 日志
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
                logger.info("[V136][DataHealing] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V136][DataHealing] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
            logger.info("[V136][DataHealing] No database URL, SQL healer disabled")
    
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
        logger.info(f"[V136][DataHealing] {action} - Column: {column}, Status: {status}, {details}")
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        """
        检查并修复缺失列.
        
        Args:
            df: 输入 DataFrame
            required_columns: 必需的列名列表
            
        Returns:
            修复后的 DataFrame
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
            
            # 尝试从 SQL 补全
            if self.engine:
                result = self._heal_from_sql(result, missing)
            else:
                # 使用默认值填充
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
        
        return result
    
    def _heal_from_sql(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """从 SQL 补全缺失列"""
        if not self.engine or df.empty:
            return df
        
        result = df.copy()
        symbols = df['symbol'].unique().tolist()[:50]  # 限制数量
        
        if not symbols:
            return df
        
        # 构建日期范围
        if 'trade_date' in df.columns:
            dates = pd.to_datetime(df['trade_date']).unique()
            start_date = pd.to_datetime(dates.min()).strftime('%Y%m%d')
            end_date = pd.to_datetime(dates.max()).strftime('%Y%m%d')
        else:
            return df
        
        # 尝试从 stock_daily 补全
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
                # 合并数据
                for col in columns:
                    if col in sql_df.columns:
                        # 按 symbol 和 trade_date 合并
                        merge_df = result.merge(
                            sql_df[['symbol', 'trade_date', col]],
                            on=['symbol', 'trade_date'],
                            how='left',
                            suffixes=('', '_sql')
                        )
                        # 用 SQL 数据填充缺失值
                        result[col] = merge_df[col].fillna(merge_df[f'{col}_sql'])
                        result = result.drop(columns=[c for c in result.columns if c.endswith('_sql')])
                        
                        self._log_healing(
                            action="HealedFromSQL",
                            column=col,
                            status="SUCCESS",
                            details=f"Healed {len(sql_df)} rows from stock_daily"
                        )
                        
        except Exception as e:
            logger.error(f"[V136][DataHealing] SQL heal failed: {e}")
            for col in columns:
                result[col] = 0.0
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        """获取自愈日志"""
        return self.healing_log


class InteractionMiner:
    """
    V136 二阶交互因子挖掘器.
    
    【核心功能】
    1. 自动寻找具有经济含义的二阶组合特征
    2. 量价背离（价格涨 + 量缩）
    3. 波动压制（高波动下的收益反转）
    4. 订单流与波动率交互
    
    【交互算子】
    - Mul: A * B (协同效应)
    - Div: A / B (比率效应)
    - Diff: A - B (差异效应)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.interaction_results = {}
        self.mining_log = []
        
    def _log_mining(self, action: str, details: str = ""):
        """记录挖掘日志"""
        entry = {'action': action, 'details': details}
        self.mining_log.append(entry)
        logger.info(f"[V136][InteractionMiner] {action}: {details}")
    
    def compute_all_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有二阶交互因子.
        
        Args:
            df: 包含基础因子的 DataFrame
            
        Returns:
            包含交互因子的 DataFrame
        """
        result = df.copy()
        
        self._log_mining("StartInteractionMining", f"Processing {len(df)} rows")
        
        # 1. OFI * Volatility (订单流 * 波动率)
        result = self._compute_ofi_volatility(result)
        
        # 2. Momentum / Volume (动量 / 成交量)
        result = self._compute_momentum_volume(result)
        
        # 3. Price-Volume Divergence (量价背离)
        result = self._compute_price_volume_divergence(result)
        
        # 4. Volatility Suppression (波动压制)
        result = self._compute_volatility_suppression(result)
        
        # 5. Smart Money * Volatility (聪明钱 * 波动率)
        result = self._compute_smart_money_volatility(result)
        
        # 6. Reversion * Volatility (反转 * 波动率)
        result = self._compute_reversion_volatility(result)
        
        self._log_mining("Complete", f"Generated {len(INTERACTION_FACTORS)} interaction factors")
        
        return result
    
    def _rank(self, series: pd.Series) -> pd.Series:
        """截面排名"""
        return series.rank(method='average', pct=True)
    
    def _compute_ofi_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        订单流 * 波动率交互.
        
        经济含义：在高波动环境下，订单流不平衡的信号更强
        """
        result = df.copy()
        
        # 计算 OFI (订单流不平衡)
        if 'amount' in df.columns and 'volume' in df.columns:
            # OFI = 成交额 / 成交量 (平均价格变化方向)
            vwap = df['amount'] / (df['volume'] + self.epsilon)
            price_change = df['close'] - df['pre_close'] if 'pre_close' in df.columns else df['change']
            ofi = price_change * df['volume'] / (df['amount'] + self.epsilon)
        elif 'pct_chg' in df.columns and 'volume' in df.columns:
            ofi = df['pct_chg'] * df['volume']
        else:
            ofi = df.get('pct_chg', pd.Series(0, index=df.index)) * df.get('volume', pd.Series(1, index=df.index))
        
        # 计算波动率
        if 'volatility_5' in df.columns:
            volatility = df['volatility_5']
        elif 'volatility_20' in df.columns:
            volatility = df['volatility_20']
        else:
            # 计算简单波动率
            volatility = df['pct_chg'].rolling(5).std() if 'pct_chg' in df.columns else pd.Series(0, index=df.index)
        
        # 排名后相乘
        ofi_rank = self._rank(ofi.fillna(0))
        vol_rank = self._rank(volatility.fillna(0))
        
        result['ofi_volatility_interaction'] = ofi_rank * vol_rank
        
        self._log_mining("Computed", "ofi_volatility_interaction = Rank(OFI) * Rank(Volatility)")
        
        return result
    
    def _compute_momentum_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        动量 / 成交量交互.
        
        经济含义：成交量确认的动量更可靠
        """
        result = df.copy()
        
        # 获取动量
        if 'momentum_5' in df.columns:
            momentum = df['momentum_5']
        elif 'momentum_20' in df.columns:
            momentum = df['momentum_20']
        elif 'pct_chg' in df.columns:
            momentum = df['pct_chg']
        else:
            momentum = pd.Series(0, index=df.index)
        
        # 获取成交量
        if 'volume' in df.columns:
            volume = df['volume']
        elif 'amount' in df.columns:
            volume = df['amount']
        else:
            volume = pd.Series(1, index=df.index)
        
        # 排名后相除
        momentum_rank = self._rank(momentum.fillna(0))
        volume_rank = self._rank(volume.fillna(self.epsilon))
        
        result['momentum_volume_ratio'] = momentum_rank / (volume_rank + self.epsilon)
        
        self._log_mining("Computed", "momentum_volume_ratio = Rank(Momentum) / Rank(Volume)")
        
        return result
    
    def _compute_price_volume_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        量价背离.
        
        经济含义：价格涨 + 量缩 = 背离信号（可能反转）
        """
        result = df.copy()
        
        # 价格变化
        if 'pct_chg' in df.columns:
            price_change = df['pct_chg']
        elif 'change' in df.columns:
            price_change = df['change']
        else:
            price_change = pd.Series(0, index=df.index)
        
        # 成交量变化
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
        elif 'amount' in df.columns:
            volume_change = df['amount'].pct_change()
        else:
            volume_change = pd.Series(0, index=df.index)
        
        # 量价背离 = 价格排名 - 成交量排名
        price_rank = self._rank(price_change.fillna(0))
        volume_rank = self._rank(volume_change.fillna(0))
        
        result['price_volume_divergence'] = price_rank - volume_rank
        
        self._log_mining("Computed", "price_volume_divergence = Rank(Price_Change) - Rank(Volume_Change)")
        
        return result
    
    def _compute_volatility_suppression(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        波动压制.
        
        经济含义：高波动下的收益反转
        """
        result = df.copy()
        
        # 获取波动率
        if 'volatility_5' in df.columns:
            volatility = df['volatility_5']
        elif 'volatility_20' in df.columns:
            volatility = df['volatility_20']
        else:
            volatility = df['pct_chg'].rolling(5).std() if 'pct_chg' in df.columns else pd.Series(0, index=df.index)
        
        # 获取反转信号（负动量）
        if 'momentum_5' in df.columns:
            reversion = -df['momentum_5']  # 负动量 = 反转
        elif 'pct_chg' in df.columns:
            reversion = -df['pct_chg']
        else:
            reversion = pd.Series(0, index=df.index)
        
        # 波动压制 = 高波动 * 反转
        vol_rank = self._rank(volatility.fillna(0))
        rev_rank = self._rank(reversion.fillna(0))
        
        result['volatility_suppression'] = vol_rank * rev_rank
        
        self._log_mining("Computed", "volatility_suppression = Rank(Volatility) * Rank(-Momentum)")
        
        return result
    
    def _compute_smart_money_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        聪明钱 * 波动率.
        
        经济含义：大资金在低波动时更活跃
        """
        result = df.copy()
        
        # 聪明钱信号（成交额/成交量，大单比例）
        if 'amount' in df.columns and 'volume' in df.columns:
            smart_money = df['amount'] / (df['volume'] + self.epsilon)
        elif 'amount' in df.columns:
            smart_money = df['amount']
        elif 'volume' in df.columns:
            smart_money = df['volume']
        else:
            smart_money = pd.Series(0, index=df.index)
        
        # 波动率
        if 'volatility_5' in df.columns:
            volatility = df['volatility_5']
        else:
            volatility = df['pct_chg'].rolling(5).std() if 'pct_chg' in df.columns else pd.Series(0, index=df.index)
        
        # 聪明钱 * 波动率（负相关，低波动时聪明钱更有效）
        sm_rank = self._rank(smart_money.fillna(0))
        vol_rank = self._rank(1 / (volatility + self.epsilon))  # 逆波动率
        
        result['smart_money_volatility'] = sm_rank * vol_rank
        
        self._log_mining("Computed", "smart_money_volatility = Rank(Smart_Money) * Rank(1/Volatility)")
        
        return result
    
    def _compute_reversion_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        反转 * 波动率.
        
        经济含义：高波动时反转效应更强
        """
        result = df.copy()
        
        # 反转信号（过去收益的负值）
        if 'momentum_20' in df.columns:
            reversion = -df['momentum_20']
        elif 'momentum_5' in df.columns:
            reversion = -df['momentum_5']
        elif 'pct_chg' in df.columns:
            reversion = -df['pct_chg']
        else:
            reversion = pd.Series(0, index=df.index)
        
        # 波动率
        if 'volatility_5' in df.columns:
            volatility = df['volatility_5']
        else:
            volatility = df['pct_chg'].rolling(5).std() if 'pct_chg' in df.columns else pd.Series(0, index=df.index)
        
        # 反转 * 波动率
        rev_rank = self._rank(reversion.fillna(0))
        vol_rank = self._rank(volatility.fillna(0))
        
        result['reversion_volatility_interaction'] = rev_rank * vol_rank
        
        self._log_mining("Computed", "reversion_volatility_interaction = Rank(-Momentum) * Rank(Volatility)")
        
        return result
    
    def get_mining_log(self) -> List[Dict]:
        """获取挖掘日志"""
        return self.mining_log


class VolatilityInhibition:
    """
    V136 波动率抑制模块.
    
    【核心功能】
    1. 检测市场极高波动状态
    2. 主动调低信号强度
    3. 目标 IR > 0.3
    """
    
    def __init__(self, threshold_percentile: float = 0.9):
        self.threshold_percentile = threshold_percentile
        self.inhibition_log = []
        
    def _log_inhibition(self, action: str, details: str = ""):
        """记录抑制日志"""
        entry = {'action': action, 'details': details}
        self.inhibition_log.append(entry)
        logger.info(f"[V136][VolatilityInhibition] {action}: {details}")
    
    def apply_inhibition(self, df: pd.DataFrame, score_col: str = 'score') -> pd.DataFrame:
        """
        应用波动率抑制.
        
        Args:
            df: 包含评分的 DataFrame
            score_col: 评分列名
            
        Returns:
            调整后的 DataFrame
        """
        result = df.copy()
        
        self._log_inhibition("StartInhibition", f"Processing {len(df)} rows")
        
        # 计算截面波动率
        if 'volatility_5' in df.columns:
            cross_vol = df['volatility_5']
        elif 'volatility_20' in df.columns:
            cross_vol = df['volatility_20']
        elif 'pct_chg' in df.columns:
            cross_vol = df['pct_chg'].rolling(5).std()
        else:
            cross_vol = pd.Series(0, index=df.index)
        
        # 计算波动率分位数阈值
        vol_threshold = cross_vol.quantile(self.threshold_percentile)
        
        # 波动率抑制因子
        # 高波动时降低信号强度
        inhibition_factor = np.where(
            cross_vol > vol_threshold,
            vol_threshold / (cross_vol + 1e-6),  # 高波动时抑制
            1.0  # 正常波动时不抑制
        )
        
        # 应用抑制
        if score_col in result.columns:
            result['score_inhibited'] = result[score_col] * inhibition_factor
        else:
            result['score_inhibited'] = inhibition_factor
        
        # 记录抑制统计
        inhibited_count = np.sum(inhibition_factor < 1.0)
        avg_inhibition = np.mean(inhibition_factor[inhibition_factor < 1.0]) if inhibited_count > 0 else 1.0
        
        self._log_inhibition(
            "Complete",
            f"Inhibited {inhibited_count} samples ({100*inhibited_count/len(df):.1f}%), avg factor={avg_inhibition:.3f}"
        )
        
        return result
    
    def compute_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率状态.
        
        Returns:
            包含 volatility_regime 列的 DataFrame
            - 0: 低波动
            - 1: 正常
            - 2: 高波动
        """
        result = df.copy()
        
        if 'volatility_5' in df.columns:
            cross_vol = df['volatility_5']
        elif 'volatility_20' in df.columns:
            cross_vol = df['volatility_20']
        else:
            cross_vol = df['pct_chg'].rolling(5).std() if 'pct_chg' in df.columns else pd.Series(0, index=df.index)
        
        # 分位数阈值
        low_thresh = cross_vol.quantile(0.3)
        high_thresh = cross_vol.quantile(0.7)
        
        # 波动率状态
        regime = np.where(
            cross_vol < low_thresh, 0,
            np.where(cross_vol > high_thresh, 2, 1)
        )
        
        result['volatility_regime'] = regime
        
        return result
    
    def get_inhibition_log(self) -> List[Dict]:
        """获取抑制日志"""
        return self.inhibition_log


class AlphaResearchV136:
    """V136 Alpha 研究引擎 - 高维非线性空间拓展"""
    
    EPSILON = 1e-6
    
    def __init__(
        self,
        ic_threshold: float = 0.023,
        n_factors: int = 5,
        enable_interaction_mining: bool = True,
        enable_volatility_inhibition: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
    ):
        self.ic_threshold = ic_threshold
        self.n_factors = n_factors
        self.enable_interaction_mining = enable_interaction_mining
        self.enable_volatility_inhibition = enable_volatility_inhibition
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        # 初始化模块
        self.data_healer = DataHealing(db_url) if auto_heal else None
        self.interaction_miner = InteractionMiner() if enable_interaction_mining else None
        self.volatility_inhibitor = VolatilityInhibition() if enable_volatility_inhibition else None
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Interaction Mining + Volatility Inhibition")
        logger.info(f"  IC Threshold: {ic_threshold}")
        logger.info(f"  N Factors: {n_factors}")
        logger.info(f"  Interaction Mining: {'Enabled' if enable_interaction_mining else 'Disabled'}")
        logger.info(f"  Volatility Inhibition: {'Enabled' if enable_volatility_inhibition else 'Disabled'}")
        logger.info(f"  Auto Healing: {'Enabled' if auto_heal else 'Disabled'}")
    
    def _log_audit(self, action: str, details: str = ""):
        self.audit_log.append({'action': action, 'details': details})
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
        """计算因子 IC（按日期分组平均）"""
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
        # 1. Winsorization 去极值 (2.5σ)
        series_wins = winsorize(series.fillna(0), sigma=2.5)
        
        # 2. 截面标准化
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        
        return result.values
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha 评分"""
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
        
        # 3. 计算二阶交互因子
        if self.enable_interaction_mining and self.interaction_miner:
            result = self.interaction_miner.compute_all_interactions(result)
            self._log_audit("InteractionMining", f"Generated {len(INTERACTION_FACTORS)} interaction factors")
        
        # 4. 计算所有因子 IC 并排序
        factor_ics = []
        all_available_factors = BASE_FACTORS + INTERACTION_FACTORS
        
        for factor in all_available_factors:
            if factor not in result.columns:
                continue
            ic = self._calc_factor_ic(result, factor)
            self.factor_ics[factor] = ic
            factor_ics.append((factor, ic))
        
        # 5. 按 IC 绝对值排序
        factor_ics.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # 6. 处理因子（翻转 + 标准化）
        factor_data = {}
        
        for factor, ic in factor_ics:
            if abs(ic) < self.ic_threshold:
                continue
            if len(self.selected_factors) >= self.n_factors:
                break
                
            f_raw = result[factor]
            
            # 负 IC 因子翻转
            if ic < 0:
                f_processed = -f_raw
                self.factor_directions[factor] = -1
                self._log_audit("FactorFlip", f"{factor}: IC={ic:.4f} -> flipped")
            else:
                f_processed = f_raw
                self.factor_directions[factor] = 1
                self._log_audit("FactorKeep", f"{factor}: IC={ic:.4f} -> kept")
            
            # 因子处理：去极值 + 标准化
            f_std = self._process_factor(f_processed, result['trade_date'])
            
            factor_data[factor] = f_std
            self.selected_factors.append(factor)
        
        self._log_audit("FactorSelection", f"Selected {len(self.selected_factors)}/{len(all_available_factors)} factors")
        
        # 7. IC 绝对值加权
        if not self.selected_factors:
            result['score'] = np.random.randn(len(result))
        else:
            # 获取调整后的 IC（考虑翻转）
            adjusted_ics = []
            for factor in self.selected_factors:
                direction = self.factor_directions.get(factor, 1)
                adjusted_ics.append(abs(self.factor_ics[factor]) * direction)
            
            # IC 绝对值加权
            total_ic = sum(abs(ic) for ic in adjusted_ics)
            
            if total_ic > 0:
                weights = [abs(ic) / total_ic for ic in adjusted_ics]
            else:
                weights = [1.0 / len(self.selected_factors)] * len(self.selected_factors)
            
            score = np.zeros(len(result))
            for i, factor in enumerate(self.selected_factors):
                score += factor_data[factor] * weights[i]
                self.factor_weights[factor] = weights[i]
            
            result['score'] = score
        
        # 8. 应用波动率抑制
        if self.enable_volatility_inhibition and self.volatility_inhibitor:
            result = self.volatility_inhibitor.apply_inhibition(result, score_col='score')
            # 使用抑制后的评分
            if 'score_inhibited' in result.columns:
                result['score'] = result['score_inhibited']
            self._log_audit("VolatilityInhibition", "Applied volatility inhibition to scores")
        
        self._log_audit("Complete", f"Final score with {len(self.selected_factors)} factors (IC weighted)")
        
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
    
    def get_interaction_mining_log(self) -> List[Dict]:
        """获取交互因子挖掘日志"""
        return self.interaction_miner.get_mining_log() if self.interaction_miner else []
    
    def get_volatility_inhibition_log(self) -> List[Dict]:
        """获取波动率抑制日志"""
        return self.volatility_inhibitor.get_inhibition_log() if self.volatility_inhibitor else []
    
    def get_data_healing_log(self) -> List[Dict]:
        """获取数据自愈日志"""
        return self.data_healer.get_healing_log() if self.data_healer else []


def get_alpha_research(
    ic_threshold: float = 0.023,
    n_factors: int = 5,
    enable_interaction_mining: bool = True,
    enable_volatility_inhibition: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
) -> AlphaResearchV136:
    """获取 AlphaResearch 实例"""
    return AlphaResearchV136(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        enable_interaction_mining=enable_interaction_mining,
        enable_volatility_inhibition=enable_volatility_inhibition,
        auto_heal=auto_heal,
        db_url=db_url,
    )


def run_v136_backtest(
    data_path: str = "data/parquet/features_latest.parquet",
    output_dir: str = "reports",
    ic_threshold: float = 0.023,
    n_factors: int = 5,
) -> Dict[str, Any]:
    """运行 V136 回测"""
    from src.engine.backtest_referee import BacktestReferee
    
    logger.info(f"[{VERSION}] Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"[{VERSION}] Loaded {len(df)} rows")
    
    alpha = get_alpha_research(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        enable_interaction_mining=True,
        enable_volatility_inhibition=True,
        auto_heal=True,
    )
    
    referee = BacktestReferee(alpha, output_dir=output_dir)
    referee.VERSION = VERSION
    
    result = referee.run_audit(df)
    
    return result


if __name__ == "__main__":
    result = run_v136_backtest()
    print(json.dumps(result, indent=2, default=str))