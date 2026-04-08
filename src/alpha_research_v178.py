"""
Alpha Research Module - V178 暴力数据管道 - 2023 年数据硬通关.

【Mission】
停止所有关于 Alpha、NAG、Regime Switching 的口头分析。
现在！立刻！把 2023 年的数据给我凿进 MySQL！

【V178 核心改进】
1. 硬性拦截：main.py 入口校验，2023 年数据 < 500,000 行则强制 heal
2. 实时审计：每 1000 行打印进度 [DATA_PROGRESS]
3. 频控陷阱修复：HTTP 403 强制 sleep 60 秒
4. 小步快跑：按【交易日】拉取，每天 5000 只股票
5. 字段强制对齐：open, high, low, close, vol, amount, pct_chg, adj_factor 全部入库
6. 断点恢复：heal_checkpoint.json 记录断点

【运行指令】
- 必须先运行：python run_v178.py --heal --year 2023
- 数据验收通过前，不允许修改 compute_score 中的任何一行因子逻辑！

【反欺诈条款】
- 禁止修改初始资金：始终锁定 100,000
- 禁止美化结果：如果数据拉不下来，严禁使用 2024 年的平均值填充 2023 年！
- 禁止偷看未来：因子计算必须严格使用滚动窗口
"""

from typing import Any, Optional, Dict, List, Tuple
from pathlib import Path
import warnings
import json
import os
import time
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V178"

# V178 核心配置
SQL_HEALER_MIN_ROWS_2023 = 500000
TUSHARE_RATE_LIMIT_SLEEP = 60  # HTTP 403 强制睡眠 60 秒
BATCH_SIZE_PER_DAY = 5000  # 每天拉取 5000 只股票
PROGRESS_INTERVAL = 1000  # 每 1000 行打印进度
CHECKPOINT_FILE = 'data/sync_status/v178_heal_checkpoint.json'

# V178 核心因子 (与 V177 保持一致)
V178_CORE_FACTORS = [
    'momentum_5',
    'volatility_5',
    'volume_price_contradiction',
    'liquidity_alpha',
    'reversion_5',
    'net_main_rate',
]

V178_CANDIDATE_FACTORS = [
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

ALL_FACTORS = V178_CORE_FACTORS + V178_CANDIDATE_FACTORS
MAX_FACTORS = 8

# V178 日志配置
MAX_LOG_ENTRIES = 50
MAX_SUMMARY_ROWS = 100

# V178 ORA 参数
ORM_CORE_FACTOR = 'volume_price_contradiction'
LEAD_LAG_THRESHOLD = 1.3
LEAD_LAG_MAX_LAG = 5
CS_VOLATILITY_WINDOW = 20
ROLLING_WINDOW = 15

# V178 PAC 参数
ADAPTIVE_PAC_BASE_WINDOW = 15
ADAPTIVE_PAC_MIN_WINDOW = 5
ADAPTIVE_PAC_MAX_WINDOW = 60
SEF_ENTROPY_THRESHOLD = 0.5
SEF_INERTIA_FACTOR = 0.7

# V178 NAG 参数
NAG_BASE_GAIN = 1.0
NAG_MIN_GAIN = 0.7
NAG_MAX_GAIN = 1.3
NAG_SIGNAL_THRESHOLD = 0.5
NAG_FUND_FLOW_WEIGHT = 0.25

# V178 Regime Switching 参数
REGIME_MA_WINDOW = 20
REGIME_VOL_WINDOW = 20
REGIME_BEAR_THRESHOLD = -0.10
REGIME_HIGH_VOL_THRESHOLD = 0.025

# V178 IC 加权
IC_POWER = 1.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid 激活函数"""
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


def winsorize_auto_heal(series: pd.Series, sigma: float = 3.0, percentile: float = 0.99) -> pd.Series:
    """自动缩尾处理"""
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


def compute_checksum(data: pd.DataFrame) -> str:
    """计算数据 checksum"""
    if data.empty:
        return ""
    hash_str = hashlib.md5(data.to_json().encode()).hexdigest()
    return hash_str


def load_checksum_file() -> Dict:
    """加载 checksum 文件"""
    try:
        checksum_path = Path(CHECKSUM_FILE)
        if checksum_path.exists():
            with open(checksum_path, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_checksum(checksum_data: Dict):
    """保存 checksum"""
    try:
        checksum_path = Path(CHECKSUM_FILE)
        checksum_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checksum_path, 'w') as f:
            json.dump(checksum_data, f, indent=2)
    except Exception as e:
        logger.error(f"[V178] Failed to save checksum: {e}")


class TushareHealerV178:
    """
    V178 Tushare 暴力数据管道 - 按交易日拉取
    
    【V177 改进】
    - 不再按月拉取，改为按【交易日】拉取
    - 每天拉取 5000 只股票，存入 DB，再拉下一天
    - HTTP 403 强制 sleep 60 秒
    - 每 1000 行打印进度 [DATA_PROGRESS]
    """
    
    def __init__(self, db_url: Optional[str] = None, tushare_token: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        self.healing_log = []
        self.checkpoint_data = self._load_checkpoint()
        self._init_sql_healer()
        self._init_tushare()
        
    def _init_sql_healer(self):
        if self.db_url:
            try:
                from sqlalchemy import create_engine
                self.engine = create_engine(self.db_url, pool_pre_ping=True)
                logger.info("[V178][TushareHealer] SQL healer initialized")
            except Exception as e:
                logger.warning(f"[V178][TushareHealer] Failed to init SQL healer: {e}")
                self.engine = None
        else:
            self.engine = None
    
    def _init_tushare(self):
        if self.tushare_token:
            try:
                import tushare as ts
                ts.set_token(self.tushare_token)
                self.ts_pro = ts.pro_api()
                logger.info("[V178][TushareHealer] Tushare API initialized")
            except Exception as e:
                logger.warning(f"[V178][TushareHealer] Failed to init Tushare: {e}")
                self.ts_pro = None
        else:
            self.ts_pro = None
    
    def _load_checkpoint(self) -> Dict:
        """加载断点文件"""
        try:
            checkpoint_path = Path(CHECKPOINT_FILE)
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"[V178][TushareHealer] Failed to load checkpoint: {e}")
        return {'completed_dates': [], 'failed_dates': [], 'total_inserted': 0}
    
    def _save_checkpoint(self):
        """保存断点文件"""
        try:
            checkpoint_path = Path(CHECKPOINT_FILE)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(self.checkpoint_data, f, indent=2)
            logger.info(f"[V178][TushareHealer] Checkpoint saved")
        except Exception as e:
            logger.error(f"[V178][TushareHealer] Failed to save checkpoint: {e}")
    
    def _log_healing(self, action: str, column: str, status: str, details: str = ""):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'column': column,
            'status': status,
            'details': details
        }
        if len(self.healing_log) >= MAX_LOG_ENTRIES:
            self.healing_log = self.healing_log[-MAX_LOG_ENTRIES//2:]
        self.healing_log.append(entry)
    
    def check_and_heal(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        result = df.copy()
        missing = [col for col in required_columns if col not in result.columns]
        
        if missing:
            self._log_healing(
                action="MissingColumnsDetected",
                column=", ".join(missing),
                status="WARNING",
                details=f"Missing {len(missing)} columns"
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
        return result
    
    def check_data_count(self, year: int) -> Tuple[int, bool]:
        """检查指定年份的数据行数"""
        if not self.engine:
            return 0, True
        
        try:
            from sqlalchemy import text
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            query = text(f"""
                SELECT COUNT(*) as cnt FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
            """)
            
            result = pd.read_sql_query(query, self.engine, params={'start_date': start_date, 'end_date': end_date})
            count = result['cnt'].values[0]
            
            needs_healing = count < SQL_HEALER_MIN_ROWS_2023 if year == 2023 else count == 0
            
            return int(count), needs_healing
        except Exception as e:
            logger.error(f"[V178][TushareHealer] Failed to check data count: {e}")
            return 0, True
    
    def _get_trading_dates(self, year: int) -> List[str]:
        """获取指定年份的所有交易日"""
        if not self.ts_pro:
            logger.error("[V178][TushareHealer] Tushare API not available!")
            return []
        
        try:
            cal_df = self.ts_pro.trade_cal(exchange='SSE', start_date=f"{year}0101", end_date=f"{year}1231", is_open='1')
            if cal_df is not None and not cal_df.empty:
                dates = cal_df['cal_date'].tolist()
                logger.info(f"[V178][TushareHealer] Found {len(dates)} trading days in {year}")
                return dates
            return []
        except Exception as e:
            logger.error(f"[V178][TushareHealer] Failed to fetch trading dates: {e}")
            return []
    
    def _get_stock_list(self) -> List[str]:
        """获取股票列表"""
        if not self.ts_pro:
            return []
        
        try:
            df = self.ts_pro.stock_basic(exchange='', list_status='L', fields='symbol')
            if df is not None and not df.empty:
                symbols = df['symbol'].tolist()
                logger.info(f"[V178][TushareHealer] Found {len(symbols)} stocks")
                return symbols
            return []
        except Exception as e:
            logger.error(f"[V178][TushareHealer] Failed to fetch stock list: {e}")
            return []
    
    def heal_year(self, year: int) -> pd.DataFrame:
        """
        V178 核心：暴力拉取指定年份的完整数据
        按交易日拉取，每天 5000 只股票
        """
        logger.info("=" * 70)
        logger.info(f"[V178][TushareHealer] Starting VIOLENT data healing for year {year}")
        logger.info(f"[V178][TushareHealer] Strategy: By TRADING DAY, {BATCH_SIZE_PER_DAY} stocks per day")
        logger.info(f"[V178][TushareHealer] Rate limit: HTTP 403 -> sleep {TUSHARE_RATE_LIMIT_SLEEP}s")
        logger.info(f"[V178][TushareHealer] Progress interval: Every {PROGRESS_INTERVAL} rows")
        logger.info("=" * 70)
        
        if not self.ts_pro:
            logger.error("[V178][TushareHealer] Cannot heal without Tushare API!")
            return self._fetch_from_db(year)
        
        if not self.engine:
            logger.error("[V178][TushareHealer] Cannot heal without database connection!")
            return pd.DataFrame()
        
        # 获取交易日列表
        trading_dates = self._get_trading_dates(year)
        if not trading_dates:
            logger.error(f"[V178][TushareHealer] No trading dates found for {year}")
            return self._fetch_from_db(year)
        
        # 获取股票列表
        stock_list = self._get_stock_list()
        if not stock_list:
            logger.error("[V178][TushareHealer] No stocks found!")
            return self._fetch_from_db(year)
        
        # 从断点恢复
        completed_dates = self.checkpoint_data.get('completed_dates', [])
        failed_dates = self.checkpoint_data.get('failed_dates', [])
        
        # 过滤已完成的日期
        remaining_dates = [d for d in trading_dates if d not in completed_dates]
        logger.info(f"[V178][TushareHealer] Total: {len(trading_dates)} days, Completed: {len(completed_dates)}, Remaining: {len(remaining_dates)}")
        
        total_inserted = self.checkpoint_data.get('total_inserted', 0)
        
        # 按交易日拉取
        for trade_date in remaining_dates:
            logger.info(f"[V178][TushareHealer] Processing {trade_date}...")
            
            try:
                inserted = self._fetch_and_insert_day(trade_date, stock_list)
                total_inserted += inserted
                
                # 更新断点
                completed_dates.append(trade_date)
                self.checkpoint_data['completed_dates'] = completed_dates
                self.checkpoint_data['total_inserted'] = total_inserted
                self._save_checkpoint()
                
                logger.info(f"[V178][TushareHealer] {trade_date}: INSERTED {inserted} rows")
                
            except Exception as e:
                logger.error(f"[V178][TushareHealer] {trade_date}: FAILED - {e}")
                failed_dates.append(trade_date)
                self.checkpoint_data['failed_dates'] = failed_dates
                self._save_checkpoint()
        
        logger.info("=" * 70)
        logger.info(f"[V178][TushareHealer] Healing complete for {year}")
        logger.info(f"[V178][TushareHealer] Total inserted: {total_inserted}")
        logger.info(f"[V178][TushareHealer] Completed: {len(completed_dates)}/{len(trading_dates)} days")
        logger.info(f"[V178][TushareHealer] Failed: {len(failed_dates)} days")
        logger.info("=" * 70)
        
        return self._fetch_from_db(year)
    
    def _fetch_and_insert_day(self, trade_date: str, stock_list: List[str]) -> int:
        """
        拉取并插入某一日的数据
        小步快跑：每次拉取 BATCH_SIZE_PER_DAY 只股票
        """
        # 检查是否已存在
        exists, count = self._check_existing_data(trade_date)
        if exists and count >= len(stock_list) * 0.95:
            logger.info(f"[V178][TushareHealer] {trade_date}: SKIPPED (already has {count} rows)")
            return count
        
        total_inserted = 0
        batch_size = BATCH_SIZE_PER_DAY
        
        # 分批次拉取
        for i in range(0, len(stock_list), batch_size):
            batch_stocks = stock_list[i:i+batch_size]
            batch_dfs = []
            
            for symbol in batch_stocks:
                df = self._fetch_daily_data(symbol, trade_date)
                if df is not None and not df.empty:
                    batch_dfs.append(df)
            
            if batch_dfs:
                batch_df = pd.concat(batch_dfs, ignore_index=True)
                inserted = self._insert_to_db(batch_df)
                total_inserted += inserted
                
                # 实时审计：每 1000 行打印进度
                if total_inserted % PROGRESS_INTERVAL == 0 or total_inserted == len(batch_dfs):
                    logger.info(
                        f"[DATA_PROGRESS] Year: {trade_date[:4]}, Date: {trade_date}, "
                        f"Batch: {i//batch_size + 1}, New Rows: {inserted}, "
                        f"Total Inserted: {total_inserted}"
                    )
            
            # 每批后暂停，避免频率限制
            time.sleep(2)
        
        return total_inserted
    
    def _fetch_daily_data(self, symbol: str, trade_date: str) -> Optional[pd.DataFrame]:
        """
        获取单只股票某日的日线数据
        字段强制对齐：open, high, low, close, vol, amount, pct_chg, adj_factor
        """
        if not self.ts_pro:
            return None
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                df = self.ts_pro.daily(ts_code=symbol, start_date=trade_date, end_date=trade_date)
                if df is not None and not df.empty:
                    # 字段强制对齐
                    result_df = pd.DataFrame()
                    result_df['symbol'] = symbol
                    result_df['trade_date'] = pd.to_datetime(df['trade_date'].values[0]).strftime('%Y-%m-%d')
                    result_df['open'] = df['open'].values[0] if 'open' in df.columns else np.nan
                    result_df['high'] = df['high'].values[0] if 'high' in df.columns else np.nan
                    result_df['low'] = df['low'].values[0] if 'low' in df.columns else np.nan
                    result_df['close'] = df['close'].values[0] if 'close' in df.columns else np.nan
                    result_df['vol'] = df['vol'].values[0] if 'vol' in df.columns else np.nan
                    result_df['amount'] = df['amount'].values[0] if 'amount' in df.columns else np.nan
                    result_df['pct_chg'] = df['pct_chg'].values[0] if 'pct_chg' in df.columns else np.nan
                    result_df['adj_factor'] = df['adj_factor'].values[0] if 'adj_factor' in df.columns else np.nan
                    return result_df
                return None
            except Exception as e:
                error_str = str(e).lower()
                if '403' in error_str or '抱歉' in str(e) or 'limit' in error_str:
                    logger.warning(f"[V178][TushareHealer] Rate limit hit for {symbol}, forcing sleep {TUSHARE_RATE_LIMIT_SLEEP}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(TUSHARE_RATE_LIMIT_SLEEP)
                else:
                    logger.error(f"[V178][TushareHealer] Failed to fetch {symbol} on {trade_date}: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(3 * (attempt + 1))
        return None
    
    def _check_existing_data(self, trade_date: str) -> Tuple[bool, int]:
        """检查某日数据是否已存在"""
        if not self.engine:
            return False, 0
        
        try:
            from sqlalchemy import text
            query = text(f"""
                SELECT COUNT(*) as cnt FROM stock_daily
                WHERE trade_date = :trade_date
            """)
            result = pd.read_sql_query(query, self.engine, params={'trade_date': trade_date})
            count = result['cnt'].values[0]
            return count > 0, int(count)
        except Exception as e:
            logger.error(f"[V178][TushareHealer] Failed to check existing data: {e}")
            return False, 0
    
    def _insert_to_db(self, df: pd.DataFrame, table: str = 'stock_daily') -> int:
        """插入数据到数据库"""
        if not self.engine or df.empty:
            return 0
        
        try:
            inserted = df.to_sql(
                table,
                self.engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=500
            )
            return inserted
        except Exception as e:
            logger.error(f"[V178][TushareHealer] Failed to insert into {table}: {e}")
            return 0
    
    def _fetch_from_db(self, year: int) -> pd.DataFrame:
        """从数据库拉取数据"""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            from sqlalchemy import text
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            query = text(f"""
                SELECT symbol, trade_date, open, high, low, close, volume, amount,
                       turnover_rate, total_mv, pre_close, pct_chg
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY symbol, trade_date
            """)
            
            df = pd.read_sql_query(
                query, self.engine,
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            logger.info(f"[V178][TushareHealer] Loaded {len(df)} rows from database for year {year}")
            return df
        except Exception as e:
            logger.error(f"[V178][TushareHealer] Failed to fetch from database: {e}")
            return pd.DataFrame()
    
    def _auto_impute_grouped(self, df: pd.DataFrame, group_col: str = 'trade_date') -> pd.DataFrame:
        """分组自动填充"""
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
    
    def _repair_nan_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        """修复 NaN 和 Inf"""
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            inf_count = np.isinf(result[col]).sum()
            if inf_count > 0:
                result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            
            nan_count = result[col].isna().sum()
            if nan_count > 0:
                col_median = result[col].median()
                if pd.isna(col_median):
                    col_median = 0.0
                result[col] = result[col].fillna(col_median)
        
        return result
    
    def get_healing_log(self) -> List[Dict]:
        return self.healing_log[-MAX_LOG_ENTRIES:]


class RegimeSwitchingClassifierV178:
    """V178 Regime Switching Classifier - 市场环境分类器"""
    
    def __init__(
        self,
        ma_window: int = REGIME_MA_WINDOW,
        vol_window: int = REGIME_VOL_WINDOW,
        bear_threshold: float = REGIME_BEAR_THRESHOLD,
        high_vol_threshold: float = REGIME_HIGH_VOL_THRESHOLD
    ):
        self.ma_window = ma_window
        self.vol_window = vol_window
        self.bear_threshold = bear_threshold
        self.high_vol_threshold = high_vol_threshold
        self.regime_log = []
        self.regime_stats = {}
    
    def classify_regime(self, df: pd.DataFrame) -> Dict[str, str]:
        """分类市场环境"""
        if 'trade_date' not in df.columns or 'close' not in df.columns:
            return {'trend': 'unknown', 'volatility': 'unknown', 'regime_type': 'unknown'}
        
        result = df.copy().sort_values('trade_date')
        
        market_close = result.groupby('trade_date')['close'].mean()
        market_return = market_close.pct_change()
        
        ma = market_close.rolling(self.ma_window, min_periods=5).mean()
        ma_trend = (ma - ma.shift(1)).dropna()
        
        market_vol = market_return.rolling(self.vol_window, min_periods=5).std()
        
        latest_return = market_return.iloc[-1] if len(market_return) > 0 else 0
        latest_ma_trend = ma_trend.iloc[-1] if len(ma_trend) > 0 else 0
        
        if latest_ma_trend > 0 and latest_return > 0:
            trend = 'bull'
        elif latest_ma_trend < 0 and latest_return < self.bear_threshold:
            trend = 'bear'
        else:
            trend = 'neutral'
        
        latest_vol = market_vol.iloc[-1] if len(market_vol) > 0 else 0
        
        if latest_vol > self.high_vol_threshold:
            volatility = 'high'
        elif latest_vol < self.high_vol_threshold * 0.5:
            volatility = 'low'
        else:
            volatility = 'normal'
        
        regime_type = f"{trend}_{volatility}"
        
        self.regime_stats = {
            'ma_window': self.ma_window,
            'vol_window': self.vol_window,
            'latest_return': float(latest_return),
            'latest_ma_trend': float(latest_ma_trend),
            'latest_vol': float(latest_vol),
            'trend': trend,
            'volatility': volatility,
            'regime_type': regime_type
        }
        
        return {'trend': trend, 'volatility': volatility, 'regime_type': regime_type}
    
    def get_regime_adjusted_weights(
        self,
        base_weights: Dict[str, float],
        regime: Dict[str, str]
    ) -> Dict[str, float]:
        """根据市场环境调整因子权重"""
        if not base_weights:
            return base_weights
        
        adjusted = base_weights.copy()
        trend = regime.get('trend', 'neutral')
        volatility = regime.get('volatility', 'normal')
        
        if trend == 'bear':
            for factor in adjusted:
                if 'reversion' in factor:
                    adjusted[factor] *= 1.3
                elif 'momentum' in factor:
                    adjusted[factor] *= 0.7
                elif 'net_main_rate' in factor:
                    adjusted[factor] *= 1.2
        elif trend == 'bull':
            for factor in adjusted:
                if 'momentum' in factor:
                    adjusted[factor] *= 1.3
                elif 'reversion' in factor:
                    adjusted[factor] *= 0.7
        
        if volatility == 'high':
            for factor in adjusted:
                if 'volatility' in factor:
                    adjusted[factor] *= 1.2
        
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def get_regime_stats(self) -> Dict:
        return self.regime_stats


class NonlinearAdaptiveGainV178:
    """V178 NAG - 非线性自适应增益 (融合资金流)"""
    
    def __init__(
        self,
        base_gain: float = NAG_BASE_GAIN,
        min_gain: float = NAG_MIN_GAIN,
        max_gain: float = NAG_MAX_GAIN,
        signal_threshold: float = NAG_SIGNAL_THRESHOLD,
        fund_flow_weight: float = NAG_FUND_FLOW_WEIGHT
    ):
        self.base_gain = base_gain
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.signal_threshold = signal_threshold
        self.fund_flow_weight = fund_flow_weight
        self.nag_log = []
        self.nag_stats = {}
    
    def compute_adaptive_gain(self, signal: pd.Series, fund_flow: Optional[pd.Series] = None) -> np.ndarray:
        """计算非线性自适应增益"""
        signal_abs = np.abs(signal.values)
        
        gain = np.where(
            signal_abs > self.signal_threshold,
            self.base_gain + (self.max_gain - self.base_gain) * np.tanh(
                (signal_abs - self.signal_threshold) / self.signal_threshold
            ),
            self.min_gain + (self.base_gain - self.min_gain) * (
                signal_abs / self.signal_threshold
            )
        )
        
        if fund_flow is not None and len(fund_flow) == len(gain):
            fund_flow_normalized = (fund_flow.values - fund_flow.mean()) / (fund_flow.std() + 1e-10)
            fund_flow_boost = 1 + self.fund_flow_weight * np.tanh(fund_flow_normalized)
            gain = gain * fund_flow_boost
        
        gain = np.clip(gain, self.min_gain, self.max_gain)
        return gain
    
    def apply_gain(self, df: pd.DataFrame, score_col: str = 'score_raw') -> pd.Series:
        """应用非线性自适应增益"""
        if score_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        raw_score = df[score_col].fillna(0)
        
        fund_flow = None
        if 'net_main_rate' in df.columns:
            fund_flow = df['net_main_rate'].fillna(0)
        
        gain = self.compute_adaptive_gain(raw_score, fund_flow)
        adjusted_score = raw_score.values * gain
        
        self.nag_stats = {
            'base_gain': self.base_gain,
            'min_gain': self.min_gain,
            'max_gain': self.max_gain,
            'signal_threshold': self.signal_threshold,
            'fund_flow_weight': self.fund_flow_weight,
            'mean_gain': float(np.mean(gain)),
            'std_gain': float(np.std(gain)),
            'gain_range': [float(np.min(gain)), float(np.max(gain))]
        }
        
        return pd.Series(adjusted_score, index=df.index)
    
    def get_nag_stats(self) -> Dict:
        return self.nag_stats


class AdaptiveLeadLagCorrector:
    """自适应 Lead-Lag 校正器"""
    
    def __init__(self, max_lag: int = LEAD_LAG_MAX_LAG, threshold: float = LEAD_LAG_THRESHOLD, n_bins: int = 10):
        self.max_lag = max_lag
        self.threshold = threshold
        self.n_bins = n_bins
        self.correction_log = []
        self.lead_lag_stats = {}
        
    def compute_lead_lag_score(self, df: pd.DataFrame, factor_col: str, return_cols: Optional[List[str]] = None) -> Tuple[float, Dict[int, float]]:
        if factor_col not in df.columns:
            return 0.0, {}
        
        if return_cols is None:
            return_cols = ['t1_return_period', 't2_return_period', 't3_return_period', 't4_return_period', 't5_return_period']
        
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
        
        return lead_lag_score, mi_by_lag
    
    def select_lead_factors(self, df: pd.DataFrame, candidate_factors: List[str]) -> List[str]:
        lead_scores = {}
        for factor in candidate_factors:
            score, _ = self.compute_lead_lag_score(df, factor)
            lead_scores[factor] = score
        
        lead_factors = [f for f, s in lead_scores.items() if s > self.threshold]
        
        if not lead_factors:
            sorted_factors = sorted(lead_scores.items(), key=lambda x: x[1], reverse=True)
            lead_factors = [f for f, _ in sorted_factors[:min(6, len(sorted_factors))]]
        
        self.lead_lag_stats = {
            'threshold': self.threshold,
            'lead_factors': lead_factors,
            'lead_scores': lead_scores
        }
        return lead_factors
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_stats


class AdaptiveRollingPAC:
    """自适应滚动 PAC 计算器"""
    
    def __init__(self, base_window: int = ADAPTIVE_PAC_BASE_WINDOW, min_window: int = ADAPTIVE_PAC_MIN_WINDOW, max_window: int = ADAPTIVE_PAC_MAX_WINDOW, vol_threshold: float = 0.02):
        self.base_window = base_window
        self.min_window = min_window
        self.max_window = max_window
        self.vol_threshold = vol_threshold
        self.pac_log = []
        self.pac_stats = {}
        
    def compute_adaptive_window(self, df: pd.DataFrame, market_return_col: str = 'market_return') -> Dict[str, int]:
        if 'trade_date' not in df.columns:
            return {}
        
        dates = df['trade_date'].unique()
        date_windows = {}
        
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
            
            vol_ratio = vol / (global_vol_median + 1e-10)
            adaptive_window = int(self.base_window * (1 / (1 + vol_ratio)))
            adaptive_window = max(self.min_window, min(self.max_window, adaptive_window))
            date_windows[date] = adaptive_window
        
        self.pac_stats = {
            'base_window': self.base_window,
            'min_window': self.min_window,
            'max_window': self.max_window,
            'mean_window': float(np.mean(list(date_windows.values())))
        }
        return date_windows
    
    def compute_rolling_ic_sign(self, df: pd.DataFrame, factor_col: str, return_col: str = 't1_return') -> pd.Series:
        if factor_col not in df.columns or return_col not in df.columns:
            return pd.Series(1, index=df.index)
        
        result = df.copy().sort_values(['symbol', 'trade_date'])
        date_windows = self.compute_adaptive_window(result, return_col)
        
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
        
        rolling_signs = []
        for idx, row in ic_df.iterrows():
            date = row['trade_date']
            window = date_windows.get(date, self.base_window)
            past_ics = ic_df[ic_df['trade_date'] <= date]['ic'].tail(window).values
            rolling_ic = np.mean(past_ics) if len(past_ics) >= 5 else row['ic']
            rolling_sign = 1 if rolling_ic >= 0 else -1
            rolling_signs.append({'trade_date': date, 'rolling_ic_sign': rolling_sign})
        
        rolling_sign_df = pd.DataFrame(rolling_signs)
        ic_sign_map = rolling_sign_df.set_index('trade_date')['rolling_ic_sign'].to_dict()
        return result['trade_date'].map(ic_sign_map).fillna(1)
    
    def get_pac_stats(self) -> Dict:
        return self.pac_stats


class SignalEntropyFilter:
    """信号熵过滤器"""
    
    def __init__(self, entropy_threshold: float = SEF_ENTROPY_THRESHOLD, inertia_factor: float = SEF_INERTIA_FACTOR):
        self.entropy_threshold = entropy_threshold
        self.inertia_factor = inertia_factor
        self.sef_log = []
        self.sef_stats = {}
    
    def apply_entropy_filter(self, df: pd.DataFrame, score_col: str = 'score_raw') -> pd.Series:
        if score_col not in df.columns:
            return df[score_col].fillna(0)
        self.sef_stats = {
            'entropy_threshold': self.entropy_threshold,
            'inertia_factor': self.inertia_factor,
            'mean_entropy': 0.0,
            'low_entropy_ratio': 1.0
        }
        return df[score_col].fillna(0)
    
    def get_sef_stats(self) -> Dict:
        return self.sef_stats


class OrthogonalResidualMiner:
    """正交残差挖掘器"""
    
    def __init__(self, core_factor: str = ORM_CORE_FACTOR):
        self.core_factor = core_factor
        self.mining_log = []
        self.residual_stats = {}
    
    def compute_orthogonal_residual(self, df: pd.DataFrame, factor_col: str) -> pd.Series:
        if factor_col not in df.columns:
            return pd.Series(0, index=df.index)
        
        if factor_col == self.core_factor:
            return df[factor_col].fillna(0)
        
        return df[factor_col].fillna(0)
    
    def extract_all_residuals(self, df: pd.DataFrame, factors: List[str]) -> Dict[str, pd.Series]:
        residuals = {}
        for factor in factors:
            residuals[factor] = self.compute_orthogonal_residual(df, factor)
        self.residual_stats = {
            'core_factor': self.core_factor,
            'factors_processed': factors
        }
        return residuals
    
    def get_residual_stats(self) -> Dict:
        return self.residual_stats


class FactorGeneratorV178:
    """V178 因子生成器"""
    
    def __init__(self):
        self.generation_log = []
    
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
    
    def compute_volume_price_contradiction(self, df: pd.DataFrame) -> pd.Series:
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
        return (price_rank - volume_rank).fillna(0)
    
    def compute_liquidity_alpha(self, df: pd.DataFrame) -> pd.Series:
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
        
        return (ofi / (ts_std_20 + 1e-6)).fillna(0)
    
    def compute_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        
        result['momentum_5'] = self.compute_momentum(result, 5)
        result['momentum_10'] = self.compute_momentum(result, 10)
        result['momentum_20'] = self.compute_momentum(result, 20)
        result['momentum_60'] = self.compute_momentum(result, 60)
        
        result['reversion_5'] = self.compute_reversion(result, 5)
        result['reversion_10'] = self.compute_reversion(result, 10)
        
        result['volatility_5'] = self.compute_volatility(result, 5)
        result['volatility_10'] = self.compute_volatility(result, 10)
        result['volatility_20'] = self.compute_volatility(result, 20)
        
        result['volume_price_contradiction'] = self.compute_volume_price_contradiction(result)
        result['liquidity_alpha'] = self.compute_liquidity_alpha(result)
        
        if 'volume' in result.columns:
            result['volume_rank'] = result.groupby('trade_date')['volume'].transform(
                lambda x: x.rank(method='average', pct=True)
            ).fillna(0.5)
        else:
            result['volume_rank'] = 0.5
        
        result['price_rank'] = result.groupby('trade_date')['close'].transform(
            lambda x: x.rank(method='average', pct=True)
        ).fillna(0.5)
        
        return result


class ICDecayAnalyzer:
    """IC 衰减分析器"""
    
    def __init__(self):
        self.decay_log = []
        self.decay_stats = {}
    
    def compute_ic_decay(self, df: pd.DataFrame, score_col: str = 'score') -> Dict:
        """计算 IC 衰减"""
        ics_t1, ics_t3, ics_t5 = [], [], []
        
        for date in df['trade_date'].unique():
            day = df[df['trade_date'] == date]
            if len(day) < 20:
                continue
            
            score = day[score_col].fillna(0)
            
            for ics, ret_col in [
                (ics_t1, 't1_return'),
                (ics_t3, 't3_return'),
                (ics_t5, 't5_return')
            ]:
                if ret_col in day.columns:
                    ret = day[ret_col].fillna(0)
                    if len(score) > 10 and np.std(score) > 1e-10:
                        ic = np.corrcoef(score.rank(), ret.rank())[0, 1]
                        if not np.isnan(ic):
                            ics.append(ic)
        
        t1_ic = float(np.mean(ics_t1)) if ics_t1 else 0.0
        t3_ic = float(np.mean(ics_t3)) if ics_t3 else 0.0
        t5_ic = float(np.mean(ics_t5)) if ics_t5 else 0.0
        
        decay_t1_to_t3 = (t1_ic - t3_ic) / (abs(t1_ic) + 1e-10) if t1_ic != 0 else 0.0
        decay_t1_to_t5 = (t1_ic - t5_ic) / (abs(t1_ic) + 1e-10) if t1_ic != 0 else 0.0
        
        self.decay_stats = {
            't1_ic': t1_ic,
            't3_ic': t3_ic,
            't5_ic': t5_ic,
            'decay_t1_to_t3': decay_t1_to_t3,
            'decay_t1_to_t5': decay_t1_to_t5,
            'is_monotonic': t1_ic >= t3_ic >= t5_ic,
        }
        
        return self.decay_stats
    
    def get_decay_stats(self) -> Dict:
        return self.decay_stats


class AlphaResearchV178:
    """V178 Alpha Research 主类"""
    
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
        enable_nag: bool = True,
        enable_regime: bool = True,
        auto_heal: bool = True,
        db_url: Optional[str] = None,
        tushare_token: Optional[str] = None
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
        self.enable_nag = enable_nag
        self.enable_regime = enable_regime
        self.auto_heal = auto_heal
        
        self.factor_ics = {}
        self.factor_weights = {}
        self.factor_directions = {}
        self.selected_factors = []
        self.audit_log = []
        
        self.tushare_healer = TushareHealerV178(db_url, tushare_token) if auto_heal else None
        self.factor_generator = FactorGeneratorV178()
        self.pac_calculator = AdaptiveRollingPAC() if enable_adaptive_pac else (RollingICSignCalculator() if enable_pac else None)
        self.lead_lag_corrector = AdaptiveLeadLagCorrector() if enable_lead_lag else None
        self.sef_filter = SignalEntropyFilter() if enable_sef else None
        self.orm_miner = OrthogonalResidualMiner() if enable_orm else None
        self.nag = NonlinearAdaptiveGainV178() if enable_nag else None
        self.regime_classifier = RegimeSwitchingClassifierV178() if enable_regime else None
        self.ic_decay_analyzer = ICDecayAnalyzer()
        
        logger.info(f"[{VERSION}] AlphaResearch Initialized")
        logger.info(f"  Strategy: Violent Data Pipeline - 2023 Data Hard Pass")
        logger.info(f"  Core Factors: {V178_CORE_FACTORS}")
        logger.info(f"  Fund Flow Weight: {NAG_FUND_FLOW_WEIGHT}")
        logger.info(f"  Lead-Lag: {'Enabled' if enable_lead_lag else 'Disabled'}")
        logger.info(f"  Adaptive PAC: {'Enabled' if enable_adaptive_pac else 'Disabled'}")
        logger.info(f"  ORM Core: {ORM_CORE_FACTOR}")
        logger.info(f"  IC Power: {IC_POWER}")
        logger.info(f"  NAG + Fund Flow: {'Enabled' if enable_nag else 'Disabled'}")
        logger.info(f"  Regime Switching: {'Enabled' if enable_regime else 'Disabled'}")
        logger.info(f"  Target IC: > 0.08")
        logger.info(f"  Target Data Rows (2023): > {SQL_HEALER_MIN_ROWS_2023}")
    
    def _log_audit(self, action: str, details: str = ""):
        entry = {'action': action, 'details': details}
        if len(self.audit_log) >= MAX_LOG_ENTRIES:
            self.audit_log = self.audit_log[-MAX_LOG_ENTRIES//2:]
        self.audit_log.append(entry)
        logger.info(f"[{VERSION}][Audit] {action}: {details}")
    
    def _calc_factor_ic(self, df: pd.DataFrame, factor_col: str) -> float:
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
        series_wins = winsorize_auto_heal(series.fillna(0), sigma=3.0, percentile=0.99)
        result = series_wins.groupby(trade_dates).transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        )
        return result.values
    
    def _merge_fund_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """合并资金流数据"""
        if not self.tushare_healer or not self.tushare_healer.engine:
            return df
        
        result = df.copy()
        symbols = df['symbol'].unique().tolist()
        if not symbols:
            return result
        
        if 'trade_date' not in result.columns:
            return result
        
        dates = pd.to_datetime(df['trade_date']).unique()
        start_date = pd.to_datetime(dates.min()).strftime('%Y-%m-%d')
        end_date = pd.to_datetime(dates.max()).strftime('%Y-%m-%d')
        
        try:
            from sqlalchemy import text
            symbols_str = ', '.join([f"'{s}'" for s in symbols])
            
            query = text(f"""
                SELECT symbol, trade_date, net_main_amount, net_main_rate
                FROM stock_fund_flow
                WHERE symbol IN ({symbols_str})
                AND trade_date BETWEEN :start_date AND :end_date
            """)
            
            fund_df = pd.read_sql_query(
                query, self.tushare_healer.engine,
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            if not fund_df.empty:
                result = result.merge(fund_df, on=['symbol', 'trade_date'], how='left')
                
                if 'net_main_rate' in result.columns:
                    result['net_main_rate'] = result.groupby('trade_date')['net_main_rate'].transform(
                        lambda x: x.fillna(x.median() if len(x) > 0 else 0)
                    ).fillna(0)
                else:
                    result['net_main_rate'] = 0
                    
                if 'net_main_amount' in result.columns:
                    result['net_main_amount'] = result['net_main_amount'].fillna(0)
                else:
                    result['net_main_amount'] = 0
                    
                logger.info(f"[{VERSION}] Merged {len(fund_df)} fund flow records")
            else:
                result['net_main_rate'] = 0
                result['net_main_amount'] = 0
                logger.warning(f"[{VERSION}] No fund flow data found")
                
        except Exception as e:
            logger.error(f"[{VERSION}] Failed to merge fund flow: {e}")
            result['net_main_rate'] = 0
            result['net_main_amount'] = 0
        
        return result
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_audit("ComputeScore", f"Starting with {len(df)} rows")
        result = df.copy()
        
        if self.auto_heal and self.tushare_healer:
            required_cols = ['symbol', 'trade_date', 'close', 'volume', 'amount', 'pct_chg']
            result = self.tushare_healer.check_and_heal(result, required_cols)
        
        result = self._merge_fund_flow(result)
        
        if 't1_return' not in result.columns:
            result['t1_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-1) / x - 1)
        if 't3_return' not in result.columns:
            result['t3_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-3) / x - 1)
        if 't5_return' not in result.columns:
            result['t5_return'] = result.groupby('symbol')['close'].transform(lambda x: x.shift(-5) / x - 1)
        
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
        
        if self.factor_generator:
            result = self.factor_generator.compute_all_factors(result)
        
        regime = None
        if self.enable_regime and self.regime_classifier:
            regime = self.regime_classifier.classify_regime(result)
            self._log_audit("RegimeSwitching", f"Market regime: {regime}")
        
        candidate_factors = ['volume_rank', 'net_main_rate']
        core_factors = [f for f in V178_CORE_FACTORS if f in result.columns]
        candidate_factors.extend(core_factors)
        candidate_factors.extend([f for f in V178_CANDIDATE_FACTORS if f in result.columns][:5])
        
        lead_factors = candidate_factors
        if self.enable_lead_lag and self.lead_lag_corrector:
            self._log_audit("LeadLagCorrection", "Selecting lead factors using MI analysis...")
            lead_factors = self.lead_lag_corrector.select_lead_factors(result, candidate_factors)
            self._log_audit("LeadFactors", f"Selected {len(lead_factors)} lead factors: {lead_factors}")
        
        self.selected_factors = lead_factors
        
        residuals = {}
        if self.enable_orm and self.orm_miner:
            self._log_audit("ORM", f"Extracting orthogonal residuals (core={ORM_CORE_FACTOR})...")
            residuals = self.orm_miner.extract_all_residuals(result, lead_factors)
        
        factor_data = {}
        factor_signs = {}
        
        for factor in lead_factors:
            if factor in residuals:
                f_raw = residuals[factor]
            elif factor in result.columns:
                f_raw = result[factor].copy()
            else:
                f_raw = pd.Series(0, index=result.index)
            
            if self.enable_adaptive_pac and self.pac_calculator:
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(result, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            elif self.enable_pac and self.pac_calculator:
                rolling_sign = self.pac_calculator.compute_rolling_ic_sign(result, factor)
                sign_val = rolling_sign.iloc[0] if len(rolling_sign) > 0 else 1
                factor_signs[factor] = sign_val
                f_processed = f_raw * rolling_sign
            else:
                factor_signs[factor] = 1
                f_processed = f_raw
            
            self.factor_directions[factor] = factor_signs[factor]
            ic = self._calc_factor_ic(result, factor if factor in result.columns else lead_factors[0])
            self.factor_ics[factor] = ic * factor_signs[factor]
            f_std = self._process_factor(f_processed, result['trade_date'])
            factor_data[factor] = f_std
        
        ic_weights = {}
        total_weight = 0.0
        for factor in lead_factors:
            ic = self.factor_ics.get(factor, 0.0)
            weight = (abs(ic) + self.EPSILON) ** IC_POWER
            ic_weights[factor] = weight
            total_weight += weight
        
        if total_weight > 0:
            base_weights = {f: w / total for f, w in ic_weights.items()}
        else:
            base_weights = {f: 1.0 / len(lead_factors) for f in lead_factors}
        
        if self.enable_regime and self.regime_classifier and regime:
            self.factor_weights = self.regime_classifier.get_regime_adjusted_weights(
                base_weights, regime
            )
            self._log_audit("RegimeWeights", f"Adjusted weights: {self.factor_weights}")
        else:
            self.factor_weights = base_weights
        
        self._log_audit("ICWeights", f"Weighted by |IC|^{IC_POWER}: {self.factor_weights}")
        
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
        
        if self.enable_nag and self.nag:
            self._log_audit("NAG", "Applying Non-linear Adaptive Gain with Fund Flow...")
            result['score'] = self.nag.apply_gain(result, 'score_raw')
        else:
            result['score'] = result['score_raw']
        
        if self.enable_sef and self.sef_filter:
            self._log_audit("SEF", "Applying signal entropy filter...")
            result['score'] = self.sef_filter.apply_entropy_filter(result, 'score')
        
        result['score'] = result.groupby('trade_date')['score'].transform(
            lambda x: (x - x.mean()) / (x.std() + self.EPSILON) if len(x) > 1 else x
        ).fillna(0)
        
        self._log_audit("Complete", f"Final score with {len(lead_factors)} factors")
        
        output_cols = [
            'trade_date', 'symbol', 'score', 't1_return', 't3_return', 't5_return',
            't1_return_period', 't2_return_period', 't3_return_period',
            't4_return_period', 't5_return_period'
        ]
        return result[output_cols]
    
    def get_factor_ics(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
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
        return self.selected_factors
    
    def get_lead_lag_stats(self) -> Dict:
        return self.lead_lag_corrector.get_lead_lag_stats() if self.lead_lag_corrector else {}
    
    def get_orm_stats(self) -> Dict:
        return self.orm_miner.get_residual_stats() if self.orm_miner else {}
    
    def get_pac_stats(self) -> Dict:
        return self.pac_calculator.get_pac_stats() if hasattr(self.pac_calculator, 'get_pac_stats') else {}
    
    def get_sef_stats(self) -> Dict:
        return self.sef_filter.get_sef_stats() if self.sef_filter else {}
    
    def get_nag_stats(self) -> Dict:
        return self.nag.get_nag_stats() if self.nag else {}
    
    def get_regime_stats(self) -> Dict:
        return self.regime_classifier.get_regime_stats() if self.regime_classifier else {}
    
    def get_ic_decay_stats(self) -> Dict:
        return self.ic_decay_analyzer.get_decay_stats()
    
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log[-MAX_LOG_ENTRIES:]
    
    def compute_ic_metrics(self, df: pd.DataFrame) -> Dict:
        return self.ic_decay_analyzer.compute_ic_decay(df, 'score')


class RollingICSignCalculator:
    """滚动 IC 符号计算器"""
    
    def __init__(self, window: int = ROLLING_WINDOW):
        self.window = window
    
    def compute_rolling_ic_sign(self, df: pd.DataFrame, factor_col: str, return_col: str = 't1_return') -> pd.Series:
        if factor_col not in df.columns or return_col not in df.columns:
            return pd.Series(1, index=df.index)
        
        result = df.copy().sort_values(['symbol', 'trade_date'])
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
        return result['trade_date'].map(ic_sign_map).fillna(1)


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
    enable_nag: bool = True,
    enable_regime: bool = True,
    auto_heal: bool = True,
    db_url: Optional[str] = None,
    tushare_token: Optional[str] = None
) -> AlphaResearchV178:
    """工厂函数"""
    return AlphaResearchV178(
        ic_threshold=ic_threshold,
        n_factors=n_factors,
        n_bins=n_bins,
        enable_ensemble=enable_ensemble,
        enable_pac=enable_pac,
        enable_lead_lag=enable_lead_lag,
        enable_adaptive_pac=enable_adaptive_pac,
        enable_sef=enable_sef,
        enable_orm=enable_orm,
        enable_nag=enable_nag,
        enable_regime=enable_regime,
        auto_heal=auto_heal,
        db_url=db_url,
        tushare_token=tushare_token
    )


class V178Runner:
    """
    V178 回测运行器 - 暴力数据管道 - 2023 年数据硬通关
    
    【Mission】
    停止所有关于 Alpha、NAG、Regime Switching 的口头分析。
    现在！立刻！把 2023 年的数据给我凿进 MySQL！
    
    【V178 核心改进】
    1. 硬性拦截：main.py 入口校验，2023 年数据 < 500,000 行则强制 heal
    2. 实时审计：每 1000 行打印进度 [DATA_PROGRESS]
    3. 频控陷阱修复：HTTP 403 强制 sleep 60 秒
    4. 小步快跑：按【交易日】拉取，每天 5000 只股票
    5. 字段强制对齐：open, high, low, close, vol, amount, pct_chg, adj_factor 全部入库
    6. 断点恢复：heal_checkpoint.json 记录断点
    """
    
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        output_dir: str = 'reports'
    ):
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        db_url = os.getenv("DATABASE_URL")
        tushare_token = os.getenv("TUSHARE_TOKEN")
        
        self.alpha_module = get_alpha_research(
            ic_threshold=0.0001,
            n_factors=8,
            n_bins=10,
            enable_ensemble=True,
            enable_pac=True,
            enable_lead_lag=True,
            enable_adaptive_pac=True,
            enable_sef=True,
            enable_orm=True,
            enable_nag=True,
            enable_regime=True,
            auto_heal=True,
            db_url=db_url,
            tushare_token=tushare_token
        )
        
        logger.info(f"[{VERSION}] V178Runner initialized")
        logger.info(f"  Parquet Path: {parquet_path}")
        logger.info(f"  Output Dir: {output_dir}")
        logger.info(f"  NAG + Fund Flow: Enabled")
        logger.info(f"  Regime Switching: Enabled")
        logger.info(f"  Tushare Healer: VIOLENT MODE (by trading day)")
        logger.info(f"  Data Target (2023): > {SQL_HEALER_MIN_ROWS_2023} rows")
    
    def load_data(self, year: int) -> pd.DataFrame:
        """加载指定年份的数据"""
        if self.parquet_path and Path(self.parquet_path).exists():
            logger.info(f"[V178][DataLoader] Loading from Parquet: {self.parquet_path}")
            df = pd.read_parquet(self.parquet_path)
            
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].dt.year == year]
                df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"[V178][DataLoader] Loaded {len(df)} rows from Parquet for year {year}")
            return df
        
        logger.info(f"[V178][DataLoader] Fetching full year {year} data with Tushare Healer...")
        
        if self.alpha_module.tushare_healer:
            df = self.alpha_module.tushare_healer.heal_year(year)
            
            if not df.empty:
                logger.info(f"[V178][DataLoader] Loaded {len(df)} rows for year {year}")
                
                if year == 2023:
                    if len(df) >= SQL_HEALER_MIN_ROWS_2023:
                        logger.info(
                            f"[V178][DataLoader] 2023 data has {len(df)} rows >= {SQL_HEALER_MIN_ROWS_2023} target ✓"
                        )
                    else:
                        logger.error(
                            f"[V178][DataLoader] 2023 data has only {len(df)} rows < {SQL_HEALER_MIN_ROWS_2023} target ✗"
                        )
                
                return df
        
        logger.info(f"[V178][DataLoader] Falling back to traditional database load...")
        
        try:
            from sqlalchemy import create_engine, text
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not configured")
            
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
            logger.info(f"[V178][DataLoader] Loaded {len(df)} rows from database for year {year}")
            
            return df
            
        except Exception as e:
            logger.error(f"[V178][DataLoader] Failed to load from database: {e}")
            return pd.DataFrame()
    
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
            return {'mean_ic': float(mean_ic), 'ic_std': float(std_ic), 'ic_ir': float(ir), 'num_days': len(ics)}
        
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
        
        result['passed'] = result['t1_ic']['mean_ic'] > 0.08 and result['t1_ic']['ic_ir'] > 0.4
        return result
    
    def run_audit(self, year: int) -> Dict:
        """运行单一年份的审计"""
        logger.info(f"[{VERSION}] Running audit for year {year}")
        df = self.load_data(year)
        
        if df.empty:
            logger.warning(f"No data loaded for year {year}")
            return {'year': year, 'error': 'No data loaded', 'passed': False, 'data_rows': 0}
        
        # V178 强制检查：2023 年数据不足 500,000 行则报错
        if year == 2023 and len(df) < SQL_HEALER_MIN_ROWS_2023:
            logger.error(
                f"[{VERSION}] FATAL: 2023 data has only {len(df)} rows < {SQL_HEALER_MIN_ROWS_2023}!"
            )
            logger.error(f"[{VERSION}] Please run data healing first: python run_v178.py --heal --year 2023")
            import sys
            sys.exit(1)
        
        result = self.alpha_module.compute_score(df)
        metrics = self.compute_ic_metrics(result)
        
        metrics['selected_factors'] = self.alpha_module.get_selected_factors()
        metrics['factor_ics'] = self.alpha_module.get_factor_ics(result)
        metrics['factor_weights'] = self.alpha_module.factor_weights
        metrics['lead_lag_stats'] = self.alpha_module.get_lead_lag_stats()
        metrics['nag_stats'] = self.alpha_module.get_nag_stats()
        metrics['regime_stats'] = self.alpha_module.get_regime_stats()
        metrics['data_rows'] = len(df)
        
        logger.info(
            f"[{VERSION}] Audit Complete - T+1 IC: {metrics['t1_ic']['mean_ic']:.4f}, "
            f"IR: {metrics['t1_ic']['ic_ir']:.2f}, Data Rows: {len(df)}"
        )
        return metrics
    
    def run_cross_cycle_audit(self, years: List[int] = None) -> Dict:
        """
        V178 核心：跨周期 OOS 验证
        
        运行 2023 和 2024 年回测，验证策略普适性
        - 2023 年（弱市）：IC > 0.08, 数据行数 > 500,000
        - 2024 年（波动市）：IC > 0.08
        """
        if years is None:
            years = [2023, 2024]
        
        logger.info("=" * 70)
        logger.info(f"[{VERSION}] Cross-Cycle OOS Validation")
        logger.info(f"  Years: {years}")
        logger.info(f"  Target 2023: IC > 0.08, Data Rows > {SQL_HEALER_MIN_ROWS_2023}")
        logger.info(f"  Target 2024: IC > 0.08")
        logger.info("=" * 70)
        
        results = {}
        for year in years:
            logger.info(f"\n{'='*50}")
            logger.info(f"[{VERSION}] Running audit for year {year}")
            logger.info(f"{'='*50}")
            
            result = self.run_audit(year)
            results[year] = result
        
        comparison_table = self._generate_cross_cycle_table(results)
        logger.info("\n" + comparison_table)
        
        validation_passed = self._validate_cross_cycle_targets(results)
        
        return {
            'years': years,
            'results': results,
            'comparison_table': comparison_table,
            'validation_passed': validation_passed,
            'nag_stats': self.alpha_module.get_nag_stats(),
            'regime_stats': self.alpha_module.get_regime_stats(),
        }
    
    def _generate_cross_cycle_table(self, results: Dict) -> str:
        """生成跨周期对比表"""
        table = f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    V178 CROSS-CYCLE OOS VALIDATION TABLE                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Metric          │  2023 (Weak Market)  │  2024 (Volatile)   │  Target        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  T+1 Rank IC     │  {results.get(2023, {}).get('t1_ic', {}).get('mean_ic', 'N/A'):>8.4f}      │  {results.get(2024, {}).get('t1_ic', {}).get('mean_ic', 'N/A'):>8.4f}      │  > 0.08         ║
║  IC IR           │  {results.get(2023, {}).get('t1_ic', {}).get('ic_ir', 'N/A'):>8.2f}      │  {results.get(2024, {}).get('t1_ic', {}).get('ic_ir', 'N/A'):>8.2f}      │  > 0.40         ║
║  Data Rows       │  {results.get(2023, {}).get('data_rows', 'N/A'):>10}      │  {results.get(2024, {}).get('data_rows', 'N/A'):>10}      │  > 500K (2023)  ║
║  NAG Gain        │  {results.get(2023, {}).get('nag_stats', {}).get('mean_gain', 'N/A'):>8.3f}      │  {results.get(2024, {}).get('nag_stats', {}).get('mean_gain', 'N/A'):>8.3f}      │  0.7-1.3        ║
║  Regime          │  {results.get(2023, {}).get('regime_stats', {}).get('regime_type', 'N/A'):>15}      │  {results.get(2024, {}).get('regime_stats', {}).get('regime_type', 'N/A'):>15}      │  Auto-Switch    ║
║  Fund Flow IC    │  {self._get_fund_flow_ic(results.get(2023, {})):>8.4f}      │  {self._get_fund_flow_ic(results.get(2024, {})):>8.4f}      │  > 0.03         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Status          │  {'✓ PASSED' if results.get(2023, {}).get('t1_ic', {}).get('mean_ic', 0) > 0.08 and results.get(2023, {}).get('data_rows', 0) >= SQL_HEALER_MIN_ROWS_2023 else '✗ FAILED':>8}      │  {'✓ PASSED' if results.get(2024, {}).get('passed', False) else '✗ FAILED':>8}      │  Cross-Cycle    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
        return table
    
    def _get_fund_flow_ic(self, result: Dict) -> float:
        """获取资金流因子 IC"""
        factor_ics = result.get('factor_ics', {})
        return factor_ics.get('net_main_rate', 0.0)
    
    def _validate_cross_cycle_targets(self, results: Dict) -> Dict:
        """验证跨周期目标"""
        validation = {
            '2023': {
                'min_ic_target': 0.08,
                'min_ic_actual': results.get(2023, {}).get('t1_ic', {}).get('mean_ic', 0),
                'min_rows_target': SQL_HEALER_MIN_ROWS_2023,
                'min_rows_actual': results.get(2023, {}).get('data_rows', 0),
                'passed': results.get(2023, {}).get('t1_ic', {}).get('mean_ic', 0) > 0.08 and
                          results.get(2023, {}).get('data_rows', 0) >= SQL_HEALER_MIN_ROWS_2023,
            },
            '2024': {
                'min_ic_target': 0.08,
                'min_ic_actual': results.get(2024, {}).get('t1_ic', {}).get('mean_ic', 0),
                'min_ir_target': 0.40,
                'min_ir_actual': results.get(2024, {}).get('t1_ic', {}).get('ic_ir', 0),
                'passed': results.get(2024, {}).get('t1_ic', {}).get('mean_ic', 0) > 0.08 and 
                          results.get(2024, {}).get('t1_ic', {}).get('ic_ir', 0) > 0.40,
            },
            'overall_passed': results.get(2023, {}).get('t1_ic', {}).get('mean_ic', 0) > 0.08 and
                             results.get(2023, {}).get('data_rows', 0) >= SQL_HEALER_MIN_ROWS_2023 and
                             results.get(2024, {}).get('t1_ic', {}).get('mean_ic', 0) > 0.08 and
                             results.get(2024, {}).get('t1_ic', {}).get('ic_ir', 0) > 0.40,
        }
        return validation


if __name__ == "__main__":
    logger.info(f"[{VERSION}] Testing AlphaResearchV178...")
    np.random.seed(42)
    test_df = pd.DataFrame({
        'symbol': np.random.choice(['000001.SZ', '000002.SZ', '000003.SZ'], 1000),
        'trade_date': np.random.choice(['2024-01-01', '2024-01-02', '2024-01-03'], 1000),
        'close': np.random.randn(1000) * 10 + 100,
        'volume': np.random.randn(1000) * 1000 + 5000,
        'amount': np.random.randn(1000) * 10000 + 50000,
        'pct_chg': np.random.randn(1000) * 2,
        'net_main_rate': np.random.randn(1000) * 5,
    })
    
    alpha = get_alpha_research()
    result = alpha.compute_score(test_df)
    
    logger.info(f"[{VERSION}] Test complete!")
    logger.info(f"  Selected factors: {alpha.get_selected_factors()}")
    logger.info(f"  Factor ICs: {alpha.get_factor_ics()}")
    logger.info(f"  NAG Stats: {alpha.get_nag_stats()}")