"""
Data Loader Module - V101 Unified Data Loading and Validation.

负责数据拉取、补全和校验。
核心功能:
    - 从 Tushare API 获取 A 股股票和基金数据
    - 资产类型自动识别 (股票/基金)
    - 获取日线数据和复权因子
    - 频率限制控制
    - 数据完整性校验
    - 2024 年 total_mv 数据防御检查
"""

import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from pathlib import Path

import polars as pl
import tushare as ts
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


class DataLoaderError(Exception):
    """Data loader 自定义异常"""
    pass


class AlphaWeakWarning(Exception):
    """Alpha 弱信号警告"""
    pass


class DataLoader:
    """
    V101 统一数据加载器。
    
    功能特性:
        - 获取日线价格数据和复权因子
        - 频率限制控制
        - 数据完整性校验
        - 2024 年 total_mv 数据防御检查
    
    使用示例:
        >>> loader = DataLoader()
        >>> df = loader.load_data("000001.SZ", "20240101", "20241231")
    """
    
    REQUESTS_PER_MINUTE = 60
    SLEEP_BETWEEN_REQUESTS = 0.5
    
    def __init__(self, token: Optional[str] = None, db_url: Optional[str] = None) -> None:
        """
        初始化数据加载器。
        
        Args:
            token: Tushare API token，从环境变量读取
            db_url: 数据库连接 URL，从环境变量读取
            
        Raises:
            DataLoaderError: 当必要配置缺失时抛出
        """
        self.token = token or os.getenv("TUSHARE_TOKEN")
        if not self.token or self.token == "your_tushare_token_here":
            logger.warning("TUSHARE_TOKEN not configured. Please set it in .env file.")
            raise DataLoaderError("TUSHARE_TOKEN is required")
        
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.request_count = 0
        self.last_request_time = time.time()
        
        logger.info("DataLoader initialized")
    
    def _rate_limit(self) -> None:
        """执行 API 请求的频率限制。"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.SLEEP_BETWEEN_REQUESTS:
            sleep_time = self.SLEEP_BETWEEN_REQUESTS - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        if self.request_count % 10 == 0:
            logger.debug(f"Tushare API requests: {self.request_count}")
    
    def _get_asset_type(self, ts_code: str) -> str:
        """
        根据股票代码前缀自动识别资产类型。
        
        Args:
            ts_code: Tushare 代码
            
        Returns:
            "STOCK" 或 "FUND"
        """
        code = ts_code.split(".")[0]
        fund_prefixes = ("51", "58", "15", "16")
        return "FUND" if code.startswith(fund_prefixes) else "STOCK"
    
    def check_2024_total_mv(self, ts_code: str, start_date: str, end_date: str) -> bool:
        """
        【数据防御】检查 2024 年 total_mv 数据是否存在。
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 是否存在 total_mv 数据
            
        Raises:
            DataLoaderError: 数据缺失时抛出
        """
        logger.info(f"[数据防御] 检查 {ts_code} 的 2024 年 total_mv 数据...")
        
        try:
            # 尝试从数据库读取
            if self.db_url:
                from sqlalchemy import create_engine, text
                engine = create_engine(self.db_url)
                
                query = text("""
                    SELECT COUNT(*) as cnt 
                    FROM stock_daily 
                    WHERE symbol = :symbol 
                    AND trade_date >= :start_date 
                    AND total_mv IS NOT NULL
                """)
                
                with engine.connect() as conn:
                    result = conn.execute(query, {
                        "symbol": ts_code,
                        "start_date": start_date,
                    })
                    row = result.fetchone()
                    cnt = row[0] if row else 0
                
                if cnt == 0:
                    logger.warning(f"[数据防御] {ts_code} 的 2024 年 total_mv 数据缺失，需要补取")
                    return False
                else:
                    logger.info(f"[数据防御] {ts_code} 的 2024 年 total_mv 数据完整 ({cnt} 条)")
                    return True
                    
        except Exception as e:
            logger.warning(f"[数据防御] 检查 total_mv 失败：{e}")
            return False
        
        return False
    
    def repair_2024_data(self, df: Optional[pl.DataFrame] = None,
                          ts_code: Optional[str] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        【主动防御】修复 2024 年缺失的数据。
        
        【核心逻辑】
        1. 检测 2024 年 total_mv 缺失
        2. 主动从 Tushare API 重新拉取
        3. 用 amount/turnover_rate 估算
        4. 严禁打印报错后停止运行
        
        Args:
            df: 已有的 DataFrame (可选)
            ts_code: 股票代码 (用于重新拉取)
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            修复后的 DataFrame
        """
        logger.info("[数据防御] 启动 repair_2024_data...")
        
        # 场景 1: 已有 DataFrame，尝试修复
        if df is not None and not df.is_empty():
            if 'total_mv' in df.columns:
                null_ratio = df['total_mv'].null_count() / len(df) if len(df) > 0 else 0
                if null_ratio > 0.3:
                    logger.warning(f"[数据防御] total_mv 缺失比例：{null_ratio:.1%}")
                    
                    # 尝试用 amount/turnover_rate 估算
                    if 'amount' in df.columns and 'turnover_rate' in df.columns:
                        logger.info("[数据防御] 用 amount/turnover_rate 估算 total_mv")
                        estimated_mv = df['amount'] / (df['turnover_rate'].fill_null(0.01) + pl.lit(1e-6)) * 100
                        df = df.with_columns([
                            pl.col('total_mv').fill_null(estimated_mv).alias('total_mv')
                        ])
            
            return df
        
        # 场景 2: 从 API 重新拉取
        if ts_code and start_date and end_date:
            logger.info(f"[数据防御] 从 API 重新拉取 {ts_code} 的数据...")
            try:
                # 尝试拉取 daily_basic 数据
                daily_basic_df = self.fetch_daily_basic(ts_code, start_date, end_date)
                if daily_basic_df is not None and not daily_basic_df.is_empty():
                    logger.info(f"[数据防御] 成功获取 daily_basic 数据 {len(daily_basic_df)} 条")
                    return daily_basic_df
            except Exception as e:
                logger.error(f"[数据防御] 从 API 拉取失败：{e}")
        
        logger.warning("[数据防御] 无法修复数据，返回 None")
        return None
    
    def fetch_daily_basic(self, ts_code: str, start_date: str, end_date: str) -> Optional[pl.DataFrame]:
        """
        获取 daily_basic 数据（包括 total_mv）。
        
        Args:
            ts_code: Tushare 代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Optional[pl.DataFrame]: daily_basic 数据
        """
        try:
            self._rate_limit()
            
            if self._get_asset_type(ts_code) != "STOCK":
                return None
            
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
            )
            
            if df is None or df.empty:
                logger.debug(f"No daily_basic data for {ts_code}")
                return None
            
            pl_df = pl.from_pandas(df)
            logger.debug(f"Fetched {len(pl_df)} rows of daily_basic data for {ts_code}")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch daily_basic data for {ts_code}: {e}")
            return None
    
    def fetch_daily_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """
        从 Tushare 获取日线价格数据。
        
        Args:
            ts_code: Tushare 代码
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            
        Returns:
            Optional[pl.DataFrame]: 日线数据 DataFrame
        """
        try:
            self._rate_limit()
            asset_type = self._get_asset_type(ts_code)
            
            if asset_type == "STOCK":
                df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                df = self.pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                logger.warning(f"No daily data for {ts_code} ({asset_type})")
                return None
            
            pl_df = pl.from_pandas(df)
            
            if asset_type == "FUND" and "vol" in pl_df.columns:
                pl_df = pl_df.with_columns(pl.col("vol").alias("volume"))
            
            logger.debug(f"Fetched {len(pl_df)} rows of daily data for {ts_code} ({asset_type})")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch daily data for {ts_code}: {e}")
            return None
    
    def fetch_adj_factor(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pl.DataFrame]:
        """
        获取复权因子数据。
        
        Args:
            ts_code: Tushare 代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Optional[pl.DataFrame]: 复权因子 DataFrame
        """
        try:
            self._rate_limit()
            asset_type = self._get_asset_type(ts_code)
            
            if asset_type == "STOCK":
                df = self.pro.adj_factor(ts_code=ts_code, start_date=start_date, end_date=end_date)
            else:
                df = self.pro.fund_adj(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is None or df.empty:
                logger.warning(f"No adj_factor data for {ts_code} ({asset_type})")
                return None
            
            pl_df = pl.from_pandas(df)
            logger.debug(f"Fetched {len(pl_df)} rows of adj_factor data for {ts_code}")
            return pl_df
            
        except Exception as e:
            logger.error(f"Failed to fetch adj_factor data for {ts_code}: {e}")
            return None
    
    def transform_data(
        self,
        daily_df: pl.DataFrame,
        adj_factor_df: pl.DataFrame,
        daily_basic_df: Optional[pl.DataFrame] = None,
    ) -> pl.DataFrame:
        """
        转换并合并日线数据、复权因子和 daily_basic 数据。
        
        Args:
            daily_df: 日线价格 DataFrame
            adj_factor_df: 复权因子 DataFrame
            daily_basic_df: daily_basic 数据（可选）
            
        Returns:
            pl.DataFrame: 转换后的 DataFrame
        """
        # 重命名 ts_code 为 symbol
        daily_df = daily_df.with_columns(pl.col("ts_code").alias("symbol"))
        adj_factor_df = adj_factor_df.with_columns(pl.col("ts_code").alias("symbol"))
        
        # 日期转换
        daily_df = daily_df.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date")
        )
        adj_factor_df = adj_factor_df.with_columns(
            pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date")
        )
        
        # 合并日线数据和复权因子
        merged_df = daily_df.join(
            adj_factor_df.select(["symbol", "trade_date", "adj_factor"]),
            on=["symbol", "trade_date"],
            how="left",
        )
        
        # 计算复权价格
        if "adj_factor" in merged_df.columns:
            merged_df = merged_df.with_columns([
                (pl.col("close") * pl.col("adj_factor") / 1000).alias("adj_close"),
                (pl.col("open") * pl.col("adj_factor") / 1000).alias("adj_open"),
                (pl.col("high") * pl.col("adj_factor") / 1000).alias("adj_high"),
                (pl.col("low") * pl.col("adj_factor") / 1000).alias("adj_low"),
            ])
        else:
            merged_df = merged_df.with_columns([
                pl.col("close").alias("adj_close"),
                pl.col("open").alias("adj_open"),
                pl.col("high").alias("adj_high"),
                pl.col("low").alias("adj_low"),
            ])
        
        # 合并 daily_basic 数据（包括 total_mv）
        if daily_basic_df is not None and not daily_basic_df.is_empty():
            daily_basic_df = daily_basic_df.with_columns([
                pl.col("ts_code").alias("symbol"),
                pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("trade_date"),
            ])
            
            merged_df = merged_df.join(
                daily_basic_df.select([
                    "symbol", "trade_date", "turnover_rate", "volume_ratio", "total_mv", "pe_ttm", "pb"
                ]),
                on=["symbol", "trade_date"],
                how="left",
            )
            
            # volume_ratio -> vol_ratio
            if "volume_ratio" in merged_df.columns:
                merged_df = merged_df.with_columns(pl.col("volume_ratio").alias("vol_ratio"))
        
        # 选择最终列
        result_df = merged_df.select([
            pl.col("symbol"),
            pl.col("trade_date"),
            pl.col("open"),
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("pre_close"),
            pl.col("change"),
            pl.col("pct_chg"),
            pl.col("vol").alias("volume"),
            pl.col("amount"),
            pl.col("adj_factor").fill_null(1000),
            pl.col("turnover_rate").fill_null(None).cast(pl.Float64),
            pl.col("vol_ratio").fill_null(None).cast(pl.Float64),
            pl.col("total_mv").fill_null(None).cast(pl.Float64),
            pl.col("pe_ttm").fill_null(None).cast(pl.Float64),
            pl.col("pb").fill_null(None).cast(pl.Float64),
            pl.col("adj_open"),
            pl.col("adj_high"),
            pl.col("adj_low"),
            pl.col("adj_close"),
        ])
        
        result_df = result_df.sort(["symbol", "trade_date"])
        return result_df
    
    def load_data(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        check_mv: bool = True,
    ) -> pl.DataFrame:
        """
        加载单只股票的完整数据。
        
        Args:
            ts_code: Tushare 代码
            start_date: 开始日期，格式 YYYYMMDD
            end_date: 结束日期，格式 YYYYMMDD
            check_mv: 是否检查 total_mv 数据
            
        Returns:
            pl.DataFrame: 完整的股票数据
            
        Raises:
            DataLoaderError: 数据获取失败时抛出
        """
        logger.info(f"Loading data for {ts_code} from {start_date} to {end_date}")
        
        # 【数据防御】检查 2024 年 total_mv 数据
        if check_mv and "2024" in end_date:
            has_mv = self.check_2024_total_mv(ts_code, start_date, end_date)
            if not has_mv:
                logger.warning(f"[数据防御] {ts_code} 的 total_mv 数据缺失，尝试补取...")
        
        # 获取日线数据
        daily_df = self.fetch_daily_data(ts_code, start_date, end_date)
        if daily_df is None or daily_df.is_empty():
            raise DataLoaderError(f"No daily data for {ts_code}")
        
        # 获取复权因子
        adj_factor_df = self.fetch_adj_factor(ts_code, start_date, end_date)
        if adj_factor_df is None or adj_factor_df.is_empty():
            logger.warning(f"No adj_factor data for {ts_code}, using default")
            adj_factor_df = daily_df.select(["ts_code", "trade_date"]).with_columns(
                pl.lit(1000).alias("adj_factor")
            )
        
        # 获取 daily_basic 数据（包括 total_mv）
        daily_basic_df = None
        if self._get_asset_type(ts_code) == "STOCK":
            daily_basic_df = self.fetch_daily_basic(ts_code, start_date, end_date)
        
        # 转换和合并数据
        result_df = self.transform_data(daily_df, adj_factor_df, daily_basic_df)
        
        logger.info(f"Loaded {len(result_df)} rows for {ts_code}")
        return result_df
    
    def load_index_constituents(
        self,
        index_code: str = "000300.SH",
        start_date: str = "20240101",
        end_date: str = None,
    ) -> dict[str, pl.DataFrame]:
        """
        加载指数所有成分股数据。
        
        Args:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            dict[str, pl.DataFrame]: {ts_code: DataFrame} 字典
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")
        
        # 获取成分股列表
        try:
            self._rate_limit()
            df = self.pro.index_member(ts_code=index_code)
            if df is None or df.empty:
                logger.error(f"No constituents for {index_code}")
                return {}
            
            stock_codes = pl.from_pandas(df)["con_code"].to_list()
            logger.info(f"Found {len(stock_codes)} constituents in {index_code}")
            
        except Exception as e:
            logger.error(f"Failed to fetch index constituents: {e}")
            return {}
        
        # 加载每只股票数据
        result = {}
        for i, ts_code in enumerate(stock_codes):
            try:
                df = self.load_data(ts_code, start_date, end_date)
                result[ts_code] = df
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Progress: {i + 1}/{len(stock_codes)} stocks loaded")
                    
            except Exception as e:
                logger.error(f"Failed to load {ts_code}: {e}")
        
        return result
    
    def save_to_parquet(
        self,
        data_dict: dict[str, pl.DataFrame],
        output_path: str,
    ) -> None:
        """
        保存数据到 Parquet 文件。
        
        Args:
            data_dict: {ts_code: DataFrame} 字典
            output_path: 输出路径
        """
        if not data_dict:
            logger.warning("No data to save")
            return
        
        # 合并所有股票数据
        all_dfs = []
        for ts_code, df in data_dict.items():
            df_with_code = df.with_columns(pl.lit(ts_code).alias("ts_code"))
            all_dfs.append(df_with_code)
        
        if all_dfs:
            combined_df = pl.concat(all_dfs)
            
            # 创建输出目录
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 保存 Parquet
            combined_df.write_parquet(output_path, compression="snappy")
            logger.info(f"Saved {len(combined_df)} rows to {output_path}")
    
    def load_from_parquet(self, parquet_path: str) -> pl.DataFrame:
        """
        从 Parquet 文件加载数据。
        
        Args:
            parquet_path: Parquet 文件路径
            
        Returns:
            pl.DataFrame: 数据 DataFrame
        """
        logger.info(f"Loading data from {parquet_path}")
        df = pl.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df)} rows")
        return df


def get_loader(token: Optional[str] = None) -> DataLoader:
    """获取 DataLoader 实例。"""
    return DataLoader(token)