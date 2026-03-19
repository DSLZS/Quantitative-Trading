"""
V41 Data Loader Module - 数据加载模块

【核心功能】
1. 从数据库加载股票数据
2. 数据清洗和验证
3. 板块/行业数据加载
4. 内存优化（使用 LazyFrame 和流式处理）

作者：量化系统
版本：V41.0
日期：2026-03-19
"""

import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
import numpy as np
import polars as pl
import polars.selectors as cs
from loguru import logger

try:
    from ..db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 数据加载配置
# ===========================================

# 默认数据查询配置
DEFAULT_CHUNK_SIZE = 10000  # 数据库读取块大小
MARKET_INDEX_SYMBOL = "000300.SH"  # 市场指数代码

# 内置备用成分股列表（50 只蓝筹股）
FALLBACK_STOCKS = [
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "300750.SZ", "name": "宁德时代"},
    {"symbol": "000858.SZ", "name": "五粮液"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "600036.SH", "name": "招商银行"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "002415.SZ", "name": "海康威视"},
    {"symbol": "601888.SH", "name": "中国中免"},
    {"symbol": "600276.SH", "name": "恒瑞医药"},
    {"symbol": "601166.SH", "name": "兴业银行"},
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000002.SZ", "name": "万科 A"},
    {"symbol": "600030.SH", "name": "中信证券"},
    {"symbol": "000651.SZ", "name": "格力电器"},
    {"symbol": "000725.SZ", "name": "京东方 A"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "300059.SZ", "name": "东方财富"},
    {"symbol": "601398.SH", "name": "工商银行"},
    {"symbol": "601988.SH", "name": "中国银行"},
    {"symbol": "601857.SH", "name": "中国石油"},
    {"symbol": "600000.SH", "name": "浦发银行"},
    {"symbol": "600016.SH", "name": "民生银行"},
    {"symbol": "600028.SH", "name": "中国石化"},
    {"symbol": "600031.SH", "name": "三一重工"},
    {"symbol": "600048.SH", "name": "保利发展"},
    {"symbol": "600050.SH", "name": "中国联通"},
    {"symbol": "600104.SH", "name": "上汽集团"},
    {"symbol": "600309.SH", "name": "万华化学"},
    {"symbol": "600436.SH", "name": "片仔癀"},
    {"symbol": "600585.SH", "name": "海螺水泥"},
    {"symbol": "600588.SH", "name": "用友网络"},
    {"symbol": "600690.SH", "name": "海尔智家"},
    {"symbol": "600809.SH", "name": "山西汾酒"},
    {"symbol": "600887.SH", "name": "伊利股份"},
    {"symbol": "600900.SH", "name": "长江电力"},
    {"symbol": "601012.SH", "name": "隆基绿能"},
    {"symbol": "601088.SH", "name": "中国神华"},
    {"symbol": "601288.SH", "name": "农业银行"},
    {"symbol": "601328.SH", "name": "交通银行"},
    {"symbol": "601601.SH", "name": "中国太保"},
    {"symbol": "601628.SH", "name": "中国人寿"},
    {"symbol": "601668.SH", "name": "中国建筑"},
    {"symbol": "601688.SH", "name": "华泰证券"},
    {"symbol": "601766.SH", "name": "中国中车"},
    {"symbol": "601816.SH", "name": "京沪高铁"},
    {"symbol": "601898.SH", "name": "中煤能源"},
    {"symbol": "601919.SH", "name": "中远海控"},
    {"symbol": "601939.SH", "name": "建设银行"},
    {"symbol": "601985.SH", "name": "中国核电"},
    {"symbol": "601995.SH", "name": "中金公司"},
]


@dataclass
class DataLoaderConfig:
    """数据加载器配置"""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    use_lazy: bool = True  # 默认使用 LazyFrame
    cache_enabled: bool = True  # 启用缓存
    validate_data: bool = True  # 验证数据质量
    start_date: str = "2025-01-01"
    end_date: str = "2026-03-19"


class DataLoader:
    """
    V41 数据加载器 - 模块化设计
    
    【核心功能】
    1. 从数据库加载股票数据
    2. 数据清洗和验证
    3. 板块/行业数据加载
    4. 内存优化（使用 LazyFrame 和流式处理）
    """
    
    def __init__(self, db=None, config: DataLoaderConfig = None):
        self.db = db or DatabaseManager.get_instance()
        self.config = config or DataLoaderConfig()
        self._data_cache: Dict[str, pl.DataFrame] = {}
    
    def load_stock_data(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
        use_fallback: bool = True,
    ) -> pl.DataFrame:
        """
        加载股票数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbols: 股票代码列表，None 表示加载所有股票
            use_fallback: 是否使用备用股票列表
        
        Returns:
            包含股票数据的 DataFrame
        """
        cache_key = f"stock_{start_date}_{end_date}_{','.join(symbols or [])}"
        
        if self.config.cache_enabled and cache_key in self._data_cache:
            logger.info(f"Using cached data for {cache_key}")
            return self._data_cache[cache_key].clone()
        
        logger.info(f"Loading stock data from {start_date} to {end_date}...")
        
        try:
            # 构建查询
            if symbols:
                symbol_list = symbols
            elif use_fallback:
                symbol_list = [s["symbol"] for s in FALLBACK_STOCKS]
            else:
                # 从数据库获取所有股票
                symbol_query = "SELECT DISTINCT symbol FROM stock_daily"
                symbol_df = self.db.read_sql(symbol_query)
                symbol_list = symbol_df["symbol"].to_list() if not symbol_df.is_empty() else []
            
            if not symbol_list:
                logger.warning("No symbols to load, using fallback list")
                symbol_list = [s["symbol"] for s in FALLBACK_STOCKS]
            
            # 分批加载数据（内存优化）
            all_data = []
            batch_size = 10  # 每批处理 10 只股票
            
            for i in range(0, len(symbol_list), batch_size):
                batch_symbols = symbol_list[i:i + batch_size]
                placeholders = ",".join(["%s"] * len(batch_symbols))
                
                query = f"""
                    SELECT symbol, trade_date, open, high, low, close, volume, turnover_rate
                    FROM stock_daily
                    WHERE trade_date >= %s AND trade_date <= %s AND symbol IN ({placeholders})
                    ORDER BY symbol, trade_date
                """
                
                params = [start_date, end_date] + batch_symbols
                batch_df = self.db.read_sql(query, params)
                
                if not batch_df.is_empty():
                    all_data.append(batch_df)
                    logger.debug(f"Loaded batch {i // batch_size + 1}/{(len(symbol_list) + batch_size - 1) // batch_size}")
            
            if not all_data:
                logger.warning("No data loaded from database, generating fallback data")
                return self._generate_fallback_data(start_date, end_date, symbol_list)
            
            # 合并数据
            df = pl.concat(all_data)
            
            # 数据清洗
            df = self._clean_data(df)
            
            # 数据验证
            if self.config.validate_data:
                self._validate_data(df)
            
            # 缓存
            if self.config.cache_enabled:
                self._data_cache[cache_key] = df
            
            logger.info(f"Loaded {len(df)} records with {df['symbol'].n_unique()} stocks")
            return df
            
        except Exception as e:
            logger.error(f"load_stock_data FAILED: {e}")
            logger.error(traceback.format_exc())
            # 回退到生成模拟数据
            symbol_list = [s["symbol"] for s in FALLBACK_STOCKS]
            return self._generate_fallback_data(start_date, end_date, symbol_list)
    
    def load_industry_data(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """
        加载行业板块数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            包含行业数据的 DataFrame
        """
        logger.info(f"Loading industry data from {start_date} to {end_date}...")
        
        try:
            query = """
                SELECT symbol, trade_date, industry_name, industry_mv_ratio
                FROM stock_industry_daily
                WHERE trade_date >= %s AND trade_date <= %s
                ORDER BY trade_date, industry_name
            """
            
            df = self.db.read_sql(query, [start_date, end_date])
            
            if df.is_empty():
                logger.warning("No industry data found")
                return pl.DataFrame()
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load industry data: {e}")
            return pl.DataFrame()
    
    def load_market_index_data(
        self,
        start_date: str,
        end_date: str,
        symbol: str = MARKET_INDEX_SYMBOL,
    ) -> pl.DataFrame:
        """
        加载市场指数数据（用于计算市场波动率）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbol: 指数代码
        
        Returns:
            包含指数数据的 DataFrame
        """
        logger.info(f"Loading market index data ({symbol}) from {start_date} to {end_date}...")
        
        try:
            query = """
                SELECT symbol, trade_date, open, high, low, close, volume
                FROM index_daily
                WHERE symbol = %s AND trade_date >= %s AND trade_date <= %s
                ORDER BY trade_date
            """
            
            df = self.db.read_sql(query, [symbol, start_date, end_date])
            
            if df.is_empty():
                logger.warning(f"No index data found for {symbol}")
                return pl.DataFrame()
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load index data: {e}")
            return pl.DataFrame()
    
    def load_stock_metadata(self) -> pl.DataFrame:
        """
        加载股票元数据（行业、市值等）
        
        Returns:
            包含股票元数据的 DataFrame
        """
        logger.info("Loading stock metadata...")
        
        try:
            query = """
                SELECT symbol, stock_name, industry_name, market_cap
                FROM stock_metadata
                ORDER BY symbol
            """
            
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.warning("No stock metadata found, using fallback")
                return self._create_fallback_metadata()
            
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load stock metadata: {e}")
            return self._create_fallback_metadata()
    
    def _clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        数据清洗
        
        Args:
            df: 原始数据
        
        Returns:
            清洗后的数据
        """
        try:
            # 填充 NaN 和 Null 值
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'turnover_rate']
            
            for col in numeric_cols:
                if col in df.columns:
                    df = df.with_columns([
                        pl.col(col).cast(pl.Float64, strict=False).alias(col)
                    ])
                    df = df.with_columns([
                        pl.col(col).fill_nan(0).fill_null(0)
                    ])
            
            # 确保价格为正
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df = df.with_columns([
                        pl.when(pl.col(col) <= 0)
                        .then(pl.lit(0.01))
                        .otherwise(pl.col(col))
                        .alias(col)
                    ])
            
            # 确保 high >= low
            if 'high' in df.columns and 'low' in df.columns:
                df = df.with_columns([
                    pl.max_horizontal(['high', 'low']).alias('high'),
                    pl.min_horizontal(['high', 'low']).alias('low')
                ])
            
            return df
            
        except Exception as e:
            logger.error(f"_clean_data FAILED: {e}")
            raise
    
    def _validate_data(self, df: pl.DataFrame) -> bool:
        """
        数据验证
        
        Args:
            df: 数据
        
        Returns:
            是否验证通过
        """
        try:
            # 检查必需列
            required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            
            # 检查数据范围
            if df['close'].min() <= 0:
                logger.warning("Found non-positive close prices")
            
            if df['volume'].min() < 0:
                logger.warning("Found negative volume")
            
            # 检查日期范围
            dates = df['trade_date'].unique().sort()
            if len(dates) < 10:
                logger.warning(f"Only {len(dates)} trading days found")
            
            return True
            
        except Exception as e:
            logger.error(f"_validate_data FAILED: {e}")
            return False
    
    def _generate_fallback_data(
        self,
        start_date: str,
        end_date: str,
        symbols: List[str],
    ) -> pl.DataFrame:
        """
        生成备用模拟数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            symbols: 股票代码列表
        
        Returns:
            模拟数据 DataFrame
        """
        logger.info("Generating fallback simulated data...")
        
        import random
        random.seed(42)
        np.random.seed(42)
        
        # 生成交易日期
        dates = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        while current <= end:
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        n_days = len(dates)
        all_data = []
        
        for symbol in symbols:
            initial_price = random.uniform(50, 200)
            prices = [initial_price]
            
            for _ in range(n_days - 1):
                ret = random.gauss(0.0005, 0.02)
                new_price = max(5, prices[-1] * (1 + ret))
                prices.append(new_price)
            
            opens = []
            highs = []
            lows = []
            for i, (o, c) in enumerate(zip([initial_price] + prices[:-1], prices)):
                opens.append(o * random.uniform(0.99, 1.01))
                highs.append(max(o, c) * random.uniform(1.0, 1.02))
                lows.append(min(o, c) * random.uniform(0.98, 1.0))
            
            volumes = [random.randint(100000, 5000000) for _ in dates]
            turnover_rates = [random.uniform(0.01, 0.08) for _ in dates]
            
            data = {
                'symbol': [symbol] * n_days,
                'trade_date': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': prices,
                'volume': volumes,
                'turnover_rate': turnover_rates,
            }
            all_data.append(pl.DataFrame(data))
        
        df = pl.concat(all_data)
        logger.info(f"Generated {len(df)} records with {df['symbol'].n_unique()} stocks")
        
        return df
    
    def _create_fallback_metadata(self) -> pl.DataFrame:
        """
        创建备用元数据
        
        Returns:
            元数据 DataFrame
        """
        data = []
        for stock in FALLBACK_STOCKS:
            data.append({
                'symbol': stock['symbol'],
                'stock_name': stock['name'],
                'industry_name': 'General',
                'market_cap': 100000000000,  # 默认 1000 亿
            })
        return pl.DataFrame(data)
    
    def load_all_data(self) -> Dict[str, pl.DataFrame]:
        """
        加载所有数据
        
        Returns:
            包含所有数据的字典
        """
        logger.info("Loading all data...")
        
        result = {
            'price_data': self.load_stock_data(
                self.config.start_date,
                self.config.end_date,
                use_fallback=True
            ),
            'industry_data': self.load_industry_data(
                self.config.start_date,
                self.config.end_date
            ),
            'index_data': self.load_market_index_data(
                self.config.start_date,
                self.config.end_date
            ),
            'metadata': self.load_stock_metadata()
        }
        
        logger.info(f"All data loaded: {len(result['price_data'])} price records")
        return result
    
    def clear_cache(self):
        """清除数据缓存"""
        self._data_cache.clear()
        logger.info("Data cache cleared")
