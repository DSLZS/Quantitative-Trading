# src/data/database_manager.py
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from config.settings import MYSQL_CONFIG, DB_POOL_CONFIG
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import logging
import re
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理类（使用连接池）"""
    
    # 类级别的连接池
    _engine = None
    
    def __init__(self, config=None):
        self.config = config or MYSQL_CONFIG
        self.pool_config = DB_POOL_CONFIG
        
        # 定义表结构所需的列顺序
        self.stock_daily_columns = [
            'symbol', 'market', 'trade_date', 'open', 'high', 'low', 'close',
            'volume', 'amount', 'adjust_factor_back', 'adjust_factor_forward',
            'change', 'pct_change', 'turnover_rate'
        ]
        
        # 定义市场代码映射 - 根据您的数据库实际情况调整
        self.market_mapping = {
            'SH': 'SSE',      # 上海证券交易所 - 您数据库中存储的是 SSE
            'SZ': 'SZSE',     # 深圳证券交易所
            'BJ': 'BSE',      # 北京证券交易所
            'HK': 'HKEX',     # 香港交易所
            'US': 'NYSE',     # 纽约交易所
            'SSE': 'SSE',     # 直接映射，防止大小写问题
            'sse': 'SSE',     # 小写映射到大写
        }
        
        # 反向映射
        self.reverse_market_mapping = {v: k for k, v in self.market_mapping.items()}
        
        # 初始化连接池
        self._init_engine()
    
    @classmethod
    def _init_engine(cls):
        """初始化数据库连接池（单例模式）"""
        if cls._engine is None:
            config = MYSQL_CONFIG
            pool_config = DB_POOL_CONFIG
            
            # 构建数据库URL
            db_url = f"mysql+pymysql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset={config.get('charset', 'utf8mb4')}"
            
            # 创建连接池引擎
            cls._engine = create_engine(
                db_url,
                poolclass=QueuePool,
                pool_size=pool_config['pool_size'],
                max_overflow=pool_config['max_overflow'],
                pool_recycle=pool_config['pool_recycle'],
                pool_timeout=pool_config['pool_timeout'],
                echo=pool_config['echo']
            )
            logger.info(f"数据库连接池初始化完成，池大小: {pool_config['pool_size']}")
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        connection = None
        try:
            connection = self._engine.connect()
            yield connection
        except SQLAlchemyError as e:
            logger.error(f"数据库连接错误: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def parse_stock_code(self, full_code: str) -> Tuple[str, str]:
        """
        解析股票完整代码，返回代码和交易所
        
        Args:
            full_code: 完整股票代码，如 '510300.SH'
            
        Returns:
            (symbol, market) 元组，如 ('510300', 'SSE')
        """
        if '.' in full_code:
            symbol_part, exchange_part = full_code.split('.')
            
            # 清理空格和标准化
            symbol_part = symbol_part.strip()
            exchange_part = exchange_part.strip().upper()  # 转换为大写
            
            # 获取市场代码
            if exchange_part in self.market_mapping:
                market = self.market_mapping[exchange_part]
            else:
                # 尝试直接使用，不进行映射转换
                market = exchange_part
                logger.debug(f"直接使用交易所代码: {exchange_part} -> {market}")
            
            logger.debug(f"解析股票代码: {full_code} -> symbol={symbol_part}, market={market}")
            return symbol_part, market
        else:
            # 如果没有点号，尝试从常见格式中解析
            match = re.match(r'([0-9]{6})([A-Z]{2,4})?', full_code.upper())
            if match:
                symbol_part = match.group(1)
                if match.group(2):
                    exchange_part = match.group(2)
                    if exchange_part in self.market_mapping:
                        market = self.market_mapping[exchange_part]
                    else:
                        market = exchange_part
                else:
                    # 如果没有交易所代码，需要根据symbol判断
                    # 这里简单判断：60开头为上证，00开头为深证，30开头为创业板
                    if symbol_part.startswith('60'):
                        market = 'SSE'
                    elif symbol_part.startswith('00') or symbol_part.startswith('30'):
                        market = 'SZSE'
                    elif symbol_part.startswith('68'):
                        market = 'SSE'  # 科创板
                    elif symbol_part.startswith('83') or symbol_part.startswith('87'):
                        market = 'BSE'  # 北交所
                    else:
                        market = None
                logger.debug(f"解析无点号股票代码: {full_code} -> symbol={symbol_part}, market={market}")
                return symbol_part, market
            else:
                logger.warning(f"无法解析股票代码格式: {full_code}, 将使用纯代码作为symbol")
                return full_code, None
    
    def format_stock_code(self, symbol: str, market: str) -> str:
        """
        将代码和市场组合成完整代码
        
        Args:
            symbol: 股票代码，如 '510300'
            market: 市场代码，如 'SSE'
            
        Returns:
            完整股票代码，如 '510300.SH'
        """
        if not market:
            return symbol
        
        market = str(market).strip().upper()  # 确保大写
        
        # 查找反向映射
        for key, value in self.reverse_market_mapping.items():
            if key == market:
                return f"{symbol}.{value}"
        
        # 如果没有找到，尝试常见映射
        common_mappings = {
            'SSE': 'SH',
            'SZSE': 'SZ',
            'BSE': 'BJ',
            'HKEX': 'HK',
            'NYSE': 'US',
            'NASDAQ': 'US'
        }
        
        if market in common_mappings:
            return f"{symbol}.{common_mappings[market]}"
        else:
            # 使用原始市场代码
            return f"{symbol}.{market}"
    
    def execute_query(self, query, params=None, connection=None):
        """执行查询"""
        close_connection = False
        if connection is None:
            connection = self._engine.connect()
            close_connection = True
        
        try:
            result = connection.execute(text(query), params or {})
            return result.fetchall()
        except SQLAlchemyError as e:
            logger.error(f"查询执行失败: {e}")
            return None
        finally:
            if close_connection and connection:
                connection.close()
    
    def execute_many(self, query, data_list, connection=None):
        """批量执行插入/更新"""
        close_connection = False
        if connection is None:
            connection = self._engine.connect()
            close_connection = True
        
        try:
            # 使用事务
            with connection.begin():
                result = connection.execute(text(query), data_list)
                affected_rows = result.rowcount
                logger.info(f"批量操作成功，影响行数: {affected_rows}")
                return affected_rows
        except SQLAlchemyError as e:
            logger.error(f"批量操作失败: {e}")
            return 0
        finally:
            if close_connection and connection:
                connection.close()
    
    def insert_stock_daily(self, data_df, replace=False):
        """
        插入股票日线数据到stock_daily表
        
        Args:
            data_df: DataFrame，包含股票日线数据
            replace: 是否使用REPLACE代替INSERT（处理主键冲突）
        """
        if data_df.empty:
            logger.warning("数据为空，跳过插入")
            return 0
        
        # 检查必要的列是否存在
        required_columns = ['symbol', 'market', 'trade_date']
        missing_columns = [col for col in required_columns if col not in data_df.columns]
        
        if missing_columns:
            logger.error(f"DataFrame中缺少必需的列: {missing_columns}")
            logger.info(f"DataFrame中的列: {list(data_df.columns)}")
            return 0
        
        # 打印数据摘要
        logger.info(f"准备插入 {len(data_df)} 行数据")
        logger.info(f"数据日期范围: {data_df['trade_date'].min()} 到 {data_df['trade_date'].max()}")
        
        # 准备数据
        data_list = []
        for _, row in data_df.iterrows():
            try:
                # 处理每一行数据
                data_tuple = {
                    'symbol': str(row['symbol']).strip() if pd.notna(row.get('symbol')) else '',
                    'market': str(row['market']).strip().upper() if pd.notna(row.get('market')) else '',
                    'trade_date': pd.to_datetime(row['trade_date']).strftime('%Y-%m-%d') if pd.notna(row.get('trade_date')) else None,
                    'open': float(row.get('open')) if pd.notna(row.get('open')) else None,
                    'high': float(row.get('high')) if pd.notna(row.get('high')) else None,
                    'low': float(row.get('low')) if pd.notna(row.get('low')) else None,
                    'close': float(row.get('close')) if pd.notna(row.get('close')) else None,
                    'volume': int(row.get('volume')) if pd.notna(row.get('volume')) else None,
                    'amount': float(row.get('amount')) if pd.notna(row.get('amount')) else None,
                    'adjust_factor_back': float(row.get('adjust_factor_back', 1.0)) if pd.notna(row.get('adjust_factor_back', 1.0)) else 1.0,
                    'adjust_factor_forward': float(row.get('adjust_factor_forward', 1.0)) if pd.notna(row.get('adjust_factor_forward', 1.0)) else 1.0,
                    'change': float(row.get('change')) if pd.notna(row.get('change')) else None,
                    'pct_change': float(row.get('pct_change')) if pd.notna(row.get('pct_change')) else None,
                    'turnover_rate': float(row.get('turnover_rate')) if pd.notna(row.get('turnover_rate')) else None
                }
                data_list.append(data_tuple)
            except Exception as e:
                logger.warning(f"处理行数据时出错（跳过该行）: {e}")
                continue
        
        if not data_list:
            logger.error("数据准备失败，没有有效的数据行")
            return 0
        
        logger.info(f"成功准备 {len(data_list)} 行数据用于插入")
        
        # SQL语句 - 匹配表结构
        if replace:
            sql = """
            REPLACE INTO stock_daily 
            (symbol, market, trade_date, `open`, high, low, `close`, volume, amount, 
             adjust_factor_back, adjust_factor_forward, `change`, pct_change, turnover_rate)
            VALUES (:symbol, :market, :trade_date, :open, :high, :low, :close, :volume, :amount, 
                    :adjust_factor_back, :adjust_factor_forward, :change, :pct_change, :turnover_rate)
            """
        else:
            sql = """
            INSERT IGNORE INTO stock_daily 
            (symbol, market, trade_date, `open`, high, low, `close`, volume, amount, 
             adjust_factor_back, adjust_factor_forward, `change`, pct_change, turnover_rate)
            VALUES (:symbol, :market, :trade_date, :open, :high, :low, :close, :volume, :amount, 
                    :adjust_factor_back, :adjust_factor_forward, :change, :pct_change, :turnover_rate)
            """
        
        # 使用连接池执行批量操作
        with self.get_connection() as conn:
            affected_rows = self.execute_many(sql, data_list, conn)
        
        logger.info(f"数据库操作完成，影响行数: {affected_rows}")
        return affected_rows
    
    def fetch_stock_data(self, symbol: str, start_date: str = None, 
                        end_date: str = None, fields: List[str] = None,
                        use_adj: str = 'forward') -> pd.DataFrame:
        """
        从stock_daily表获取股票日线数据
        
        args:
            symbol: 股票代码，如 '510300.SH' 或纯代码 '510300' 510300.SH
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            fields: 需要获取的字段列表，如果为None则获取所有字段
            use_adj: 复权类型 'forward' (前复权), 'back' (后复权), 'none' (不复权)
            
        Returns:
            DataFrame 包含股票数据，按trade_date升序排列
            返回的DataFrame中会包含完整代码列 'full_symbol'
        """
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                from datetime import timedelta
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # 确定价格列映射
            price_column_map = {
                'forward': 'close_adj_forward',
                'back': 'close_adj_back',
                'none': 'close'
            }
            
            if use_adj not in price_column_map:
                logger.warning(f"未知的复权类型: {use_adj}，默认使用前复权")
                use_adj = 'forward'
            
            price_column = price_column_map[use_adj]
            
            # fields为要获取的字段 为空则默认获取所有字段
            if fields is None:
                fields = [
                    'symbol', 'market', 'trade_date',
                    'open', 'high', 'low', 'close',
                    'volume', 'amount', 'pct_change', 'turnover_rate',
                    'adjust_factor_back', 'adjust_factor_forward',
                    'open_adj_forward', 'high_adj_forward', 'low_adj_forward', 'close_adj_forward',
                    'open_adj_back', 'high_adj_back', 'low_adj_back', 'close_adj_back'
                ]
            
            # 确保必要的字段存在
            required_fields = ['symbol', 'market', 'trade_date', price_column]
            for field in required_fields:
                if field not in fields:
                    fields.append(field)
            
            # 构建SQL查询
            fields_str = ', '.join([f'`{f}`' for f in fields])
            
            # 构建WHERE条件
            where_conditions = []
            params = {}

            # 解析股票代码
            symbol_code, market_code = symbol.split('.')[0], symbol.split('.')[1]
            
            where_conditions.append("symbol = :symbol_code AND market = :market_code")
            params['symbol_code'] = symbol_code
            params['market_code'] = market_code
            
            # 日期条件
            where_conditions.append("trade_date BETWEEN :start_date AND :end_date")
            params['start_date'] = start_date
            params['end_date'] = end_date
            
            # 完整WHERE子句
            where_clause = " AND ".join(where_conditions)
            
            # 构建完整查询
            query = f"""
            SELECT {fields_str} 
            FROM stock_daily 
            WHERE {where_clause}
            ORDER BY trade_date ASC
            """
            logger.debug(f"执行查询: {query}")
            
            # 使用连接池执行查询
            with self.get_connection() as conn:
                result = conn.execute(text(query), params)
                results = result.fetchall()
            
            if not results:
                logger.warning(f"未找到 {symbol} 在 {start_date} 到 {end_date} 的数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(results, columns=fields)
            
            # 类型转换
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # 设置price列
            df['price'] = df[price_column]
            
            # 添加完整代码列
            if 'symbol' in df.columns and 'market' in df.columns:
                df['full_symbol'] = df.apply(
                    lambda row: self.format_stock_code(row['symbol'], row['market']), 
                    axis=1
                )
            
            logger.info(f"成功获取 {symbol} 共 {len(df)} 条数据，时间范围: {df['trade_date'].min()} 到 {df['trade_date'].max()}")
            logger.info(f"使用{use_adj}复权价格，价格列: {price_column}")
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票数据失败: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_available_symbols(self, full_format: bool = True) -> List[str]:
        """
        获取数据库中所有可用的股票代码
        
        Args:
            full_format: 是否返回完整格式的代码（如 '510300.SH'），默认为True
            
        Returns:
            股票代码列表
        """
        try:
            if full_format:
                query = "SELECT DISTINCT symbol, market FROM stock_daily ORDER BY symbol"
            else:
                query = "SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol"
            
            with self.get_connection() as conn:
                result = conn.execute(text(query))
                results = result.fetchall()
            
            if full_format:
                symbols = []
                for result in results:
                    symbol, market = result
                    full_symbol = self.format_stock_code(str(symbol), str(market))
                    symbols.append(full_symbol)
            else:
                symbols = [str(result[0]) for result in results]
            
            logger.info(f"数据库中共有 {len(symbols)} 个股票代码")
            return symbols
            
        except Exception as e:
            logger.error(f"获取股票代码列表失败: {e}")
            return []
    
    def get_date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        获取某只股票的数据日期范围
        
        Args:
            symbol: 股票代码，如 '510300.SH' 或纯代码 '510300'
            
        Returns:
            (开始日期, 结束日期) 元组
        """
        try:
            # 解析股票代码
            symbol_code, market_code = self.parse_stock_code(symbol)
            
            if market_code:
                query = """
                SELECT MIN(trade_date), MAX(trade_date) 
                FROM stock_daily 
                WHERE symbol = :symbol AND market = :market
                """
                params = {'symbol': symbol_code, 'market': market_code}
            else:
                query = """
                SELECT MIN(trade_date), MAX(trade_date) 
                FROM stock_daily 
                WHERE symbol = :symbol
                """
                params = {'symbol': symbol_code}
            
            with self.get_connection() as conn:
                result = conn.execute(text(query), params)
                result_data = result.fetchone()
            
            if result_data and result_data[0] and result_data[1]:
                start_date = result_data[0].strftime('%Y-%m-%d') if hasattr(result_data[0], 'strftime') else str(result_data[0])
                end_date = result_data[1].strftime('%Y-%m-%d') if hasattr(result_data[1], 'strftime') else str(result_data[1])
                return (start_date, end_date)
            return None, None
            
        except Exception as e:
            logger.error(f"获取日期范围失败: {e}")
            return None, None
    
    def get_latest_trade_date(self, symbol=None, market=None):
        """获取最新交易日期"""
        try:
            if symbol:
                # 解析股票代码
                symbol_code, market_code = self.parse_stock_code(symbol)
                
                if market_code:
                    query = """
                    SELECT MAX(trade_date) FROM stock_daily 
                    WHERE symbol = :symbol AND market = :market
                    """
                    params = {'symbol': symbol_code, 'market': market_code}
                else:
                    query = """
                    SELECT MAX(trade_date) FROM stock_daily 
                    WHERE symbol = :symbol
                    """
                    params = {'symbol': symbol_code}
            else:
                query = "SELECT MAX(trade_date) FROM stock_daily"
                params = {}
            
            with self.get_connection() as conn:
                result = conn.execute(text(query), params)
                result_data = result.fetchone()
            
            if result_data and result_data[0]:
                date_value = result_data[0]
                if hasattr(date_value, 'strftime'):
                    return date_value.strftime('%Y-%m-%d')
                else:
                    return str(date_value)
            return None
        except Exception as e:
            logger.error(f"获取最新交易日期失败: {e}")
            return None
    
    def check_table_exists(self):
        """检查表是否存在"""
        try:
            query = """
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = :database AND table_name = 'stock_daily'
            """
            params = {'database': self.config['database']}
            
            with self.get_connection() as conn:
                result = conn.execute(text(query), params)
                result_data = result.fetchone()
            
            return result_data[0] > 0 if result_data else False
        except Exception as e:
            logger.error(f"检查表是否存在失败: {e}")
            return False
    
    def check_connection(self):
        """检查数据库连接"""
        try:
            with self.get_connection() as conn:
                # 执行一个简单查询来测试连接
                result = conn.execute(text("SELECT 1"))
                test_result = result.fetchone()
                
                if test_result and test_result[0] == 1:
                    logger.info("数据库连接测试成功")
                    return True
            return False
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def __enter__(self):
        # 对于上下文管理器，我们返回连接
        return self._engine.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 注意：这里实际上不会关闭连接，连接由连接池管理
        pass