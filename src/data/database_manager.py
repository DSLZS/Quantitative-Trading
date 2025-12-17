# src/data/database_manager.py
import pymysql
from pymysql import Error
from config.settings import MYSQL_CONFIG
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
import logging
import re
from decimal import Decimal

logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理类"""
    
    def __init__(self, config=None):
        self.config = config or MYSQL_CONFIG
        self.connection = None
        
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
            # 添加更多映射
            'SSE': 'SSE',     # 直接映射，防止大小写问题
            'sse': 'SSE',     # 小写映射到大写
        }
        
        # 反向映射
        self.reverse_market_mapping = {v: k for k, v in self.market_mapping.items()}
    
    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(**self.config)
            logger.info(f"成功连接到数据库: {self.config['database']}")
            return True
        except Error as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    def disconnect(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")
    
    def execute_query(self, query, params=None):
        """执行查询"""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params or ())
                return cursor.fetchall()
        except Error as e:
            logger.error(f"查询执行失败: {e}")
            return None
    
    def execute_many(self, query, data_list):
        """批量执行插入/更新"""
        try:
            with self.connection.cursor() as cursor:
                cursor.executemany(query, data_list)
            self.connection.commit()
            affected_rows = cursor.rowcount
            logger.info(f"批量操作成功，影响行数: {affected_rows}")
            return affected_rows
        except Error as e:
            self.connection.rollback()
            logger.error(f"批量操作失败: {e}")
            return 0
    
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
    
    def convert_decimal_to_float(self, value):
        """将Decimal类型转换为float"""
        if isinstance(value, Decimal):
            return float(value)
        elif isinstance(value, (list, tuple)):
            return [self.convert_decimal_to_float(v) for v in value]
        elif isinstance(value, dict):
            return {k: self.convert_decimal_to_float(v) for k, v in value.items()}
        else:
            return value
    
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
        logger.info(f"数据列: {list(data_df.columns)}")
        logger.info(f"数据日期范围: {data_df['trade_date'].min()} 到 {data_df['trade_date'].max()}")
        logger.info(f"股票列表: {data_df['symbol'].unique()[:5]}")  # 只显示前5只股票
        
        # 准备数据
        data_list = []
        for _, row in data_df.iterrows():
            try:
                # 处理每一行数据
                data_tuple = (
                    str(row['symbol']).strip() if pd.notna(row.get('symbol')) else '',
                    str(row['market']).strip().upper() if pd.notna(row.get('market')) else '',  # 市场代码大写
                    pd.to_datetime(row['trade_date']).strftime('%Y-%m-%d') if pd.notna(row.get('trade_date')) else None,
                    float(row.get('open')) if pd.notna(row.get('open')) else None,
                    float(row.get('high')) if pd.notna(row.get('high')) else None,
                    float(row.get('low')) if pd.notna(row.get('low')) else None,
                    float(row.get('close')) if pd.notna(row.get('close')) else None,
                    int(row.get('volume')) if pd.notna(row.get('volume')) else None,
                    float(row.get('amount')) if pd.notna(row.get('amount')) else None,
                    float(row.get('adjust_factor_back', 1.0)) if pd.notna(row.get('adjust_factor_back', 1.0)) else 1.0,
                    float(row.get('adjust_factor_forward', 1.0)) if pd.notna(row.get('adjust_factor_forward', 1.0)) else 1.0,
                    float(row.get('change')) if pd.notna(row.get('change')) else None,
                    float(row.get('pct_change')) if pd.notna(row.get('pct_change')) else None,
                    float(row.get('turnover_rate')) if pd.notna(row.get('turnover_rate')) else None
                )
                data_list.append(data_tuple)
            except Exception as e:
                logger.warning(f"处理行数据时出错（跳过该行）: {e}")
                continue
        
        if not data_list:
            logger.error("数据准备失败，没有有效的数据行")
            return 0
        
        logger.info(f"成功准备 {len(data_list)} 行数据用于插入")
        if data_list:
            logger.debug(f"第一行数据示例: {data_list[0]}")
        
        # SQL语句 - 匹配表结构
        if replace:
            sql = """
            REPLACE INTO stock_daily 
            (symbol, market, trade_date, `open`, high, low, `close`, volume, amount, 
             adjust_factor_back, adjust_factor_forward, `change`, pct_change, turnover_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        else:
            sql = """
            INSERT IGNORE INTO stock_daily 
            (symbol, market, trade_date, `open`, high, low, `close`, volume, amount, 
             adjust_factor_back, adjust_factor_forward, `change`, pct_change, turnover_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        
        affected_rows = self.execute_many(sql, data_list)
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
            if not self.connect():
                logger.error("数据库连接失败")
                return pd.DataFrame()
            
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                # 默认获取最近一年的数据
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
            params = []

            # 简单化逻辑处理，暂时只支持510030.SH的传入
            symbol_code, market_code = symbol.split('.')[0], symbol.split('.')[1]
            
            # # 处理symbol条件
            # if isinstance(symbol, str):
            #     # 解析完整代码
            #     symbol_code, market_code = self.parse_stock_code(symbol)
            #     logger.debug(f"解析结果: symbol={symbol_code}, market={market_code}")
                
            #     if market_code:
            #         # 如果能够解析出市场代码
            #         where_conditions.append("symbol = %s AND market = %s")
            #         params.extend([symbol_code, market_code])
            #     else:
            #         # 如果不能解析出市场代码，只使用symbol
            #         where_conditions.append("symbol = %s")
            #         params.append(symbol_code)
                    
            # elif isinstance(symbol, list):
            #     # 处理多个股票代码
            #     symbol_conditions = []
            #     for s in symbol:
            #         symbol_code, market_code = self.parse_stock_code(s)
            #         if market_code:
            #             symbol_conditions.append("(symbol = %s AND market = %s)")
            #             params.extend([symbol_code, market_code])
            #         else:
            #             symbol_conditions.append("symbol = %s")
            #             params.append(symbol_code)
                
            #     if symbol_conditions:
            #         where_conditions.append(f"({' OR '.join(symbol_conditions)})")
            
            # 日期条件
            where_conditions.append("trade_date BETWEEN %s AND %s")
            params.extend([start_date, end_date])
            
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
            logger.debug(f"查询参数: {params}")
            
            # 执行查询
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
            
            if not results:
                logger.warning(f"未找到 {symbol} 在 {start_date} 到 {end_date} 的数据")
                return pd.DataFrame()
            
            # 将结果中的Decimal转换为float
            results = self.convert_decimal_to_float(results)
            
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
        finally:
            self.disconnect()
    
    def get_available_symbols(self, full_format: bool = True) -> List[str]:
        """
        获取数据库中所有可用的股票代码
        
        Args:
            full_format: 是否返回完整格式的代码（如 '510300.SH'），默认为True
            
        Returns:
            股票代码列表
        """
        try:
            if not self.connect():
                return []
            
            if full_format:
                query = "SELECT DISTINCT symbol, market FROM stock_daily ORDER BY symbol"
            else:
                query = "SELECT DISTINCT symbol FROM stock_daily ORDER BY symbol"
            
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
            
            # 转换Decimal类型
            results = self.convert_decimal_to_float(results)
            
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
        finally:
            self.disconnect()
    
    def get_date_range(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """
        获取某只股票的数据日期范围
        
        Args:
            symbol: 股票代码，如 '510300.SH' 或纯代码 '510300'
            
        Returns:
            (开始日期, 结束日期) 元组
        """
        try:
            if not self.connect():
                return None, None
            
            # 解析股票代码
            symbol_code, market_code = self.parse_stock_code(symbol)
            
            if market_code:
                query = """
                SELECT MIN(trade_date), MAX(trade_date) 
                FROM stock_daily 
                WHERE symbol = %s AND market = %s
                """
                params = (symbol_code, market_code)
            else:
                query = """
                SELECT MIN(trade_date), MAX(trade_date) 
                FROM stock_daily 
                WHERE symbol = %s
                """
                params = (symbol_code,)
            
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
            
            # 转换Decimal类型
            result = self.convert_decimal_to_float(result)
            
            if result and result[0] and result[1]:
                start_date = result[0].strftime('%Y-%m-%d') if hasattr(result[0], 'strftime') else str(result[0])
                end_date = result[1].strftime('%Y-%m-%d') if hasattr(result[1], 'strftime') else str(result[1])
                return (start_date, end_date)
            return None, None
            
        except Exception as e:
            logger.error(f"获取日期范围失败: {e}")
            return None, None
        finally:
            self.disconnect()
    
    def get_latest_trade_date(self, symbol=None, market=None):
        """获取最新交易日期"""
        try:
            if not self.connect():
                return None
            
            if symbol:
                # 解析股票代码
                symbol_code, market_code = self.parse_stock_code(symbol)
                
                if market_code:
                    query = """
                    SELECT MAX(trade_date) FROM stock_daily 
                    WHERE symbol = %s AND market = %s
                    """
                    params = (symbol_code, market_code)
                else:
                    query = """
                    SELECT MAX(trade_date) FROM stock_daily 
                    WHERE symbol = %s
                    """
                    params = (symbol_code,)
            else:
                query = "SELECT MAX(trade_date) FROM stock_daily"
                params = None
            
            result = self.execute_query(query, params)
            
            # 转换Decimal类型
            result = self.convert_decimal_to_float(result)
            
            if result and result[0][0]:
                date_value = result[0][0]
                if hasattr(date_value, 'strftime'):
                    return date_value.strftime('%Y-%m-%d')
                else:
                    return str(date_value)
            return None
        except Exception as e:
            logger.error(f"获取最新交易日期失败: {e}")
            return None
        finally:
            self.disconnect()
    
    def check_table_exists(self):
        """检查表是否存在"""
        try:
            if not self.connect():
                return False
            
            query = """
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = 'stock_daily'
            """
            result = self.execute_query(query, (self.config['database'],))
            
            # 转换Decimal类型
            result = self.convert_decimal_to_float(result)
            
            return result[0][0] > 0 if result else False
        except Exception as e:
            logger.error(f"检查表是否存在失败: {e}")
            return False
        finally:
            self.disconnect()
    
    def check_connection(self):
        """检查数据库连接"""
        try:
            if self.connect():
                logger.info("数据库连接测试成功")
                self.disconnect()
                return True
            return False
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()