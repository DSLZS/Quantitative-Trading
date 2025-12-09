# src/data/database_manager.py
import pymysql
from pymysql import Error
from config.settings import MYSQL_CONFIG
import pandas as pd
from typing import List, Dict, Optional
import logging

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
                    str(row['market']).strip() if pd.notna(row.get('market')) else '',
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
    
    def get_latest_trade_date(self, symbol=None, market=None):
        """获取最新交易日期"""
        try:
            if symbol and market:
                query = """
                SELECT MAX(trade_date) FROM stock_daily 
                WHERE symbol = %s AND market = %s
                """
                params = (symbol, market)
            else:
                query = "SELECT MAX(trade_date) FROM stock_daily"
                params = None
            
            result = self.execute_query(query, params)
            if result and result[0][0]:
                return result[0][0]
            return None
        except Exception as e:
            logger.error(f"获取最新交易日期失败: {e}")
            return None
    
    def check_table_exists(self):
        """检查表是否存在"""
        try:
            query = """
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = %s AND table_name = 'stock_daily'
            """
            result = self.execute_query(query, (self.config['database'],))
            return result[0][0] > 0 if result else False
        except Exception as e:
            logger.error(f"检查表是否存在失败: {e}")
            return False
    
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