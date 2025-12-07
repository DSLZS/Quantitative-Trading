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
        
        # 准备数据
        data_list = []
        for _, row in data_df.iterrows():
            data_tuple = (
                row['symbol'],
                row['market'],
                row['trade_date'],
                row.get('open'),
                row.get('high'),
                row.get('low'),
                row.get('close'),
                row.get('volume'),
                row.get('amount'),
                row.get('adjust_factor_back', 1.0),
                row.get('adjust_factor_forward', 1.0),
                row.get('change'),
                row.get('pct_change'),
                row.get('turnover_rate')
            )
            data_list.append(data_tuple)
        
        # SQL语句
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
        
        return self.execute_many(sql, data_list)
    
    def get_latest_trade_date(self, symbol=None, market=None):
        """获取最新交易日期"""
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
    
    def check_table_exists(self):
        """检查表是否存在"""
        query = """
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = %s AND table_name = 'stock_daily'
        """
        result = self.execute_query(query, (self.config['database'],))
        return result[0][0] > 0 if result else False
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()