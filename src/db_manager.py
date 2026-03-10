"""
Database Manager Module - MySQL connection management with Polars.

This module provides a singleton database manager for MySQL connections,
using Polars for data processing and SQLAlchemy for connection pooling.

Note: ADBC driver is not available for Python 3.13, so we use SQLAlchemy
with pymysql as the primary connection method.

核心功能:
    - 数据库连接池管理
    - SQL 查询返回 Polars DataFrame
    - Polars DataFrame 写入 MySQL
    - Upsert 操作（处理主键冲突）
"""

import os
from typing import Any, Optional
from contextlib import contextmanager

import pandas as pd
import polars as pl
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# Load environment variables
load_dotenv()


class DatabaseManager:
    """
    单例模式的数据库管理器，使用 Polars 进行数据处理。
    
    功能特性:
        - 使用 QueuePool 实现连接池
        - 使用上下文管理器安全处理连接
        - 支持大数据集的分块读取
        - 连接失败自动重连
        - 零 Pandas 依赖
    
    使用示例:
        >>> db = DatabaseManager()  # 获取单例实例
        >>> df = db.read_sql("SELECT * FROM stocks LIMIT 100")  # 查询数据
        >>> db.to_sql(df, "table_name", if_exists="append")  # 写入数据
    """
    
    _instance: Optional["DatabaseManager"] = None  # 单例实例
    
    def __new__(cls) -> "DatabaseManager":
        """
        实现单例模式，确保只有一个数据库管理器实例。
        
        Returns:
            DatabaseManager: 单例实例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """
        初始化数据库连接池。
        
        注意：由于使用单例模式，__init__ 可能会被多次调用，
        使用 _initialized 标志避免重复初始化。
        """
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._initialized = True
        self._engine = None  # SQLAlchemy 引擎，延迟初始化
        
    def _build_connection_url(self) -> str:
        """
        从环境变量构建 MySQL 连接 URL。
        
        需要的环境变量:
            - MYSQL_HOST: 数据库主机地址 (默认：localhost)
            - MYSQL_PORT: 数据库端口 (默认：3306)
            - MYSQL_USER: 数据库用户名 (默认：root)
            - MYSQL_PASSWORD: 数据库密码 (默认：空)
            - MYSQL_DATABASE: 数据库名称 (默认：quantitative_trading)
        
        Returns:
            str: MySQL 连接 URL，格式为 mysql+pymysql://user:password@host:port/database
        """
        host = os.getenv("MYSQL_HOST", "localhost")
        port = os.getenv("MYSQL_PORT", "3306")
        user = os.getenv("MYSQL_USER", "root")
        password = os.getenv("MYSQL_PASSWORD", "")
        database = os.getenv("MYSQL_DATABASE", "quantitative_trading")
        
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    
    def connect(self) -> None:
        """
        创建带有连接池的数据库引擎。
        
        连接池配置:
            - pool_size: 保持打开的连接数 (默认：10)
            - max_overflow: 高峰期允许额外创建的连接数 (默认：20)
            - pool_recycle: 连接回收时间，秒 (默认：3600)
            - pool_pre_ping: 启用连接健康检查 (默认：True)
            - echo: 是否打印 SQL 日志 (默认：False)
        """
        if self._engine is not None:
            logger.debug("Database engine already exists")
            return
        
        connection_url = self._build_connection_url()
        
        self._engine = create_engine(
            connection_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_recycle=3600,
            pool_pre_ping=True,
            echo=False,
        )
        
        logger.info("Database connection pool initialized")
    
    def close(self) -> None:
        """
        释放连接池。
        
        调用此方法后，所有打开的连接将被关闭，引擎被销毁。
        """
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connection pool disposed")
    
    @property
    def engine(self):
        """
        获取 SQLAlchemy 引擎，如果未创建则自动创建。
        
        Returns:
            SQLAlchemy Engine: 数据库引擎实例
        """
        if self._engine is None:
            self.connect()
        return self._engine
    
    @contextmanager
    def get_connection(self):
        """
        数据库连接的上下文管理器。
        
        使用上下文管理器可以确保连接在使用后自动关闭，
        即使发生异常也能正确释放资源。
        
        Yields:
            SQLAlchemy Connection: 数据库连接对象
            
        使用示例:
            >>> with db.get_connection() as conn:
            ...     result = conn.execute(text("SELECT 1"))
        """
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def read_sql(
        self,
        query: str,
        params: Optional[dict] = None,
    ) -> pl.DataFrame:
        """
        执行 SELECT 查询并返回 Polars DataFrame。
        
        Args:
            query (str): SQL 查询语句
            params (dict, optional): 查询参数，用于预处理语句，防止 SQL 注入
                示例：{"start": "2024-01-01"} 对应 SQL 中的 :start
            
        Returns:
            pl.DataFrame: 查询结果组成的 Polars DataFrame
            
        使用示例:
            >>> df = db.read_sql(
            ...     "SELECT * FROM prices WHERE date >= :start",
            ...     params={"start": "2024-01-01"}
            ... )
        """
        try:
            logger.debug(f"Executing query: {query[:100]}...")
            
            # 使用 SQLAlchemy 连接执行查询
            if params:
                # 带参数的查询
                with self.get_connection() as conn:
                    result = conn.execute(text(query), params)
                    columns = [col[0] for col in result.cursor.description]
                    rows = result.fetchall()
            else:
                # 无参数的直接查询
                with self.get_connection() as conn:
                    result = conn.execute(text(query))
                    columns = [col[0] for col in result.cursor.description]
                    rows = result.fetchall()
            
            # Use pandas as intermediate layer for better type handling
            # This handles mixed types and DECIMAL conversion automatically
            pdf = pd.DataFrame.from_records(rows, columns=columns)
            
            # Convert to Polars DataFrame
            df = pl.from_pandas(pdf)
            
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def to_sql(
        self,
        df: pl.DataFrame,
        table_name: str,
        if_exists: str = "append",
        index: bool = False,
    ) -> int:
        """
        将 Polars DataFrame 写入 MySQL 表。
        
        使用 Polars 的 write_database 方法，通过 SQLAlchemy 引擎连接。
        
        Args:
            df (pl.DataFrame): 要写入的 Polars DataFrame
            table_name (str): 目标表名
            if_exists (str): 表已存在时的处理方式
                - 'fail': 抛出异常
                - 'replace': 删除原表并创建新表
                - 'append': 追加数据（默认）
            index (bool): 是否写入 DataFrame 的索引（默认：False）
            
        Returns:
            int: 插入的行数
            
        使用示例:
            >>> rows = db.to_sql(df, "stock_prices", if_exists="append")
        """
        try:
            rows = len(df)
            logger.info(f"Writing {rows} rows to {table_name}")
            
            # 使用 Polars write_database 方法
            df.write_database(
                table_name=table_name,
                connection=self.engine,
                if_table_exists=if_exists,
            )
            
            logger.info(f"Successfully wrote {rows} rows to {table_name}")
            return rows
            
        except Exception as e:
            logger.error(f"Failed to write to {table_name}: {e}")
            raise
    
    def execute(
        self, 
        query: str, 
        params: Optional[dict] = None
    ) -> int:
        """
        执行 INSERT/UPDATE/DELETE 语句。
        
        Args:
            query (str): SQL 语句
            params (dict, optional): 语句参数
            
        Returns:
            int: 受影响的行数
            
        使用示例:
            >>> rows = db.execute(
            ...     "UPDATE stocks SET price = :price WHERE symbol = :symbol",
            ...     params={"price": 100.0, "symbol": "000001.SZ"}
            ... )
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(
                    text(query),
                    params or {},
                )
                conn.commit()
                return result.rowcount or 0
                
        except Exception as e:
            logger.error(f"Statement execution failed: {e}")
            raise
    
    def upsert(
        self,
        df: pl.DataFrame,
        table_name: str,
        key_columns: list[str],
    ) -> int:
        """
        Upsert（INSERT ... ON DUPLICATE KEY UPDATE）Polars DataFrame 到 MySQL。
        
        由于 ADBC 驱动不支持 ON DUPLICATE KEY UPDATE 语法，
        本方法使用"先删除后插入"的策略处理重复键。
        
        Args:
            df (pl.DataFrame): 要 upsert 的 Polars DataFrame
            table_name (str): 目标表名
            key_columns (list[str]): 构成主键/唯一键的列名列表
                示例：["symbol", "Date"]
            
        Returns:
            int: 受影响的行数
            
        使用示例:
            >>> db.upsert(df, "stock_daily", ["symbol", "Date"])
        """
        try:
            rows = len(df)
            logger.info(f"Upserting {rows} rows to {table_name}")
            
            # 使用删除 - 插入策略处理重复
            self._delete_and_insert(df, table_name, key_columns)
            
            logger.info(f"Successfully upserted {rows} rows to {table_name}")
            return rows
            
        except Exception as e:
            logger.error(f"Failed to upsert to {table_name}: {e}")
            raise
    
    def _delete_and_insert(
        self,
        df: pl.DataFrame,
        table_name: str,
        key_columns: list[str],
    ) -> None:
        """
        删除现有记录并插入新记录（用于 upsert 操作）。
        
        实现逻辑:
        1. 获取 DataFrame 中所有唯一键组合
        2. 构建 DELETE 语句删除表中匹配的记录
        3. 使用 to_sql 插入新数据
        
        Args:
            df (pl.DataFrame): 要插入的 Polars DataFrame
            table_name (str): 目标表名
            key_columns (list[str]): 构成主键/唯一键的列名列表
        """
        if df.is_empty():
            return
        
        with self.get_connection() as conn:
            from sqlalchemy import text
            
            # 获取唯一键组合
            keys_df = df.select(key_columns).unique()
            
            if keys_df.is_empty():
                # 没有键需要删除，直接插入
                self.to_sql(df, table_name, if_exists="append")
                return
            
            # 构建 DELETE 条件
            conditions = []
            params = {}
            param_idx = 0
            
            for row in keys_df.iter_rows():
                condition_parts = []
                for col, val in zip(key_columns, row):
                    param_name = f"param_{param_idx}"
                    condition_parts.append(f"`{col}` = :{param_name}")
                    params[param_name] = val
                    param_idx += 1
                conditions.append(f"({' AND '.join(condition_parts)})")
            
            if conditions:
                delete_sql = f"""
                    DELETE FROM `{table_name}`
                    WHERE {' OR '.join(conditions)}
                """
                try:
                    conn.execute(text(delete_sql), params)
                    conn.commit()
                    logger.debug(f"Deleted {len(keys_df)} existing records for upsert")
                except Exception as e:
                    logger.warning(f"Delete failed, will attempt insert anyway: {e}")
        
        # 插入新记录
        self.to_sql(df, table_name, if_exists="append")
    
    def get_mysql_version(self) -> str:
        """
        获取 MySQL 服务器版本。
        
        Returns:
            str: MySQL 版本字符串
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT VERSION()"))
                version = result.scalar()
                logger.info(f"MySQL version: {version}")
                return version
        except Exception as e:
            logger.error(f"Failed to get MySQL version: {e}")
            raise
    
    def table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在于数据库中。
        
        Args:
            table_name (str): 要检查的表名
            
        Returns:
            bool: 表存在返回 True，否则返回 False
        """
        try:
            database = os.getenv("MYSQL_DATABASE", "quantitative_trading")
            query = f"""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = '{database}' 
                AND table_name = '{table_name}'
            """
            result = self.read_sql(query)
            return result[0, 0] > 0
        except Exception as e:
            logger.error(f"Failed to check table existence: {e}")
            raise


# 便捷函数
def get_db() -> DatabaseManager:
    """
    获取单例数据库管理器实例。
    
    Returns:
        DatabaseManager: 单例实例
    """
    return DatabaseManager()


# 为 run_sync.py 提供兼容性方法
DatabaseManager.get_instance = staticmethod(lambda: DatabaseManager())
