#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票日线数据自动化拉取与校验脚本
- 使用 Tushare 接口拉取 2018、2020、2022 年股票日线数据
- 存入 MySQL stock_daily 表
- 实现并发控制和增量逻辑
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

import tushare as ts
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)
logger.add("logs/fetch_stock_daily_{time:YYYYMMDD}.log", rotation="10 MB", level="DEBUG")

# 配置
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "quantitative_trading")

# Tushare 积分限制配置 (根据官方文档)
# 基础积分：每分钟请求次数限制
MINUTE_LIMIT_MAP = {
    "basic": 60,      # 基础用户每分钟 60 次
    "vip": 120,       # VIP 用户每分钟 120 次
    "svip": 300,      # SVIP 用户每分钟 300 次
}

# 目标年份
TARGET_YEARS = [2018, 2020, 2022]
MIN_RECORDS_PER_YEAR = 800000  # 每年最少记录数


class TushareRateLimiter:
    """Tushare API 速率限制器"""
    
    def __init__(self, minute_limit: int = 60):
        self.minute_limit = minute_limit
        self.request_times: List[float] = []
        self.lock_count = 0
        self.total_requests = 0
        
    def acquire(self):
        """获取请求许可，必要时等待"""
        current_time = time.time()
        
        # 清理 60 秒前的请求记录
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # 如果达到限制，等待
        if len(self.request_times) >= self.minute_limit:
            oldest_request = min(self.request_times)
            wait_time = 60 - (current_time - oldest_request) + 0.5
            if wait_time > 0:
                logger.debug(f"达到速率限制，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
                self.request_times = [t for t in self.request_times if time.time() - t < 60]
        
        # 记录请求时间
        self.request_times.append(time.time())
        self.total_requests += 1
        
    def handle_rate_limit_error(self):
        """处理速率限制错误"""
        self.lock_count += 1
        logger.warning(f"触发速率限制，第 {self.lock_count} 次，等待 65 秒")
        time.sleep(65)
        self.request_times = []  # 重置请求记录


class StockDailyFetcher:
    """股票日线数据拉取器"""
    
    def __init__(self):
        self.database_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=20,
        )
        
        # 初始化 Tushare
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        
        # 速率限制器
        self.rate_limiter = TushareRateLimiter(minute_limit=60)
        
        # 统计信息
        self.stats = {
            "total_stocks": 0,
            "total_days": 0,
            "inserted": 0,
            "skipped": 0,
            "errors": 0,
        }
        
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
            
    def create_table(self):
        """创建 stock_daily 表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS `stock_daily` (
          `symbol` varchar(20) NOT NULL COMMENT '代码 (如 600519.SH)',
          `trade_date` DATE NOT NULL COMMENT '交易日期 (YYYY-MM-DD)',
          `open` decimal(18,4) DEFAULT NULL,
          `high` decimal(18,4) DEFAULT NULL,
          `low` decimal(18,4) DEFAULT NULL,
          `close` decimal(18,4) DEFAULT NULL,
          `volume` double DEFAULT NULL COMMENT '成交量',
          `amount` double DEFAULT NULL COMMENT '成交额',
          `adj_factor` decimal(18,6) DEFAULT '1.000000' COMMENT '复权因子',
          `turnover_rate` decimal(18,4) DEFAULT NULL COMMENT '换手率',
          `pre_close` decimal(18,4) DEFAULT NULL,
          `change` decimal(18,4) DEFAULT NULL,
          `pct_chg` decimal(18,4) DEFAULT NULL,
          PRIMARY KEY (`symbol`,`trade_date`),
          KEY `idx_date` (`trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
        """
        with self.get_connection() as conn:
            conn.execute(text(create_table_sql))
            logger.info("stock_daily 表创建成功")
            
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        logger.info("正在获取股票列表...")
        self.rate_limiter.acquire()
        
        try:
            # 获取所有正常上市的股票
            df = self.pro.stock_basic(
                exchange='',
                list_status='L'
            )
            # 打印列名以便调试
            logger.debug(f"stock_basic 返回列：{df.columns.tolist()}")
            # 使用 ts_code 字段
            if 'ts_code' in df.columns:
                symbols = df['ts_code'].tolist()
            elif 'symbol' in df.columns:
                symbols = df['symbol'].tolist()
            else:
                logger.error(f"未知的列名：{df.columns.tolist()}")
                return []
            logger.info(f"获取到 {len(symbols)} 只股票")
            return symbols
        except Exception as e:
            logger.error(f"获取股票列表失败：{e}")
            return []
            
    def get_trade_calendar(self, year: int) -> List[str]:
        """获取指定年份的交易日历"""
        logger.debug(f"正在获取 {year} 年交易日历...")
        self.rate_limiter.acquire()
        
        try:
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            df = self.pro.trade_cal(
                exchange='SSE',
                start_date=start_date,
                end_date=end_date,
                is_open='1'
            )
            trade_dates = df['cal_date'].tolist()
            logger.debug(f"{year} 年共有 {len(trade_dates)} 个交易日")
            return trade_dates
        except Exception as e:
            logger.error(f"获取 {year} 年交易日历失败：{e}")
            return []
            
    def check_existing_data(self, symbol: str, trade_date: str) -> bool:
        """检查指定股票和日期是否已存在数据"""
        try:
            query = text("""
                SELECT COUNT(*) FROM stock_daily 
                WHERE symbol = :symbol AND trade_date = :trade_date
            """)
            with self.get_connection() as conn:
                result = conn.execute(query, {"symbol": symbol, "trade_date": trade_date})
                count = result.scalar()
                return count > 0
        except Exception as e:
            logger.debug(f"检查数据存在性失败：{e}")
            return False
            
    def fetch_daily_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """拉取指定股票和时间范围的日线数据"""
        self.rate_limiter.acquire()
        
        try:
            df = self.pro.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            logger.error(f"拉取 {symbol} 数据失败：{e}")
            # 遇到速率限制错误
            if "权限" in str(e) or "frequency" in str(e).lower() or "limit" in str(e).lower():
                self.rate_limiter.handle_rate_limit_error()
            return None
            
    def process_and_insert(self, df: pd.DataFrame, symbol: str) -> Tuple[int, int]:
        """处理并插入数据，返回 (插入数，跳过数)"""
        if df is None or df.empty:
            return 0, 0
            
        # 数据转换
        df = df.copy()
        df['symbol'] = symbol
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        
        # 重命名列
        df = df.rename(columns={
            'ts_code': 'symbol',
            'cal_date': 'trade_date',
            'vol': 'volume',
            'turnover_rate': 'turnover_rate',
        })
        
        # 选择需要的列
        columns = [
            'symbol', 'trade_date', 'open', 'high', 'low', 'close',
            'volume', 'amount', 'adj_factor', 'turnover_rate',
            'pre_close', 'change', 'pct_chg'
        ]
        
        # 确保所有列都存在
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
                
        df = df[columns]
        
        # 过滤空值
        df = df.dropna(subset=['trade_date', 'close'])
        
        if df.empty:
            return 0, 0
            
        # 批量插入
        inserted = 0
        skipped = 0
        
        with self.get_connection() as conn:
            for _, row in df.iterrows():
                trade_date = row['trade_date']
                if isinstance(trade_date, datetime):
                    trade_date = trade_date.date()
                    
                # 检查是否已存在
                if self.check_existing_data(symbol, str(trade_date)):
                    skipped += 1
                    continue
                    
                # 插入数据 (注意：change 是 MySQL 保留字，需要用反引号括起来)
                insert_sql = text("""
                    INSERT INTO stock_daily (
                        symbol, trade_date, open, high, low, close,
                        volume, amount, adj_factor, turnover_rate,
                        pre_close, `change`, pct_chg
                    ) VALUES (
                        :symbol, :trade_date, :open, :high, :low, :close,
                        :volume, :amount, :adj_factor, :turnover_rate,
                        :pre_close, :change, :pct_chg
                    )
                """)
                
                try:
                    conn.execute(insert_sql, {
                        'symbol': symbol,
                        'trade_date': trade_date,
                        'open': float(row['open']) if pd.notna(row['open']) else None,
                        'high': float(row['high']) if pd.notna(row['high']) else None,
                        'low': float(row['low']) if pd.notna(row['low']) else None,
                        'close': float(row['close']) if pd.notna(row['close']) else None,
                        'volume': float(row['volume']) if pd.notna(row['volume']) else None,
                        'amount': float(row['amount']) if pd.notna(row['amount']) else None,
                        'adj_factor': float(row['adj_factor']) if pd.notna(row['adj_factor']) else 1.0,
                        'turnover_rate': float(row['turnover_rate']) if pd.notna(row['turnover_rate']) else None,
                        'pre_close': float(row['pre_close']) if pd.notna(row['pre_close']) else None,
                        'change': float(row['change']) if pd.notna(row['change']) else None,
                        'pct_chg': float(row['pct_chg']) if pd.notna(row['pct_chg']) else None,
                    })
                    inserted += 1
                except SQLAlchemyError as e:
                    if "Duplicate" in str(e) or "PRIMARY" in str(e):
                        skipped += 1
                    else:
                        logger.error(f"插入数据失败：{e}")
                        self.stats["errors"] += 1
                        
            conn.commit()
            
        return inserted, skipped
        
    def fetch_year_data(self, year: int, all_symbols: List[str], trade_dates: List[str]):
        """拉取指定年份的数据"""
        logger.info(f"========== 开始拉取 {year} 年数据 ==========")
        
        total_stocks = len(all_symbols)
        total_days = len(trade_dates)
        
        for idx, symbol in enumerate(all_symbols):
            progress = (idx + 1) / total_stocks * 100
            
            # 拉取该股票全年的数据
            df = self.fetch_daily_data(
                symbol,
                f"{year}0101",
                f"{year}1231"
            )
            
            if df is not None and not df.empty:
                inserted, skipped = self.process_and_insert(df, symbol)
                self.stats["inserted"] += inserted
                self.stats["skipped"] += skipped
                
                # 每 100 只股票打印进度
                if (idx + 1) % 100 == 0 or idx == total_stocks - 1:
                    logger.info(
                        f"正在处理 {symbol}，已完成 {progress:.1f}% | "
                        f"累计插入：{self.stats['inserted']:,} | "
                        f"累计跳过：{self.stats['skipped']:,} | "
                        f"错误：{self.stats['errors']}"
                    )
                    
            # 随机延迟，避免触发限制
            if (idx + 1) % 10 == 0:
                time.sleep(random.uniform(0.5, 1.5))
                
        logger.info(f"========== {year} 年数据拉取完成 ==========")
        
    def verify_data(self):
        """验证数据"""
        logger.info("\n========== 数据验证 ==========")
        
        # 按年份统计
        verify_sql = """
            SELECT YEAR(trade_date) as year, COUNT(*) as count 
            FROM stock_daily 
            GROUP BY YEAR(trade_date) 
            ORDER BY year
        """
        
        with self.get_connection() as conn:
            result = conn.execute(text(verify_sql))
            rows = result.fetchall()
            
            logger.info("\n按年份统计数据量:")
            for year, count in rows:
                status = "✓" if count >= MIN_RECORDS_PER_YEAR else "✗"
                logger.info(f"  {year} 年：{count:,} 条 {status}")
                
        # 检查 adj_factor 非空
        adj_check_sql = """
            SELECT COUNT(*) FROM stock_daily WHERE adj_factor IS NULL OR adj_factor = 0
        """
        with self.get_connection() as conn:
            result = conn.execute(text(adj_check_sql))
            null_count = result.scalar()
            status = "✓" if null_count == 0 else "✗"
            logger.info(f"\nadj_factor 为空或 0 的记录：{null_count} {status}")
            
        # 检查 close 价格为负
        negative_close_sql = """
            SELECT COUNT(*) FROM stock_daily WHERE close < 0
        """
        with self.get_connection() as conn:
            result = conn.execute(text(negative_close_sql))
            negative_count = result.scalar()
            status = "✓" if negative_count == 0 else "✗"
            logger.info(f"close 价格为负的记录：{negative_count} {status}")
            
        # 随机抽取 3 天检查
        sample_sql = """
            SELECT DISTINCT trade_date FROM stock_daily 
            ORDER BY RAND() LIMIT 3
        """
        with self.get_connection() as conn:
            result = conn.execute(text(sample_sql))
            sample_dates = [str(row[0]) for row in result.fetchall()]
            
            logger.info("\n随机抽取 3 天数据详情:")
            for sample_date in sample_dates:
                detail_sql = text("""
                    SELECT symbol, trade_date, close, adj_factor 
                    FROM stock_daily 
                    WHERE trade_date = :trade_date 
                    LIMIT 5
                """)
                result = conn.execute(detail_sql, {"trade_date": sample_date})
                rows = result.fetchall()
                logger.info(f"\n  {sample_date}:")
                for row in rows:
                    logger.info(f"    {row[0]}: close={row[2]}, adj_factor={row[3]}")
                    
    def run(self):
        """主运行函数"""
        logger.info("========== 股票日线数据拉取任务启动 ==========")
        logger.info(f"目标年份：{TARGET_YEARS}")
        logger.info(f"最小记录数/年：{MIN_RECORDS_PER_YEAR:,}")
        
        # 创建表
        self.create_table()
        
        # 获取股票列表
        all_symbols = self.get_stock_list()
        if not all_symbols:
            logger.error("未能获取股票列表，退出")
            return
            
        self.stats["total_stocks"] = len(all_symbols)
        
        # 拉取每年数据
        for year in TARGET_YEARS:
            trade_dates = self.get_trade_calendar(year)
            if not trade_dates:
                logger.warning(f"{year} 年无交易日历，跳过")
                continue
                
            self.stats["total_days"] = len(trade_dates)
            self.fetch_year_data(year, all_symbols, trade_dates)
            
        # 验证数据
        self.verify_data()
        
        # 输出统计
        logger.info("\n========== 任务完成统计 ==========")
        logger.info(f"总股票数：{self.stats['total_stocks']}")
        logger.info(f"总插入：{self.stats['inserted']:,}")
        logger.info(f"总跳过：{self.stats['skipped']:,}")
        logger.info(f"总错误：{self.stats['errors']}")
        

def main():
    """主函数"""
    try:
        fetcher = StockDailyFetcher()
        fetcher.run()
    except KeyboardInterrupt:
        logger.warning("\n用户中断任务")
        sys.exit(0)
    except Exception as e:
        logger.error(f"任务执行失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()