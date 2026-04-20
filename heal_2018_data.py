#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充 2018 年缺失数据
- 检查 2018 年实际存在的股票
- 拉取缺失的股票数据
"""

import os
import sys
import time
from datetime import datetime, date
from typing import Optional, List, Tuple
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

# 配置
TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "quantitative_trading")


class Stock2018Healer:
    """2018 年数据补充器"""
    
    def __init__(self):
        self.database_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # 初始化 Tushare
        ts.set_token(TUSHARE_TOKEN)
        self.pro = ts.pro_api()
        
        # 统计信息
        self.stats = {
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
            
    def get_existing_2018_symbols(self) -> set:
        """获取 2018 年已有的股票数据"""
        logger.info("正在获取 2018 年已有的股票...")
        
        sql = """
            SELECT DISTINCT symbol FROM stock_daily 
            WHERE YEAR(trade_date) = 2018
        """
        with self.get_connection() as conn:
            result = conn.execute(text(sql))
            symbols = {row[0] for row in result.fetchall()}
            
        logger.info(f"2018 年已有 {len(symbols)} 只股票")
        return symbols
        
    def get_stocks_listed_before_2018(self) -> List[str]:
        """获取 2018 年前上市的股票列表"""
        logger.info("正在获取 2018 年前上市的股票...")
        
        try:
            # 获取所有股票列表
            df = self.pro.stock_basic(
                exchange='',
                list_status='L'
            )
            
            # 过滤 2018 年前上市的股票
            df['list_date'] = pd.to_numeric(df['list_date'], errors='coerce')
            df_2018 = df[df['list_date'] < 20180101].copy()
            
            symbols = df_2018['ts_code'].tolist()
            logger.info(f"2018 年前共有 {len(symbols)} 只股票上市")
            return symbols
        except Exception as e:
            logger.error(f"获取股票列表失败：{e}")
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
        try:
            df = self.pro.daily(
                ts_code=symbol,
                start_date=start_date,
                end_date=end_date
            )
            return df
        except Exception as e:
            logger.error(f"拉取 {symbol} 数据失败：{e}")
            return None
            
    def process_and_insert(self, df: pd.DataFrame, symbol: str) -> Tuple[int, int]:
        """处理并插入数据，返回 (插入数，跳过数)"""
        if df is None or df.empty:
            return 0, 0
            
        # 数据转换
        df = df.copy()
        df['symbol'] = symbol
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.date
        
        # 选择需要的列
        columns = [
            'symbol', 'trade_date', 'open', 'high', 'low', 'close',
            'volume', 'amount', 'adj_factor', 'turnover_rate',
            'pre_close', 'change', 'pct_chg'
        ]
        
        # 确保所有列都存在
        for col in columns:
            if col not in df.columns:
                if col == 'turnover_rate':
                    df[col] = np.nan
                else:
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
        
    def heal_2018_data(self):
        """补充 2018 年数据"""
        logger.info("========== 开始补充 2018 年数据 ==========")
        
        # 获取已有的股票
        existing_symbols = self.get_existing_2018_symbols()
        
        # 获取 2018 年前上市的股票
        all_symbols = self.get_stocks_listed_before_2018()
        
        # 找出缺失的股票
        missing_symbols = [s for s in all_symbols if s not in existing_symbols]
        logger.info(f"发现 {len(missing_symbols)} 只缺失的股票")
        
        if not missing_symbols:
            logger.info("没有缺失的股票")
            return
            
        # 拉取缺失股票的数据
        for idx, symbol in enumerate(missing_symbols):
            progress = (idx + 1) / len(missing_symbols) * 100
            
            df = self.fetch_daily_data(symbol, "20180101", "20181231")
            
            if df is not None and not df.empty:
                inserted, skipped = self.process_and_insert(df, symbol)
                self.stats["inserted"] += inserted
                self.stats["skipped"] += skipped
                
                if (idx + 1) % 50 == 0 or idx == len(missing_symbols) - 1:
                    logger.info(
                        f"正在处理 {symbol}，已完成 {progress:.1f}% | "
                        f"累计插入：{self.stats['inserted']:,} | "
                        f"累计跳过：{self.stats['skipped']:,} | "
                        f"错误：{self.stats['errors']}"
                    )
                    
            # 延迟，避免触发限制
            if (idx + 1) % 10 == 0:
                time.sleep(0.5)
                
        logger.info("========== 2018 年数据补充完成 ==========")
        
    def verify_2018_data(self):
        """验证 2018 年数据"""
        logger.info("\n========== 2018 年数据验证 ==========")
        
        # 统计 2018 年数据
        sql = """
            SELECT COUNT(*) as total, 
                   COUNT(DISTINCT symbol) as stocks,
                   COUNT(DISTINCT trade_date) as days
            FROM stock_daily 
            WHERE YEAR(trade_date) = 2018
        """
        with self.get_connection() as conn:
            result = conn.execute(text(sql))
            row = result.fetchone()
            logger.info(f"2018 年数据总量：{row[0]:,} 条")
            logger.info(f"2018 年股票数量：{row[1]} 只")
            logger.info(f"2018 年交易天数：{row[2]} 天")
            
            # 估算应有的数据量
            expected = row[1] * row[2]
            logger.info(f"预估应有数据量：{expected:,} 条")
            

def main():
    """主函数"""
    try:
        healer = Stock2018Healer()
        healer.heal_2018_data()
        healer.verify_2018_data()
    except Exception as e:
        logger.error(f"任务执行失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()