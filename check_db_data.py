#!/usr/bin/env python
"""检查数据库中的实际数据结构"""

from dotenv import load_dotenv
load_dotenv()

import os
from sqlalchemy import create_engine, text

engine = create_engine(os.getenv("DATABASE_URL"))

with engine.connect() as conn:
    # 检查表结构
    result = conn.execute(text("DESCRIBE stock_daily"))
    print("=== 表结构 ===")
    for row in result.fetchall():
        print(row)
    
    # 检查样例数据
    print("\n=== 样例数据 ===")
    result = conn.execute(text("SELECT symbol, trade_date, close, volume, amount FROM stock_daily LIMIT 5"))
    print("Columns:", result.keys())
    for row in result.fetchall():
        print(row)
    
    # 检查 2023 年数据
    print("\n=== 2023 年数据统计 ===")
    result = conn.execute(text("""
        SELECT COUNT(*), MIN(trade_date), MAX(trade_date) 
        FROM stock_daily 
        WHERE trade_date LIKE '2023%'
    """))
    for row in result.fetchall():
        print(f"Count: {row[0]}, Min date: {row[1]}, Max date: {row[2]}")
    
    # 检查是否有 NULL 值
    print("\n=== NULL 值检查 ===")
    result = conn.execute(text("""
        SELECT 
            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as close_null,
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as volume_null,
            SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) as amount_null
        FROM stock_daily 
        WHERE trade_date LIKE '2023%'
    """))
    for row in result.fetchall():
        print(f"close NULL: {row[0]}, volume NULL: {row[1]}, amount NULL: {row[2]}")