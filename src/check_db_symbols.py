#!/usr/bin/env python
"""检查数据库股票数量"""

from src.db_manager import DatabaseManager

db = DatabaseManager()

# 检查总股票数
df = db.read_sql("SELECT COUNT(DISTINCT symbol) as cnt FROM stock_daily WHERE trade_date >= '2024-01-01'")
print(f'2024 年股票数量：{df["cnt"][0]}')

# 检查每日股票数
df = db.read_sql("SELECT trade_date, COUNT(DISTINCT symbol) as cnt FROM stock_daily WHERE trade_date >= '2024-01-01' GROUP BY trade_date ORDER BY trade_date LIMIT 10")
print(f'每日股票数（前 10 天）：')
print(df)

# 检查数据范围
df = db.read_sql("SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM stock_daily")
print(f'数据日期范围：{df["min_date"][0]} 至 {df["max_date"][0]}')