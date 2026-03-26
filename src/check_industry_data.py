#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查行业数据"""

from src.db_manager import DatabaseManager

db = DatabaseManager()

print("=" * 60)
print("检查 stock_info 表结构")
print("=" * 60)

df = db.read_sql("SELECT * FROM stock_info LIMIT 10")
print(df)

print("\n" + "=" * 60)
print("检查 stock_info 表列名")
print("=" * 60)

df = db.read_sql("DESCRIBE stock_info")
print(df)

print("\n" + "=" * 60)
print("检查 stock_info 中的行业信息")
print("=" * 60)

df = db.read_sql("SELECT industry_name, COUNT(*) as cnt FROM stock_info GROUP BY industry_name ORDER BY cnt DESC LIMIT 20")
print(df)