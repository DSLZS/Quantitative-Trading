"""数据环境探测脚本"""
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)

print("=" * 70)
print("数据环境探测")
print("=" * 70)

with engine.connect() as conn:
    # 1. 总体统计
    result = conn.execute(text('SELECT COUNT(*) as cnt, MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM stock_daily'))
    row = result.fetchone()
    print(f"总行数：{row.cnt}")
    print(f"日期范围：{row.min_date} - {row.max_date}")
    
    # 2. 按年份统计
    for year in [2022, 2023, 2024, 2025]:
        result = conn.execute(text(f"SELECT COUNT(*) as cnt FROM stock_daily WHERE trade_date BETWEEN '{year}0101' AND '{year}1231'"))
        row = result.fetchone()
        print(f"{year}年数据行数：{row.cnt}")
    
    # 3. 每日股票数统计
    result = conn.execute(text("""
        SELECT trade_date, COUNT(*) as stock_count 
        FROM stock_daily 
        GROUP BY trade_date 
        ORDER BY trade_date DESC 
        LIMIT 5
    """))
    print("\n最近 5 个交易日股票数:")
    for row in result:
        print(f"  {row.trade_date}: {row.stock_count}只")
    
    # 4. 检查字段完整性
    result = conn.execute(text("""
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'stock_daily'
        ORDER BY ORDINAL_POSITION
    """))
    print("\n字段列表:")
    for row in result:
        print(f"  {row.COLUMN_NAME}: {row.DATA_TYPE}")

print("=" * 70)