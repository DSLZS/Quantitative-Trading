"""Check 2025 data completeness."""
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)

print("=" * 60)
print("2025 年数据完整性检查")
print("=" * 60)

# 检查 2025 年数据
query = text("""
    SELECT COUNT(*) as cnt, trade_date 
    FROM stock_daily 
    WHERE trade_date >= '20250101' 
    GROUP BY trade_date 
    ORDER BY trade_date
""")
df = pd.read_sql_query(query, engine)

print(f"总行数：{df['cnt'].sum()}")
print(f"交易日期数：{len(df)}")
print(f"日期范围：{df['trade_date'].min()} - {df['trade_date'].max()}")
print(f"平均每日股票数：{df['cnt'].mean():.0f}")
print(f"最小每日股票数：{df['cnt'].min()}")

# 检查 2023-2025 全量数据
print("\n" + "=" * 60)
print("2023-2025 全量数据检查")
print("=" * 60)

query = text("""
    SELECT 
        SUBSTRING(trade_date, 1, 4) as year,
        COUNT(*) as cnt,
        COUNT(DISTINCT trade_date) as trade_days,
        COUNT(DISTINCT symbol) as symbols
    FROM stock_daily 
    WHERE trade_date >= '20230101' 
    GROUP BY SUBSTRING(trade_date, 1, 4)
    ORDER BY year
""")
df_year = pd.read_sql_query(query, engine)
print(df_year.to_string())