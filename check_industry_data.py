"""检查 stock_industry_daily 表中的数据"""
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

DB_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
engine = create_engine(DB_URL)
conn = engine.connect()

print("=" * 60)
print("检查 stock_industry_daily 表中的数据")
print("=" * 60)

# 按年份统计行数
print("\n1. 按年份统计行数:")
result = conn.execute(text("""
    SELECT YEAR(trade_date) as year, COUNT(*) as cnt 
    FROM stock_industry_daily 
    WHERE YEAR(trade_date) IN (2020, 2022, 2024) 
    GROUP BY YEAR(trade_date) 
    ORDER BY year
"""))
print("Year | Count")
print("-" * 20)
for row in result:
    print(f"{row[0]} | {row[1]:,}")

# 检查 2020 年数据样本
print("\n2. 2020 年数据样本 (前 10 行):")
result = conn.execute(text("""
    SELECT trade_date, symbol, industry_code, industry_name 
    FROM stock_industry_daily 
    WHERE YEAR(trade_date) = 2020 
    LIMIT 10
"""))
for row in result:
    print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")

# 检查 2022 年数据样本
print("\n3. 2022 年数据样本 (前 10 行):")
result = conn.execute(text("""
    SELECT trade_date, symbol, industry_code, industry_name 
    FROM stock_industry_daily 
    WHERE YEAR(trade_date) = 2022 
    LIMIT 10
"""))
for row in result:
    print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")

# 检查 2024 年数据样本
print("\n4. 2024 年数据样本 (前 10 行):")
result = conn.execute(text("""
    SELECT trade_date, symbol, industry_code, industry_name 
    FROM stock_industry_daily 
    WHERE YEAR(trade_date) = 2024 
    LIMIT 10
"""))
for row in result:
    print(f"  {row[0]} | {row[1]} | {row[2]} | {row[3]}")

# 检查 NULL 值情况
print("\n5. NULL 值统计:")
result = conn.execute(text("""
    SELECT 
        YEAR(trade_date) as year,
        COUNT(*) as total,
        SUM(CASE WHEN industry_code IS NULL OR industry_code = '' THEN 1 ELSE 0 END) as null_code,
        SUM(CASE WHEN industry_name IS NULL OR industry_name = '' THEN 1 ELSE 0 END) as null_name
    FROM stock_industry_daily 
    WHERE YEAR(trade_date) IN (2020, 2022, 2024)
    GROUP BY YEAR(trade_date) 
    ORDER BY year
"""))
print("Year | Total | Null Code | Null Name")
print("-" * 40)
for row in result:
    print(f"{row[0]} | {row[1]:,} | {row[2]:,} | {row[3]:,}")

conn.close()
print("\n检查完成!")