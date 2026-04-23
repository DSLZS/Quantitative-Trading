"""检查 stock_daily 和 stock_industry_daily 中的行业数据来源"""
from sqlalchemy import create_engine
import pandas as pd

# 读取.env 中的数据库配置
with open('.env', 'r') as f:
    for line in f:
        if 'DATABASE_URL' in line:
            db_url = line.strip().split('=')[1]
            break

engine = create_engine(db_url)

# 检查 stock_daily 中的 industry_code
query_daily = """
SELECT industry_code, COUNT(*) as cnt
FROM stock_daily
WHERE YEAR(trade_date) = 2020
GROUP BY industry_code
ORDER BY cnt DESC
LIMIT 20
"""

print('=== 2020 年 stock_daily.industry_code 分布 ===')
df_daily_2020 = pd.read_sql(query_daily, engine)
print(df_daily_2020)

query_daily_2024 = """
SELECT industry_code, COUNT(*) as cnt
FROM stock_daily
WHERE YEAR(trade_date) = 2024
GROUP BY industry_code
ORDER BY cnt DESC
LIMIT 20
"""

print()
print('=== 2024 年 stock_daily.industry_code 分布 ===')
df_daily_2024 = pd.read_sql(query_daily_2024, engine)
print(df_daily_2024)

# 检查 stock_industry_daily 中的 industry_code
query_industry = """
SELECT industry_code, COUNT(*) as cnt
FROM stock_industry_daily
WHERE YEAR(trade_date) = 2020
GROUP BY industry_code
ORDER BY cnt DESC
LIMIT 20
"""

print()
print('=== 2020 年 stock_industry_daily.industry_code 分布 ===')
df_ind_2020 = pd.read_sql(query_industry, engine)
print(df_ind_2020)

query_industry_2024 = """
SELECT industry_code, COUNT(*) as cnt
FROM stock_industry_daily
WHERE YEAR(trade_date) = 2024
GROUP BY industry_code
ORDER BY cnt DESC
LIMIT 20
"""

print()
print('=== 2024 年 stock_industry_daily.industry_code 分布 ===')
df_ind_2024 = pd.read_sql(query_industry_2024, engine)
print(df_ind_2024)

# 检查是否有 tushare 行业数据
query_sw = """
SELECT DISTINCT industry_code, industry_name
FROM stock_industry_daily
WHERE industry_code IS NOT NULL AND industry_code != ''
LIMIT 50
"""

print()
print('=== stock_industry_daily 中的行业代码样本 ===')
df_sw = pd.read_sql(query_sw, engine)
print(df_sw)