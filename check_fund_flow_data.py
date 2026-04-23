"""检查 stock_fund_flow 和 stock_daily 数据质量"""
from sqlalchemy import create_engine
import pandas as pd

# 读取.env 中的数据库配置
with open('.env', 'r') as f:
    for line in f:
        if 'DATABASE_URL' in line:
            db_url = line.strip().split('=')[1]
            break

engine = create_engine(db_url)

# 先检查表结构
query_desc = """
SHOW COLUMNS FROM stock_fund_flow
"""
print('=== stock_fund_flow 表结构 ===')
df_desc = pd.read_sql(query_desc, engine)
print(df_desc)

# 检查 stock_daily 表结构
query_daily_desc = """
SHOW COLUMNS FROM stock_daily
"""
print()
print('=== stock_daily 表结构 ===')
df_daily_desc = pd.read_sql(query_daily_desc, engine)
print(df_daily_desc)

# 检查 2020 年和 2024 年的 fund_flow 数据（使用正确的列名）
query_fund_2020 = """
SELECT symbol, trade_date, net_main_amount, net_main_rate
FROM stock_fund_flow
WHERE YEAR(trade_date) = 2020
LIMIT 10
"""

query_fund_2024 = """
SELECT symbol, trade_date, net_main_amount, net_main_rate
FROM stock_fund_flow
WHERE YEAR(trade_date) = 2024
LIMIT 10
"""

print()
print('=== 2020 年 stock_fund_flow 样本 ===')
df_fund_2020 = pd.read_sql(query_fund_2020, engine)
print(df_fund_2020)
if 'net_main_amount' in df_fund_2020.columns:
    print(f'net_main_amount 统计：min={df_fund_2020["net_main_amount"].min()}, max={df_fund_2020["net_main_amount"].max()}, mean={df_fund_2020["net_main_amount"].mean()}')

print()
print('=== 2024 年 stock_fund_flow 样本 ===')
df_fund_2024 = pd.read_sql(query_fund_2024, engine)
print(df_fund_2024)
if 'net_main_amount' in df_fund_2024.columns:
    print(f'net_main_amount 统计：min={df_fund_2024["net_main_amount"].min()}, max={df_fund_2024["net_main_amount"].max()}, mean={df_fund_2024["net_main_amount"].mean()}')

# 检查 stock_daily 中的 amount/volume 数据
query_daily_2020 = """
SELECT symbol, trade_date, amount, volume
FROM stock_daily
WHERE YEAR(trade_date) = 2020
LIMIT 10
"""

query_daily_2024 = """
SELECT symbol, trade_date, amount, volume
FROM stock_daily
WHERE YEAR(trade_date) = 2024
LIMIT 10
"""

print()
print('=== 2020 年 stock_daily 样本 ===')
df_daily_2020 = pd.read_sql(query_daily_2020, engine)
print(df_daily_2020)
if 'amount' in df_daily_2020.columns:
    print(f'amount 统计：min={df_daily_2020["amount"].min()}, max={df_daily_2020["amount"].max()}, mean={df_daily_2020["amount"].mean()}')
if 'volume' in df_daily_2020.columns:
    print(f'volume 统计：min={df_daily_2020["volume"].min()}, max={df_daily_2020["volume"].max()}, mean={df_daily_2020["volume"].mean()}')

print()
print('=== 2024 年 stock_daily 样本 ===')
df_daily_2024 = pd.read_sql(query_daily_2024, engine)
print(df_daily_2024)
if 'amount' in df_daily_2024.columns:
    print(f'amount 统计：min={df_daily_2024["amount"].min()}, max={df_daily_2024["amount"].max()}, mean={df_daily_2024["amount"].mean()}')
if 'volume' in df_daily_2024.columns:
    print(f'volume 统计：min={df_daily_2024["volume"].min()}, max={df_daily_2024["volume"].max()}, mean={df_daily_2024["volume"].mean()}')

# 检查行业数据
query_industry_2020 = """
SELECT industry_name, COUNT(*) as cnt
FROM stock_industry_daily
WHERE YEAR(trade_date) = 2020
GROUP BY industry_name
ORDER BY cnt DESC
LIMIT 10
"""

query_industry_2024 = """
SELECT industry_name, COUNT(*) as cnt
FROM stock_industry_daily
WHERE YEAR(trade_date) = 2024
GROUP BY industry_name
ORDER BY cnt DESC
LIMIT 10
"""

print()
print('=== 2020 年行业分布 ===')
df_ind_2020 = pd.read_sql(query_industry_2020, engine)
print(df_ind_2020)

print()
print('=== 2024 年行业分布 ===')
df_ind_2024 = pd.read_sql(query_industry_2024, engine)
print(df_ind_2024)

# 检查 2020 年 fund_flow 总量
query_count = """
SELECT YEAR(trade_date) as year, COUNT(*) as cnt
FROM stock_fund_flow
GROUP BY YEAR(trade_date)
ORDER BY year
"""
print()
print('=== 每年 fund_flow 记录数 ===')
df_count = pd.read_sql(query_count, engine)
print(df_count)
