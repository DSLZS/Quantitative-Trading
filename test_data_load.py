"""快速数据加载测试脚本 - Python 端合并"""
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import pandas as pd
import time

DATABASE_URL = 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading'
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_pre_ping=True)

start = time.time()

# 1. 先加载 stock_daily 数据
print('Loading stock_daily...')
query_daily = text("""
    SELECT 
        trade_date, symbol, 
        open, high, low, close, pre_close,
        pct_chg, volume, amount, turnover_rate
    FROM stock_daily
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date, symbol
""")
df_daily = pd.read_sql(query_daily, engine, params={'start_date': '2017-01-01', 'end_date': '2024-12-31'})
print(f'Loaded {len(df_daily):,} rows from stock_daily')

# 2. 加载 industry 数据
print('Loading stock_industry_daily...')
query_industry = text("""
    SELECT trade_date, symbol, industry_code, industry_name
    FROM stock_industry_daily
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date, symbol
""")
df_industry = pd.read_sql(query_industry, engine, params={'start_date': '2017-01-01', 'end_date': '2024-12-31'})
print(f'Loaded {len(df_industry):,} rows from stock_industry_daily')

# 3. 加载 fund_flow 数据
print('Loading stock_fund_flow...')
query_fund = text("""
    SELECT trade_date, symbol, net_main_amount, net_main_rate
    FROM stock_fund_flow
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date, symbol
""")
df_fund = pd.read_sql(query_fund, engine, params={'start_date': '2017-01-01', 'end_date': '2024-12-31'})
print(f'Loaded {len(df_fund):,} rows from stock_fund_flow')

# 4. 在 Python 中进行 LEFT JOIN
print('Merging data in Python...')
df = df_daily.merge(df_industry, on=['trade_date', 'symbol'], how='left')
df = df.merge(df_fund, on=['trade_date', 'symbol'], how='left')

# 5. 填充缺失值
df['industry_code'] = df['industry_code'].fillna('UNKNOWN')
df['industry_name'] = df['industry_name'].fillna('Unknown')
df['net_main_amount'] = df['net_main_amount'].fillna(0)
df['net_main_rate'] = df['net_main_rate'].fillna(0)

elapsed = time.time() - start
print(f'\\nLoaded {len(df):,} rows in {elapsed:.2f} seconds')
print(f'Date range: {df["trade_date"].min()} to {df["trade_date"].max()}')
print(f'Unique symbols: {df["symbol"].nunique()}')
print(f'industry_code NULL count: {df["industry_code"].isna().sum()}')
print(f'industry_name NULL count: {df["industry_name"].isna().sum()}')
engine.dispose()
