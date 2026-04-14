"""检查 2023 年数据质量"""
import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
engine = create_engine(DATABASE_URL)

# 检查 2023 年数据
query = text("""
    SELECT symbol, trade_date, pct_chg 
    FROM stock_daily 
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    LIMIT 20
""")

df = pd.read_sql_query(query, engine, params={'start_date': '2023-01-01', 'end_date': '2023-12-31'})
print('2023 data sample:')
print(df)
print()
print('pct_chg dtype:', df['pct_chg'].dtype)
print('pct_chg null count:', df['pct_chg'].isnull().sum())

# 检查 2023 年数据总量
count_query = text("""
    SELECT COUNT(*) as cnt, 
           COUNT(DISTINCT symbol) as stocks,
           COUNT(DISTINCT trade_date) as days
    FROM stock_daily 
    WHERE trade_date >= :start_date AND trade_date <= :end_date
""")
count_df = pd.read_sql_query(count_query, engine, params={'start_date': '2023-01-01', 'end_date': '2023-12-31'})
print('\n2023 data count:')
print(count_df)

# 检查 pct_chg 是否有值
pct_query = text("""
    SELECT 
        COUNT(*) as total,
        COUNT(pct_chg) as non_null_pct,
        AVG(pct_chg) as avg_pct,
        STD(pct_chg) as std_pct
    FROM stock_daily 
    WHERE trade_date >= :start_date AND trade_date <= :end_date
""")
pct_df = pd.read_sql_query(pct_query, engine, params={'start_date': '2023-01-01', 'end_date': '2023-12-31'})
print('\n2023 pct_chg stats:')
print(pct_df)