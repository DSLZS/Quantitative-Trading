"""测试单条数据插入"""
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import tushare as ts

load_dotenv()

DB_URL = os.getenv('DATABASE_URL')
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

engine = create_engine(DB_URL, pool_pre_ping=True)

# 获取一天数据
trade_date = '20250102'
print(f"获取 {trade_date} 数据...")

df = pro.daily(trade_date=trade_date)
print(f"获取到 {len(df)} 条数据")
print(f"列：{df.columns.tolist()}")
print(f"前 3 行:")
print(df.head(3))

# 处理数据
df = df.rename(columns={'ts_code': 'symbol', 'vol': 'volume'})
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

# 添加缺失列 (包括 adj_factor, turnover_rate)
for col in ['adj_factor', 'turnover_rate', 'industry_code', 'total_mv', 'is_st']:
    if col not in df.columns:
        if col == 'industry_code':
            df[col] = ''
        elif col in ['total_mv', 'is_st']:
            df[col] = 0
        else:
            df[col] = None

target_columns = [
    'symbol', 'trade_date', 'open', 'high', 'low', 'close',
    'pre_close', 'change', 'pct_chg', 'volume', 'amount',
    'adj_factor', 'turnover_rate', 'industry_code', 'total_mv', 'is_st'
]

df = df[target_columns]
print(f"\n处理后列：{df.columns.tolist()}")
print(f"前 3 行:")
print(df.head(3))

# 尝试插入
print("\n尝试插入数据库...")
try:
    with engine.connect() as conn:
        chunk = df.iloc[:5]  # 只插入前 5 条
        chunk.to_sql('stock_daily', conn, if_exists='append', index=False)
        print("成功插入 5 条数据!")
except Exception as e:
    print(f"错误：{type(e).__name__}: {e}")