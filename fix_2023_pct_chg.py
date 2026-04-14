#!/usr/bin/env python
"""修复 2023 年数据的 pct_chg 字段（使用 close 价格重新计算）"""

from dotenv import load_dotenv
load_dotenv()

import os
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
engine = create_engine(DATABASE_URL)

print("正在加载 2023 年数据...")

# 加载 2023 年数据（主键是 symbol + trade_date）
query = text("""
    SELECT symbol, trade_date, close, pre_close
    FROM stock_daily 
    WHERE trade_date BETWEEN '2023-01-01' AND '2023-12-31'
    ORDER BY symbol, trade_date
""")

chunks = []
for chunk in pd.read_sql_query(query, engine, chunksize=50000):
    chunks.append(chunk)
    print(f"  Loaded chunk: {len(chunk):,} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"Total loaded: {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")

# 按 symbol 分组，计算 pct_chg
print("Calculating pct_chg from close prices...")

# pct_chg = (close - pre_close) / pre_close * 100
results = []

for symbol in df['symbol'].unique():
    group = df[df['symbol'] == symbol].sort_values('trade_date').copy()
    
    # 对于 pre_close 缺失的情况，使用前一日 close
    if group['pre_close'].isna().any() or (group['pre_close'] == 0).any():
        group['pre_close'] = group['pre_close'].fillna(group['close'].shift(1))
        group.loc[group['pre_close'] == 0, 'pre_close'] = group['close'].shift(1)
    
    # 计算 pct_chg
    group['pct_chg'] = (group['close'] - group['pre_close']) / group['pre_close'] * 100
    results.append(group)

df = pd.concat(results, ignore_index=True)

# 删除 NaN
df = df.dropna(subset=['pct_chg'])

print(f"Valid pct_chg rows: {len(df):,}")
print(f"Columns after calc: {df.columns.tolist()}")

# 更新数据库 - 使用 bulk 方式
print("Updating database...")

# 准备更新数据
updates = []
for idx in range(len(df)):
    row = df.iloc[idx]
    updates.append({
        'symbol': str(row['symbol']),
        'trade_date': row['trade_date'],
        'pct_chg': float(row['pct_chg'])
    })

print(f"Preparing to update {len(updates):,} rows...")

# 分批更新
batch_size = 10000
updated_count = 0

with engine.connect() as conn:
    trans = conn.begin()
    try:
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i+batch_size]
            for item in batch:
                update_query = text("""
                    UPDATE stock_daily 
                    SET pct_chg = :pct_chg 
                    WHERE symbol = :symbol AND trade_date = :trade_date
                """)
                conn.execute(update_query, item)
            updated_count += len(batch)
            print(f"  Updated {updated_count:,} / {len(updates):,} rows...")
        
        trans.commit()
        print(f"更新完成！共更新 {updated_count:,} 行")
    except Exception as e:
        print(f"更新失败：{e}")
        trans.rollback()