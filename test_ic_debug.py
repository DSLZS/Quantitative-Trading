"""
IC Debug 脚本 - 深入分析信号与收益率的关系
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
engine = create_engine(DATABASE_URL)

# 加载 signals
print("Loading signals...")
signals = pd.read_csv('signals.csv')
print(f"Loaded {len(signals)} signals")
print(f"Signals columns: {signals.columns.tolist()}")
print(f"Signals date range: {signals['trade_date'].min()} to {signals['trade_date'].max()}")
print(f"Score stats: mean={signals['score'].mean():.4f}, std={signals['score'].std():.4f}")

# 加载 market data
print("\nLoading market data...")
query = text("""
    SELECT symbol, trade_date, pct_chg, close, volume 
    FROM stock_daily 
    WHERE YEAR(trade_date) IN (2023, 2024, 2025)
    ORDER BY trade_date, symbol
""")
market = pd.read_sql_query(query, engine)
print(f"Loaded {len(market)} market rows")

# 转换日期格式
signals['trade_date'] = pd.to_datetime(signals['trade_date'])
market['trade_date'] = pd.to_datetime(market['trade_date'])

# 合并数据
print("\nMerging data...")
merged = signals.merge(market, on=['symbol', 'trade_date'], how='inner')
print(f"Merged: {len(merged)} rows")

# 计算次日收益
merged = merged.sort_values(['symbol', 'trade_date'])
merged['next_return'] = merged.groupby('symbol')['pct_chg'].shift(-1) / 100.0
merged = merged.dropna(subset=['score', 'next_return'])
print(f"Valid samples: {len(merged)}")

# 分析整体相关性
print("\n=== Overall Correlation ===")
corr = merged['score'].corr(merged['next_return'], method='spearman')
print(f"Overall Spearman IC: {corr:.6f}")

# 按年份分析
print("\n=== By Year ===")
merged['year'] = merged['trade_date'].dt.year
for year in [2023, 2024, 2025]:
    year_data = merged[merged['year'] == year]
    if len(year_data) > 0:
        ic = year_data['score'].corr(year_data['next_return'], method='spearman')
        print(f"{year}: IC={ic:.6f}, samples={len(year_data)}")

# 检查信号分数分布
print("\n=== Score Distribution ===")
print(f"Score quantiles:\n{signals['score'].describe()}")

# 检查 pct_chg 分布
print("\n=== pct_chg Distribution ===")
print(f"pct_chg stats:\n{market['pct_chg'].describe()}")

# 检查 next_return 分布
print("\n=== next_return Distribution ===")
print(f"next_return stats:\n{merged['next_return'].describe()}")

# 分析每日 IC 分布
print("\n=== Daily IC Distribution ===")
daily_ics = []
for date in merged['trade_date'].unique():
    day_data = merged[merged['trade_date'] == date]
    if len(day_data) >= 100:
        ic = day_data['score'].corr(day_data['next_return'], method='spearman')
        if not np.isnan(ic):
            daily_ics.append(ic)

print(f"Daily IC count: {len(daily_ics)}")
print(f"Daily IC mean: {np.mean(daily_ics):.6f}")
print(f"Daily IC std: {np.std(daily_ics):.6f}")
print(f"Daily IC median: {np.median(daily_ics):.6f}")
print(f"Daily IC > 0: {np.mean(np.array(daily_ics) > 0):.2%}")

# 检查前 10% 和后 10% 的收益
print("\n=== Portfolio Analysis ===")
merged['score_rank'] = pd.qcut(merged['score'], q=10, labels=False, duplicates='drop')
for rank in range(0, 10):
    rank_data = merged[merged['score_rank'] == rank]
    if len(rank_data) > 0:
        mean_ret = rank_data['next_return'].mean()
        print(f"Decile {rank}: mean_return={mean_ret:.6f}, count={len(rank_data)}")

print("\nDone!")