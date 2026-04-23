"""测试 V207 特征计算 - 使用 Python 端 merge 避免字符集冲突"""
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os

load_dotenv()

DB_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
engine = create_engine(DB_URL)

print("=" * 60)
print("测试 V207 特征计算")
print("=" * 60)

# 加载 2020 年数据
start_date = '2020-01-01'
end_date = '2020-01-31'

# 分别加载三张表，在 Python 中 merge
print("\n1. 加载 stock_daily...")
query_daily = text("""
    SELECT 
        trade_date, symbol, 
        open, high, low, close, pre_close,
        pct_chg, volume, amount, turnover_rate
    FROM stock_daily
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date, symbol
""")
df_daily = pd.read_sql(query_daily, engine, params={'start_date': start_date, 'end_date': end_date})
print(f"   Loaded {len(df_daily):,} rows")

print("\n2. 加载 stock_industry_daily...")
query_industry = text("""
    SELECT trade_date, symbol, industry_code, industry_name
    FROM stock_industry_daily
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date, symbol
""")
df_industry = pd.read_sql(query_industry, engine, params={'start_date': start_date, 'end_date': end_date})
print(f"   Loaded {len(df_industry):,} rows")

print("\n3. 加载 stock_fund_flow...")
query_fund = text("""
    SELECT trade_date, symbol, net_main_amount, net_main_rate
    FROM stock_fund_flow
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date, symbol
""")
df_fund = pd.read_sql(query_fund, engine, params={'start_date': start_date, 'end_date': end_date})
print(f"   Loaded {len(df_fund):,} rows")

# Python 端 merge
print("\n4. Python 端 merge...")
df = df_daily.merge(df_industry, on=['trade_date', 'symbol'], how='left')
df = df.merge(df_fund, on=['trade_date', 'symbol'], how='left')

# 填充缺失值
df['industry_code'] = df['industry_code'].fillna('UNKNOWN')
df['industry_name'] = df['industry_name'].fillna('Unknown')
df['net_main_amount'] = df['net_main_amount'].fillna(0)
df['net_main_rate'] = df['net_main_rate'].fillna(0)

print(f"   Merged: {len(df):,} rows")

# 数据类型转换
df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
df['symbol'] = df['symbol'].astype(str)

# 检查行业分布
print(f"\n5. 行业分布 (industry_name):")
print(df['industry_name'].value_counts())

print(f"\n6. 行业分布 (industry_code):")
print(df['industry_code'].value_counts())

# 检查 amount 和 volume 是否有 0 值或 NULL
print(f"\n7. 数据质量检查:")
print(f"   Amount 为 0 的数量：{(df['amount'] == 0).sum()}")
print(f"   Volume 为 0 的数量：{(df['volume'] == 0).sum()}")
print(f"   Amount 为 NULL 的数量：{df['amount'].isna().sum()}")
print(f"   Volume 为 NULL 的数量：{df['volume'].isna().sum()}")

# 检查样本数据
print(f"\n8. 样本数据 (前 10 行):")
print(df[['trade_date', 'symbol', 'amount', 'volume', 'close', 'pct_chg']].head(10).to_string())

# 计算订单流不平衡
print(f"\n9. 计算订单流不平衡...")
df = df.sort_values(['symbol', 'trade_date'])

# 使用向量化计算
df['amount_change'] = df.groupby('symbol')['amount'].pct_change(10).fillna(0)
df['volume_change'] = df.groupby('symbol')['volume'].pct_change(10).fillna(0)
df['order_imbalance'] = df['amount_change'] - df['volume_change']

print(f"\n10. 订单流不平衡统计:")
print(f"   Mean: {df['order_imbalance'].mean():.6f}")
print(f"   Std:  {df['order_imbalance'].std():.6f}")
print(f"   Min:  {df['order_imbalance'].min():.6f}")
print(f"   Max:  {df['order_imbalance'].max():.6f}")
print(f"   Null: {df['order_imbalance'].isna().sum()}")

# 检查 amount 和 volume 的变动率
print(f"\n11. Amount 变动率统计:")
print(f"   Mean: {df['amount_change'].mean():.6f}")
print(f"   Std:  {df['amount_change'].std():.6f}")
print(f"   Min:  {df['amount_change'].min():.6f}")
print(f"   Max:  {df['amount_change'].max():.6f}")

print(f"\n12. Volume 变动率统计:")
print(f"   Mean: {df['volume_change'].mean():.6f}")
print(f"   Std:  {df['volume_change'].std():.6f}")
print(f"   Min:  {df['volume_change'].min():.6f}")
print(f"   Max:  {df['volume_change'].max():.6f}")

# 检查 order_imbalance_rank
print(f"\n13. 计算 order_imbalance_rank...")
df['order_imbalance_rank'] = df.groupby('trade_date')['order_imbalance'].transform(
    lambda x: x.rank(pct=True).fillna(0.5)
).fillna(0.5)

print(f"   Mean: {df['order_imbalance_rank'].mean():.4f}")
print(f"   Std:  {df['order_imbalance_rank'].std():.4f}")
print(f"   Min:  {df['order_imbalance_rank'].min():.4f}")
print(f"   Max:  {df['order_imbalance_rank'].max():.4f}")

# 检查行业内强度
print(f"\n14. 计算行业内强度...")
# 按行业分组计算截面排名
df['intra_industry_strength'] = 0.5
for industry in df['industry_name'].unique():
    industry_mask = df['industry_name'] == industry
    industry_data = df.loc[industry_mask]
    
    if len(industry_data) < 5:
        continue
    
    # 计算该行业内股票的截面排名
    ranks = industry_data['pct_chg'].rank(pct=True, method='average')
    df.loc[industry_mask, 'intra_industry_strength'] = ranks.values

print(f"   Mean: {df['intra_industry_strength'].mean():.4f}")
print(f"   Std:  {df['intra_industry_strength'].std():.4f}")
print(f"   Min:  {df['intra_industry_strength'].min():.4f}")
print(f"   Max:  {df['intra_industry_strength'].max():.4f}")

# 检查每个行业的股票数量
print(f"\n15. 每个行业的股票数量统计:")
industry_stats = df.groupby('industry_name')['symbol'].nunique()
print(f"   Min stocks per industry: {industry_stats.min()}")
print(f"   Max stocks per industry: {industry_stats.max()}")
print(f"   Mean stocks per industry: {industry_stats.mean():.1f}")

print(f"\n16. 每日股票数量统计:")
daily_stats = df.groupby('trade_date')['symbol'].nunique()
print(f"   Min stocks per day: {daily_stats.min()}")
print(f"   Max stocks per day: {daily_stats.max()}")
print(f"   Mean stocks per day: {daily_stats.mean():.1f}")

# 检查时空交互特征
print(f"\n17. 计算时空交互特征...")
df['spatiotemporal_interaction'] = df['order_imbalance_rank'] * df['intra_industry_strength']

print(f"   Mean: {df['spatiotemporal_interaction'].mean():.4f}")
print(f"   Std:  {df['spatiotemporal_interaction'].std():.4f}")
print(f"   Min:  {df['spatiotemporal_interaction'].min():.4f}")
print(f"   Max:  {df['spatiotemporal_interaction'].max():.4f}")

print("\n测试完成!")