"""测试 engine.py 中的数据 merge 逻辑"""
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

DB_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
engine = create_engine(DB_URL)

# 加载 2020 年数据
start_date = '2020-01-01'
end_date = '2020-01-31'

print("=" * 60)
print("测试数据 merge 逻辑")
print("=" * 60)

# 加载 stock_daily
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
print(f"\n1. stock_daily: {len(df_daily):,} rows, {df_daily['symbol'].nunique()} symbols")

# 加载 stock_industry_daily
query_industry = text("""
    SELECT trade_date, symbol, industry_code, industry_name
    FROM stock_industry_daily
    WHERE trade_date >= :start_date AND trade_date <= :end_date
    ORDER BY trade_date, symbol
""")
df_industry = pd.read_sql(query_industry, engine, params={'start_date': start_date, 'end_date': end_date})
print(f"2. stock_industry_daily: {len(df_industry):,} rows, {df_industry['symbol'].nunique()} symbols")

# 检查 industry 数据内容
print(f"\n3. stock_industry_daily 样本:")
print(df_industry[['trade_date', 'symbol', 'industry_code', 'industry_name']].head(10).to_string())

# 检查 symbol 格式
print(f"\n4. stock_daily symbol 样本: {df_daily['symbol'].head(5).tolist()}")
print(f"   stock_industry_daily symbol 样本：{df_industry['symbol'].head(5).tolist()}")

# 检查是否有 symbol 不匹配
daily_symbols = set(df_daily['symbol'].unique())
industry_symbols = set(df_industry['symbol'].unique())

only_in_daily = daily_symbols - industry_symbols
only_in_industry = industry_symbols - daily_symbols

print(f"\n5. Symbol 匹配分析:")
print(f"   - 只在 stock_daily 中：{len(only_in_daily)} symbols")
print(f"   - 只在 stock_industry_daily 中：{len(only_in_industry)} symbols")

if len(only_in_daily) > 0:
    print(f"   - 示例 (只在 daily): {list(only_in_daily)[:5]}")
if len(only_in_industry) > 0:
    print(f"   - 示例 (只在 industry): {list(only_in_industry)[:5]}")

# 执行 merge
df = df_daily.merge(df_industry, on=['trade_date', 'symbol'], how='left')
print(f"\n6. Merge 后：{len(df):,} rows")

# 检查 industry_name 是否为空
null_count = df['industry_name'].isna().sum()
print(f"   - industry_name NULL: {null_count:,} ({null_count/len(df):.2%})")

# 检查非空样本
non_null_sample = df[df['industry_name'].notna()][['trade_date', 'symbol', 'industry_code', 'industry_name']].head(5)
print(f"\n7. 非空 industry_name 样本:")
print(non_null_sample.to_string())

# 检查 NULL 样本
null_sample = df[df['industry_name'].isna()][['trade_date', 'symbol', 'industry_code', 'industry_name']].head(5)
print(f"\n8. NULL industry_name 样本:")
print(null_sample.to_string())

# 检查合并后 industry_code 和 industry_name 的分布
print(f"\n9. industry_code 分布 (前 10):")
print(df['industry_code'].value_counts().head(10))

print(f"\n10. industry_name 分布 (前 10):")
print(df['industry_name'].value_counts().head(10))

print("\n测试完成!")