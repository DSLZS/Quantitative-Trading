"""Test V118 data loading"""
import pandas as pd
from pathlib import Path

# Test Parquet loading
parquet_files = list(Path("data/parquet").glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

for pf in parquet_files:
    print(f"Loading {pf}...")
    df_temp = pd.read_parquet(pf)
    print(f"  Shape: {df_temp.shape}")
    print(f"  Columns: {df_temp.columns.tolist()[:5]}...")
    print(f"  Date dtype: {df_temp['trade_date'].dtype}")
    print(f"  Date sample: {df_temp['trade_date'].head()}")

# Now test the full loading logic
dfs = []
for pf in parquet_files:
    df_temp = pd.read_parquet(pf)
    dfs.append(df_temp)

if dfs:
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined shape: {df.shape}")
    
    if 'trade_date' in df.columns:
        # 处理日期格式 (可能是 datetime 或字符串)
        if df['trade_date'].dtype == 'datetime64[ns]':
            df['trade_date'] = df['trade_date'].dt.strftime('%Y%m%d')
        else:
            df['trade_date'] = df['trade_date'].astype(str)
        
        start_date = '20240101'
        end_date = '20241231'
        start_date_str = str(start_date).replace('-', '')
        end_date_str = str(end_date).replace('-', '')
        
        print(f"Filtering dates: {start_date_str} to {end_date_str}")
        df_filtered = df[(df['trade_date'] >= start_date_str) & (df['trade_date'] <= end_date_str)]
        print(f"Filtered shape: {df_filtered.shape}")