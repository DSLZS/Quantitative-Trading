import pandas as pd

df = pd.read_parquet('data/parquet/features_latest.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Date range: {df["trade_date"].min()} to {df["trade_date"].max()}')