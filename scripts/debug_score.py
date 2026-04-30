"""Debug script to understand why score std=0.0000"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest_engine import BacktestEngine
from src.lightgbm_model import LightGBMAlphaModel
import numpy as np

# Load data
engine = BacktestEngine(output_dir="reports")
years = [2020]
data = engine.load_data(years, warmup_year=2019, warmup_days=60)

print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
print(f"Date range: {data['trade_date'].min()} - {data['trade_date'].max()}")
print(f"Symbols: {data['symbol'].nunique()}")

# Create model and engineer features
model = LightGBMAlphaModel()
df = model._engineer_features(data.copy())

feature_cols = model._get_feature_columns()
print(f"\nFeature columns: {feature_cols}")

# Check feature statistics
for col in feature_cols:
    if col in df.columns:
        non_nan = df[col].notna().sum()
        mean_val = df[col].mean()
        std_val = df[col].std()
        unique_count = df[col].nunique()
        print(f"  {col}: non_nan={non_nan}, mean={mean_val:.6f}, std={std_val:.6f}, unique={unique_count}")

# Compute feature-weighted score
score = model._compute_feature_weighted_score(df)
print(f"\nScore stats:")
print(f"  mean={score.mean():.6f}")
print(f"  std={score.std():.6f}")
print(f"  unique={score.nunique()}")
print(f"  min={score.min():.6f}")
print(f"  max={score.max():.6f}")

# Check per-date score statistics
for date in sorted(df['trade_date'].unique())[:5]:
    day_mask = df['trade_date'] == date
    day_score = score[day_mask]
    print(f"  Date {date}: mean={day_score.mean():.6f}, std={day_score.std():.6f}, unique={day_score.nunique()}")

# Check individual feature contributions
print("\nIndividual feature contributions:")
for col in feature_cols:
    if col in df.columns:
        vals = df[col].fillna(0)
        if col == 'vol_20d':
            contrib = (1.0 / (vals + 1e-8) * 0.3)
        elif col == 'ret_5d':
            contrib = vals * (-0.5)
        else:
            contrib = vals * 0.1
        print(f"  {col}: mean={contrib.mean():.6f}, std={contrib.std():.6f}")