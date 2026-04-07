#!/usr/bin/env python3
"""调试 backtest_referee 的收益计算"""

import sys
sys.path.insert(0, 'src')

from alpha_research_v156 import get_alpha_research as get_alpha_156
from engine.backtest_referee import get_backtest_referee
from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# 加载 2024 年数据
db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)
query = text("""
    SELECT symbol, trade_date, open, high, low, close, pre_close,
           `change`, pct_chg, volume, amount, turnover_rate, total_mv
    FROM stock_daily
    WHERE trade_date BETWEEN '20240101' AND '20241231'
    ORDER BY symbol, trade_date
""")
print("Loading data...")
df = pd.read_sql_query(query, engine)
print(f'Loaded {len(df)} rows')

# V156
print("\n=== V156 ===")
alpha_156 = get_alpha_156()
result_156 = alpha_156.compute_score(df.copy())
print(f'Score stats: mean={result_156["score"].mean():.6f}, std={result_156["score"].std():.6f}')
print(f't1_return stats: mean={result_156["t1_return"].mean():.6f}, std={result_156["t1_return"].std():.6f}')
print(f't1_return null count: {result_156["t1_return"].isna().sum()}')
print(f't1_return zero count: {(result_156["t1_return"] == 0).sum()}')

referee_156 = get_backtest_referee(alpha_156)
signals = referee_156.generate_signals(result_156)
print(f'Signals: {signals["signal"].sum()} positions')

# 调试合并
returns = result_156[['symbol', 'trade_date', 't1_return']].copy()
print(f'\nReturns shape: {returns.shape}')
print(f'Returns t1_return null: {returns["t1_return"].isna().sum()}')
print(f'Returns t1_return zero: {(returns["t1_return"] == 0).sum()}')

# 模拟 backtest 合并
merged = signals.merge(
    returns[['symbol', 'trade_date', 't1_return']],
    on=['symbol', 'trade_date'],
    how='left'
)
print(f'\nMerged shape: {merged.shape}')

# 检查有信号的行
signal_positions = merged[merged['signal'] == 1]
print(f'Signal positions: {len(signal_positions)}')
print(f'Merged columns: {merged.columns.tolist()}')
# 检查 t1_return 是否在合并后的列中
t1_col = 't1_return' if 't1_return' in merged.columns else 't1_return_x' if 't1_return_x' in merged.columns else None
if t1_col:
    print(f'Signal positions t1_return null: {signal_positions[t1_col].isna().sum()}')
    print(f'Signal positions t1_return mean: {signal_positions[t1_col].mean():.6f}')
    print(f'Signal positions t1_return std: {signal_positions[t1_col].std():.6f}')
else:
    print('t1_return column not found in merged DataFrame!')

# 检查日期匹配
unique_dates_signals = sorted(signals['trade_date'].unique())
unique_dates_returns = sorted(returns['trade_date'].unique())
print(f'\nUnique dates in signals: {len(unique_dates_signals)}')
print(f'Unique dates in returns: {len(unique_dates_returns)}')

# 检查日期交集
dates_in_both = set(unique_dates_signals) & set(unique_dates_returns)
print(f'Dates in both: {len(dates_in_both)}')

# 检查第一天的数据
first_date = unique_dates_signals[0]
print(f'\nFirst date: {first_date}')
first_day_signals = signals[signals['trade_date'] == first_date]
first_day_returns = returns[returns['trade_date'] == first_date]
print(f'First day signals: {len(first_day_signals)}')
print(f'First day returns: {len(first_day_returns)}')
first_day_merged = first_day_signals.merge(
    returns[['symbol', 'trade_date', 't1_return']],
    on=['symbol', 'trade_date'],
    how='left'
)
print(f'First day merged: {len(first_day_merged)}')
first_day_positions = first_day_merged[first_day_merged['signal'] == 1]
print(f'First day positions: {len(first_day_positions)}')
if len(first_day_positions) > 0:
    print(f'First day positions t1_return mean: {first_day_positions["t1_return"].mean():.6f}')