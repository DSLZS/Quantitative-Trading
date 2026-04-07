#!/usr/bin/env python3
"""对比 V156 和 V157 的信号差异"""

import sys
sys.path.insert(0, 'src')

from alpha_research_v156 import get_alpha_research as get_alpha_156
from alpha_research_v157 import get_alpha_research as get_alpha_157
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
print(f'Score range: [{result_156["score"].min():.6f}, {result_156["score"].max():.6f}]')
print(f't1_return stats: mean={result_156["t1_return"].mean():.6f}, std={result_156["t1_return"].std():.6f}')
print(f'Unique dates: {len(result_156["trade_date"].unique())}')

referee_156 = get_backtest_referee(alpha_156)
signals_156 = referee_156.generate_signals(result_156)
print(f'Signals: {signals_156["signal"].sum()} positions')
print(f'Signal distribution: {signals_156["signal"].value_counts().to_dict()}')

# V157
print("\n=== V157 ===")
alpha_157 = get_alpha_157()
result_157 = alpha_157.compute_score(df.copy())
print(f'Score stats: mean={result_157["score"].mean():.6f}, std={result_157["score"].std():.6f}')
print(f'Score range: [{result_157["score"].min():.6f}, {result_157["score"].max():.6f}]')
print(f't1_return stats: mean={result_157["t1_return"].mean():.6f}, std={result_157["t1_return"].std():.6f}')
print(f'Unique dates: {len(result_157["trade_date"].unique())}')

referee_157 = get_backtest_referee(alpha_157)
signals_157 = referee_157.generate_signals(result_157)
print(f'Signals: {signals_157["signal"].sum()} positions')
print(f'Signal distribution: {signals_157["signal"].value_counts().to_dict()}')

# 对比
print("\n=== COMPARISON ===")
print(f"V156 positions: {signals_156['signal'].sum()}")
print(f"V157 positions: {signals_157['signal'].sum()}")