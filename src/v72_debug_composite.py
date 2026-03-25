#!/usr/bin/env python
"""V72 综合评分调试脚本"""

import sys
sys.path.insert(0, 'd:/PythonProject/Quantitative-Trading')

from src.db_manager import get_db
import polars as pl
from src.core.v72_logic import V72AlphaCenter, V72DataManager

db = get_db()
data_manager = V72DataManager(db=db)
alpha_center = V72AlphaCenter()

# 加载数据
fund_flow_df = data_manager.load_fund_flow_data('2024-01-02', '2024-01-05')
industry_df = data_manager.load_industry_data('2024-01-02', '2024-01-05')
stock_df = data_manager.load_stock_data('2024-01-02', '2024-01-05')

# 计算信号
result, status = alpha_center.compute_signals(stock_df, fund_flow_df, industry_df)

# 检查 2024-01-02 的数据
day_data = result.filter(pl.col('trade_date') == '2024-01-02')
print('2024-01-02 数据样例:')
print(day_data.select(['symbol', 'z_score', 'snr_value', 'snr_pass', 'industry_divergence', 'composite_score', 'buy_signal']).head(20))

# 统计
print('\n统计:')
print(f'总股票数：{day_data.height}')
print(f'snr_pass=True: {day_data.filter(pl.col("snr_pass") == True).height}')
print(f'industry_divergence=True: {day_data.filter(pl.col("industry_divergence") == True).height}')
print(f'composite_score>0: {day_data.filter(pl.col("composite_score") > 0).height}')
print(f'buy_signal=True: {day_data.filter(pl.col("buy_signal") == True).height}')

# 检查 composite_score 分布
score_data = day_data.filter(pl.col('composite_score') > 0).sort('composite_score', descending=True)
print('\ncomposite_score>0 的股票:')
print(score_data.select(['symbol', 'z_score', 'snr_value', 'composite_score', 'composite_percentile']).head(20))