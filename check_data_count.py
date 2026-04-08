from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('mysql+pymysql://root:123456@localhost:3306/quantitative_trading')

print('2023 stock_daily rows:', pd.read_sql("SELECT COUNT(*) as cnt FROM stock_daily WHERE trade_date BETWEEN '20230101' AND '20231231'", engine)['cnt'].values[0])
print('2024 stock_daily rows:', pd.read_sql("SELECT COUNT(*) as cnt FROM stock_daily WHERE trade_date BETWEEN '20240101' AND '20241231'", engine)['cnt'].values[0])
print('stock_fund_flow total rows:', pd.read_sql('SELECT COUNT(*) as cnt FROM stock_fund_flow', engine)['cnt'].values[0])
print('stock_fund_flow 2023 rows:', pd.read_sql("SELECT COUNT(*) as cnt FROM stock_fund_flow WHERE trade_date BETWEEN '20230101' AND '20231231'", engine)['cnt'].values[0])
print('stock_fund_flow 2024 rows:', pd.read_sql("SELECT COUNT(*) as cnt FROM stock_fund_flow WHERE trade_date BETWEEN '20240101' AND '20241231'", engine)['cnt'].values[0])