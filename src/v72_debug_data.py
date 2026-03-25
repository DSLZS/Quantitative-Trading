"""V72 数据调试脚本"""
import sys
sys.path.insert(0, 'src')
from db_manager import get_db

db = get_db()

# 检查 stock_daily 的 symbol 格式
query = "SELECT DISTINCT symbol FROM stock_daily LIMIT 10"
df = db.read_sql(query)
print("stock_daily symbol 示例:")
print(df['symbol'].to_list())

# 检查 stock_fund_flow 的 symbol 格式
query = "SELECT DISTINCT symbol FROM stock_fund_flow LIMIT 10"
df = db.read_sql(query)
print("stock_fund_flow symbol 示例:")
print(df['symbol'].to_list())

# 检查 2024-01-02 两个表的数据量
query = "SELECT COUNT(*) as cnt FROM stock_daily WHERE trade_date = '2024-01-02'"
df = db.read_sql(query)
print(f"stock_daily 2024-01-02 数据量：{df['cnt'][0]}")

query = "SELECT COUNT(*) as cnt FROM stock_fund_flow WHERE trade_date = '2024-01-02'"
df = db.read_sql(query)
print(f"stock_fund_flow 2024-01-02 数据量：{df['cnt'][0]}")

# 检查 stock_daily 的日期范围和每个日期的股票数量
query = """
SELECT trade_date, COUNT(*) as cnt 
FROM stock_daily 
GROUP BY trade_date 
ORDER BY trade_date
LIMIT 30
"""
df = db.read_sql(query)
print('stock_daily 前 30 个交易日的股票数量:')
print(df.to_string())

# 检查 2024 年的数据
query = """
SELECT trade_date, COUNT(*) as cnt 
FROM stock_daily 
WHERE trade_date >= '2024-01-01' AND trade_date <= '2024-12-31'
GROUP BY trade_date 
ORDER BY trade_date
LIMIT 30
"""
df = db.read_sql(query)
print('stock_daily 2024 年前 30 个交易日的股票数量:')
print(df.to_string())
