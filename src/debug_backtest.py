"""调试回测数据"""
import sys
sys.path.insert(0, 'd:/PythonProject/Quantitative-Trading')
from db_manager import DatabaseManager
import polars as pl

db = DatabaseManager.get_instance()

# 检查原始数据类型
data = db.read_sql("""
    SELECT symbol, trade_date, close 
    FROM stock_daily 
    WHERE trade_date >= '2024-01-01' 
    LIMIT 5
""")
print('原始数据类型:')
print(data.schema)
print('前 5 行数据:')
print(data)

# 检查日期格式
print('\ntrade_date 列类型:', type(data['trade_date'][0]))
print('trade_date 样本值:', data['trade_date'].to_list()[:5])