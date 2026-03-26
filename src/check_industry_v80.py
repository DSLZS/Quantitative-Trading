"""检查行业数据覆盖"""
from src.db_manager import DatabaseManager

db = DatabaseManager()

# 检查行业映射股票数量
df = db.read_sql('SELECT COUNT(DISTINCT symbol) as cnt FROM stock_industry_daily')
print(f"行业映射股票数量：{df['cnt'][0]}")

# 检查行业数据日期范围
df = db.read_sql('SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date FROM stock_industry_daily')
print(f"行业数据日期范围：{df['min_date'][0]} 到 {df['max_date'][0]}")

# 检查 2024 年行业数据覆盖
df = db.read_sql("SELECT COUNT(DISTINCT symbol) as cnt FROM stock_industry_daily WHERE trade_date >= '2024-01-01' AND trade_date <= '2024-12-31'")
print(f"2024 年行业数据覆盖股票数量：{df['cnt'][0]}")

# 检查 stock_daily 中的 industry_code 字段
df = db.read_sql("SELECT symbol, industry_code FROM stock_daily WHERE trade_date = '2024-01-02' LIMIT 10")
print("\nstock_daily 中的 industry_code 字段:")
print(df)