"""检查行业数据映射"""
from src.db_manager import DatabaseManager

db = DatabaseManager.get_instance()

# 检查 stock_daily 表中的 industry_code 字段
print('=== stock_daily 表中的 industry_code 字段 ===')
df = db.read_sql('SELECT DISTINCT industry_code FROM stock_daily WHERE industry_code IS NOT NULL LIMIT 30')
print(df)

print()
print('=== stock_industry_daily 表中的 industry_name 字段 ===')
df2 = db.read_sql('SELECT DISTINCT industry_name FROM stock_industry_daily WHERE industry_name IS NOT NULL LIMIT 30')
print(df2)

print()
print('=== 检查是否有"银行"相关的行业 ===')
df3 = db.read_sql("SELECT DISTINCT industry_name FROM stock_industry_daily WHERE industry_name LIKE '%银行%' LIMIT 10")
print(df3)

print()
print('=== 检查 stock_info 表中的行业数据 ===')
df4 = db.read_sql('SELECT DISTINCT industry_name FROM stock_info WHERE industry_name IS NOT NULL LIMIT 30')
print(df4)