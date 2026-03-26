"""检查 V80 回测数据"""
from src.db_manager import DatabaseManager

db = DatabaseManager()

# 检查股票数量
df = db.read_sql('SELECT COUNT(DISTINCT symbol) as cnt FROM stock_daily')
print(f"股票数量：{df['cnt'][0]}")

# 检查数据分布
df = db.read_sql('SELECT symbol, COUNT(*) as cnt FROM stock_daily GROUP BY symbol ORDER BY cnt DESC LIMIT 20')
print("\n前 20 只股票数据行数:")
print(df)

# 检查 2023 年数据
df = db.read_sql("SELECT COUNT(DISTINCT symbol) as cnt FROM stock_daily WHERE trade_date >= '2023-01-01' AND trade_date <= '2023-12-31'")
print(f"\n2023 年股票数量：{df['cnt'][0]}")

# 检查 2024 年数据
df = db.read_sql("SELECT COUNT(DISTINCT symbol) as cnt FROM stock_daily WHERE trade_date >= '2024-01-01' AND trade_date <= '2024-12-31'")
print(f"2024 年股票数量：{df['cnt'][0]}")

# 检查 2025 年数据
df = db.read_sql("SELECT COUNT(DISTINCT symbol) as cnt FROM stock_daily WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'")
print(f"2025 年股票数量：{df['cnt'][0]}")